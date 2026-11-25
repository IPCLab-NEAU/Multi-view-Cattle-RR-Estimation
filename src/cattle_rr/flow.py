from collections import defaultdict

import cv2
import numpy as np

from .runtime import TimerBook


MOTION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def motion_region(flow, common_mask, gap, config):
    """Locate the motion region for one valid observation pair."""
    supported = common_mask > 0
    if np.count_nonzero(supported) < config.min_region_pixels:
        return np.zeros_like(common_mask)
    magnitude = np.hypot(flow[..., 0], flow[..., 1]) / gap
    response = cv2.GaussianBlur(magnitude, (21, 21), 0)
    return select_motion_region(response, common_mask, config)


def select_motion_region(response, common_mask, config):
    """Apply the configured percentile and morphology to one response map."""
    supported = common_mask > 0
    if np.count_nonzero(supported) < config.min_region_pixels:
        return np.zeros_like(common_mask)
    threshold = np.percentile(response[supported], config.motion_percentile)
    region = ((response >= threshold) & supported).astype(np.uint8) * 255
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, MOTION_KERNEL)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, MOTION_KERNEL)
    region &= common_mask
    if np.count_nonzero(region) < config.min_region_pixels:
        region.fill(0)
    return region


def fixed_motion_region(gray_frames, masks, indices, config, timing):
    """Threshold each FB observation pair, then unite its region over the segment."""
    region = np.zeros_like(masks[0])
    support = np.zeros_like(masks[0])
    hits = np.zeros(gray_frames[0].shape, dtype=np.uint32)
    visibility = np.zeros_like(hits)
    previous = None
    pair_count = 0
    thresholds, pair_sizes = [], []
    for index, gray, mask in zip(indices, gray_frames, masks):
        if not mask.any():
            continue
        if previous is not None:
            previous_index, previous_gray, previous_mask = previous
            gap = index - previous_index
            common = cv2.bitwise_and(previous_mask, mask)
            observed = common > 0
            if gap <= config.max_pair_gap and np.count_nonzero(observed) >= config.min_region_pixels:
                with timing.measure("dense_flow_s"):
                    flow = cv2.calcOpticalFlowFarneback(
                        previous_gray, gray, None, config.fb_pyr_scale, config.fb_levels,
                        config.fb_window, config.fb_iterations, config.fb_poly_n, config.fb_poly_sigma, 0,
                    )
                magnitude = np.hypot(flow[..., 0], flow[..., 1]) / gap
                response = cv2.GaussianBlur(magnitude, (21, 21), 0)
                thresholds.append(float(np.percentile(response[observed], config.motion_percentile)))
                pair_region = select_motion_region(response, common, config)
                region |= pair_region
                hits += (pair_region > 0).astype(np.uint32)
                visibility += observed.astype(np.uint32)
                pair_sizes.append(int(np.count_nonzero(pair_region)))
                support |= common
                pair_count += 1
        previous = index, gray, mask
    support_pixels = int(np.count_nonzero(support))
    frequency = np.divide(hits, visibility, out=np.zeros(hits.shape, dtype=np.float32), where=visibility > 0)
    diagnostics = {
        "motion_threshold_min": min(thresholds) if thresholds else None,
        "motion_threshold_median": float(np.median(thresholds)) if thresholds else None,
        "motion_threshold_max": max(thresholds) if thresholds else None,
        "motion_threshold_evaluations": len(thresholds),
        "pair_region_pixels_median": float(np.median(pair_sizes)) if pair_sizes else 0.0,
        "abdominal_support_union_pixels": support_pixels,
        "fixed_region_fraction_of_support": float(np.count_nonzero(region) / support_pixels) if support_pixels else 0.0,
    }
    return region, pair_count, diagnostics, frequency


def initialize_target_state():
    return {
        "tracklet_events": defaultdict(list),
        "active_points": {},
        "next_tracklet_id": 0,
        "timing": TimerBook(),
    }


def reset_local_tracklets(state):
    state["active_points"].clear()


def update_persistent_tracklets(state, previous_gray, current_gray, local_mask, current_mask, frame_idx, frame_gap, config):
    """Acquire and retain points inside the fixed region's current valid support."""
    active = state["active_points"]
    needed = config.max_points - len(active)
    if needed > 0 and np.count_nonzero(local_mask) >= config.min_region_pixels:
        available = local_mask.copy()
        for point in active.values():
            cv2.circle(available, tuple(np.rint(point).astype(int)), config.min_point_distance, 0, -1)
        new_points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=needed, qualityLevel=config.quality_level,
            minDistance=config.min_point_distance, mask=available, blockSize=config.block_size,
        )
        if new_points is not None:
            for point in new_points.reshape(-1, 2):
                active[state["next_tracklet_id"]] = point
                state["next_tracklet_id"] += 1
    if not active:
        return

    points0 = np.asarray(list(active.values()), dtype=np.float32).reshape(-1, 1, 2)
    lk_options = {
        "winSize": (config.lk_window, config.lk_window),
        "maxLevel": config.lk_levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config.lk_iterations, config.lk_epsilon),
    }
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points0, None, **lk_options)
    if points1 is None or status1 is None:
        reset_local_tracklets(state)
        return
    points0_back, status2, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, points1, None, **lk_options)
    if points0_back is None or status2 is None:
        reset_local_tracklets(state)
        return

    origins, endpoints = points0.reshape(-1, 2), points1.reshape(-1, 2)
    valid = status1.ravel().astype(bool) & status2.ravel().astype(bool)
    valid &= np.linalg.norm(points0_back.reshape(-1, 2) - origins, axis=1) <= config.lk_fb_error
    h, w = current_mask.shape
    updated = {}
    for row, tracklet_id in enumerate(active):
        if not valid[row] or not np.isfinite(endpoints[row]).all():
            continue
        x, y = np.rint(endpoints[row]).astype(int)
        if 0 <= x < w and 0 <= y < h and current_mask[y, x]:
            dx, dy = (endpoints[row] - origins[row]) / frame_gap
            state["tracklet_events"][tracklet_id].append((frame_idx, float(dx), float(dy)))
            updated[tracklet_id] = endpoints[row]
    state["active_points"] = updated


def build_displacement_cube(state, config):
    first = state["first_frame"]
    count = state["last_frame"] - first + 1
    events_by_id = state["tracklet_events"]
    ids = [ident for ident, events in events_by_id.items() if len(events) >= config.min_tracklet_samples]
    ids = sorted(ids, key=lambda ident: (-len(events_by_id[ident]), ident))[:config.max_tracklets]
    cube = np.full((count, len(ids), 2), np.nan, dtype=np.float32)
    for column, ident in enumerate(ids):
        for index, dx, dy in events_by_id[ident]:
            cube[index - first, column] = (dx, dy)
    return cube, ids

