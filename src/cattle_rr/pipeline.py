import math

import cv2
import numpy as np

from .config import RRConfig
from .flow import (
    build_displacement_cube,
    initialize_target_state,
    fixed_motion_region,
    reset_local_tracklets,
    update_persistent_tracklets,
)
from .signal_processing import analyze_signal, anti_phase_align, filter_signal, safe_row_median, shared_axis


class G2LFlowRREstimator:
    """Estimate RR from one track of fixed-size abdominal frames and masks."""

    def __init__(self, config=None):
        self.config = config or RRConfig()

    def estimate(self, frames, masks, fps, frame_indices=None):
        config = self.config
        fps = float(fps)
        config.validate(fps)
        if len(frames) != len(masks) or len(frames) < 3:
            raise ValueError("Need at least three paired ROI frames and masks.")
        indices = list(range(len(frames))) if frame_indices is None else [int(i) for i in frame_indices]
        if len(indices) != len(frames):
            raise ValueError("frame_indices length must equal the number of frames.")
        if indices[0] < 0 or any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("source frame indices must be nonnegative and strictly increasing.")

        state = initialize_target_state()
        state["first_frame"], state["last_frame"] = indices[0], indices[-1]
        previous = None
        active_sizes = []
        pair_count = 0
        shape = frames[0].shape[:2]
        gray_frames, binary_masks = [], []
        for index, frame, mask in zip(indices, frames, masks):
            if frame.shape[:2] != shape or mask.shape[:2] != shape:
                raise ValueError("ROI frames and masks must have the same dimensions throughout.")
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            binary = (mask > 0).astype(np.uint8) * 255
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            gray = np.asarray(gray, dtype=np.uint8)
            gray_frames.append(gray)
            binary_masks.append(binary)

        region, pair_count, region_diagnostics, frequency = fixed_motion_region(
            gray_frames, binary_masks, indices, config, state["timing"],
        )
        for index, gray, binary in zip(indices, gray_frames, binary_masks):
            if not binary.any():
                continue

            if previous is not None:
                previous_index, previous_gray, previous_mask = previous
                gap = index - previous_index
                if gap > config.max_pair_gap:
                    reset_local_tracklets(state)
                else:
                    common = cv2.bitwise_and(previous_mask, binary)
                    local_support = cv2.bitwise_and(region, common)
                    current_support = cv2.bitwise_and(region, binary)
                    active_sizes.append(np.count_nonzero(local_support))
                    with state["timing"].measure("g2l_tracklets_s"):
                        update_persistent_tracklets(
                            state, previous_gray, gray, local_support, current_support, previous_index, gap, config,
                        )
            previous = index, gray, binary

        cube, ids = build_displacement_cube(state, config)
        axis, eigenvalues = shared_axis(cube)
        common_result = {
            "first_source_frame_index": indices[0], "last_source_frame_index": indices[-1],
            "tracklet_count": len(ids), "displacement_cube": cube,
            "g2l_region_mask": region,
            "g2l_region_frequency": frequency,
        }
        diagnostics = {
            "frames": len(frames), "fps": fps, "g2l_region_mode": "fixed_pairwise_region_union",
            "dense_flow_calls": pair_count,
            "g2l_region_selections": 1,
            "g2l_region_pixels": int(np.count_nonzero(region)),
            **region_diagnostics,
            "active_region_pixels_median": float(np.median(active_sizes)) if active_sizes else 0.0,
        }
        if axis is None:
            return {
                **common_result, "status": "insufficient_motion_tracks",
                "rr_bpm": math.nan, "peak_hz": math.nan, "snr_db": math.nan,
                "diagnostics": {
                    **diagnostics, "signal_samples": 0,
                    "failure_reason": "no_usable_motion_tracklets" if not ids else "shared_axis_unavailable",
                    "timing_seconds": dict(state["timing"].values),
                },
            }
        projected = np.einsum("tkd,d->tk", cube, axis).astype(np.float32)
        pca_signal = safe_row_median(projected)
        aligned, signs, corr, apa_diag, pair_corr, pair_overlap = anti_phase_align(projected, fps, config)
        fused = safe_row_median(aligned)
        rr = analyze_signal(fused, fps, config, state["timing"])
        filtered, _ = filter_signal(fused, fps, config)
        return {
            **common_result, "status": "ok" if rr["ok"] else "rr_estimation_failed",
            "rr_bpm": rr["rr_bpm"], "peak_hz": rr["peak_hz"], "snr_db": rr["snr_db"],
            "raw_signal": fused, "pca_signal": pca_signal, "filtered_signal": filtered,
            "shared_axis": axis, "eigenvalues": eigenvalues,
            "tracklet_signs": signs, "tracklet_alignment_correlations": corr,
            "pairwise_alignment_correlations": pair_corr, "pairwise_alignment_overlaps": pair_overlap,
            "diagnostics": {
                **apa_diag, **diagnostics, "signal_samples": rr["signal_samples"],
                "signal_valid_ratio": rr["signal_valid_ratio"], "failure_reason": None if rr["ok"] else rr["reason"],
                "timing_seconds": dict(state["timing"].values),
            },
        }

