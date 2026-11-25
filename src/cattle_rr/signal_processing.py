from typing import Any
import math, warnings
import numpy as np
from scipy.signal import butter, detrend, filtfilt, medfilt
from .runtime import TimerBook

def shared_axis(displacements: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if displacements.size == 0:
        return None, None
    vectors = displacements.reshape(-1, 2)
    vectors = vectors[np.isfinite(vectors).all(axis=1)]
    if len(vectors) < 3:
        return None, None
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or float(np.max(eigenvalues)) <= 1e-12:
        return None, None
    axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-12:
        return None, None
    axis /= norm
    # Deterministic orientation only; frequency is invariant to a global sign.
    if axis[1] < 0 or (abs(float(axis[1])) < 1e-12 and axis[0] < 0):
        axis = -axis
    return axis, eigenvalues.astype(np.float32)

def column_standardized(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float32)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        if np.count_nonzero(valid) < 3:
            continue
        x = values[valid, column].astype(np.float64)
        try:
            x = detrend(x, type="linear")
        except Exception:
            x = x - np.mean(x)
        std = float(np.std(x))
        if std > 1e-12:
            result[valid, column] = ((x - np.mean(x)) / std).astype(np.float32)
    return result

def safe_row_median(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            return np.nanmedian(matrix, axis=1).astype(np.float32)

def alignment_feature_matrix(projected: np.ndarray, fps: float, args) -> np.ndarray:
    """Create drift-resistant features used only to estimate trajectory signs.

    Missing samples are interpolated only inside each tracklet's observed span.
    The reconstructed feature is band-pass filtered in the pre-specified
    respiratory band and then standardized.  The original projected signals are
    never replaced by these features; they are only multiplied by the estimated
    signs after alignment.
    """
    features = np.full_like(projected, np.nan, dtype=np.float32)
    if projected.size == 0 or not np.isfinite(fps) or fps <= 0:
        return features

    nyquist = fps / 2.0
    low = max(float(args.rr_min_hz) / nyquist, 1e-4)
    high = min(float(args.rr_max_hz) / nyquist, 0.999)
    if high <= low:
        return column_standardized(projected)
    b, a = butter(2, [low, high], btype="bandpass")
    pad_requirement = 3 * max(len(a), len(b))

    for column in range(projected.shape[1]):
        raw = projected[:, column].astype(np.float64)
        observed = np.flatnonzero(np.isfinite(raw))
        if observed.size < 3:
            continue
        first, last = int(observed[0]), int(observed[-1])
        if last - first + 1 <= pad_requirement:
            continue
        segment = raw[first : last + 1].copy()
        valid = np.isfinite(segment)
        if np.count_nonzero(valid) < 3:
            continue
        indices = np.arange(segment.size)
        segment[~valid] = np.interp(indices[~valid], indices[valid], segment[valid])
        try:
            segment = detrend(segment, type="linear")
            segment = filtfilt(b, a, segment)
        except (ValueError, np.linalg.LinAlgError):
            continue
        center = float(np.median(segment))
        scale = float(1.4826 * np.median(np.abs(segment - center)))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = float(np.std(segment))
        if not np.isfinite(scale) or scale < 1e-12:
            continue
        standardized = (segment - center) / scale
        max_z = float(args.align_max_z)
        if max_z > 0:
            standardized = np.clip(standardized, -max_z, max_z)
        # Correlations use only truly observed displacement samples.  Interpolated
        # samples help filtering at internal gaps but do not inflate overlap.
        output = np.full(segment.size, np.nan, dtype=np.float32)
        output[valid] = standardized[valid].astype(np.float32)
        features[first : last + 1, column] = output
    return features

def _connected_components(adjacency: list[list[tuple[int, float, float]]]) -> list[list[int]]:
    components: list[list[int]] = []
    unseen = set(range(len(adjacency)))
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour, _, _ in adjacency[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components

def anti_phase_align(
    projected: np.ndarray,
    fps: float,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Synchronize signs of anti-phase local trajectories without reference RR.

    A graph is built from pairwise correlations of respiratory-band features.
    Edge signs specify whether two tracklets agree or are reversed.  A
    maximum-confidence tree initializes the signs in every connected component,
    followed by weighted sign-consensus refinement.  This avoids the cancellation
    failure of using the median of unaligned trajectories as the initial template.

    The unavoidable global sign of each connected component is fixed
    deterministically by its most strongly connected anchor.  Global sign does
    not change the FFT frequency estimate.
    """
    aligned = projected.copy()
    n_tracklets = int(projected.shape[1]) if projected.ndim == 2 else 0
    signs = np.ones(n_tracklets, dtype=np.int8)
    correlations = np.full(n_tracklets, np.nan, dtype=np.float32)
    pairwise_corr = np.full((n_tracklets, n_tracklets), np.nan, dtype=np.float32)
    pairwise_overlap = np.zeros((n_tracklets, n_tracklets), dtype=np.int32)
    min_overlap = max(
        int(args.align_min_overlap),
        int(math.ceil(float(args.align_min_overlap_s) * float(fps))),
    )
    diagnostics: dict[str, Any] = {
        "status": "insufficient_tracklets" if n_tracklets < 2 else "not_evaluated",
        "min_overlap_frames": min_overlap,
        "pairwise_links": 0,
        "graph_tracklets": 0,
        "connected_components": n_tracklets,
        "largest_component": 1 if n_tracklets else 0,
        "corr_before_median": math.nan,
        "corr_after_median": math.nan,
        "agreement_gain": math.nan,
        "alignment_applied": 0,
    }
    if n_tracklets < 2:
        return aligned, signs, correlations, diagnostics, pairwise_corr, pairwise_overlap

    features = alignment_feature_matrix(projected, fps, args)
    adjacency: list[list[tuple[int, float, float]]] = [[] for _ in range(n_tracklets)]
    retained_before: list[float] = []

    # Pair-specific Pearson correlations with missing observations, computed by
    # matrix products rather than a Python O(K^2*T) loop.  S[i,j] is the sum of
    # tracklet i over frames where tracklet j is also present.
    observed = np.isfinite(features)
    mask = observed.astype(np.float64)
    values = np.where(observed, features, 0.0).astype(np.float64)
    overlaps_float = mask.T @ mask
    pairwise_overlap = np.rint(overlaps_float).astype(np.int32)
    sums = values.T @ mask
    sums_sq = np.square(values).T @ mask
    cross = values.T @ values
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance_numerator = cross - (sums * sums.T) / overlaps_float
        variance_left = sums_sq - np.square(sums) / overlaps_float
        variance_right = sums_sq.T - np.square(sums.T) / overlaps_float
        denominator = np.sqrt(np.maximum(variance_left, 0.0) * np.maximum(variance_right, 0.0))
        correlations_matrix = covariance_numerator / denominator
    eligible_for_corr = (pairwise_overlap >= min_overlap) & np.isfinite(correlations_matrix) & (denominator > 1e-12)
    pairwise_corr[eligible_for_corr] = correlations_matrix[eligible_for_corr].astype(np.float32)
    np.fill_diagonal(pairwise_corr, np.nan)
    retained = eligible_for_corr & (np.abs(correlations_matrix) >= float(args.align_min_abs_corr))
    np.fill_diagonal(retained, False)
    for left, right in np.argwhere(np.triu(retained, k=1)):
        left, right = int(left), int(right)
        overlap = int(pairwise_overlap[left, right])
        corr = float(correlations_matrix[left, right])
        # Longer overlaps are more reliable, without allowing long tracklets to
        # completely dominate correlation strength.
        weight = abs(corr) * math.sqrt(overlap / max(1, min_overlap))
        adjacency[left].append((right, corr, weight))
        adjacency[right].append((left, corr, weight))
        retained_before.append(corr)

    components = _connected_components(adjacency)
    diagnostics["pairwise_links"] = len(retained_before)
    diagnostics["graph_tracklets"] = int(sum(bool(edges) for edges in adjacency))
    diagnostics["connected_components"] = len(components)
    diagnostics["largest_component"] = max((len(c) for c in components), default=0)
    diagnostics["corr_before_median"] = (
        float(np.median(retained_before)) if retained_before else math.nan
    )

    anchors: set[int] = set()
    for component in components:
        if len(component) < 2:
            continue
        weighted_degree = {
            node: sum(edge[2] for edge in adjacency[node] if edge[0] in component)
            for node in component
        }
        anchor = max(component, key=lambda node: (weighted_degree[node], -node))
        anchors.add(anchor)
        assigned = {anchor}
        signs[anchor] = 1

        # Maximum-confidence spanning-tree initialization.
        while len(assigned) < len(component):
            best: tuple[float, int, int, float] | None = None
            for source in sorted(assigned):
                for target, corr, weight in adjacency[source]:
                    if target in assigned or target not in component:
                        continue
                    candidate = (weight, source, target, corr)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is None:  # Defensive; a connected component should not enter here.
                break
            _, source, target, corr = best
            signs[target] = signs[source] * (1 if corr >= 0 else -1)
            assigned.add(target)

    # Weighted signed-graph consensus. Anchors fix each component's global sign.
    for _ in range(max(0, int(args.align_iterations))):
        updated = signs.copy()
        for node in range(n_tracklets):
            if node in anchors or not adjacency[node]:
                continue
            vote = sum(weight * (1 if corr >= 0 else -1) * signs[other] for other, corr, weight in adjacency[node])
            if abs(vote) > 1e-12:
                updated[node] = 1 if vote > 0 else -1
        signs = updated

    retained_after = []
    for left in range(n_tracklets):
        for right, corr, _ in adjacency[left]:
            if right > left:
                retained_after.append(float(corr) * int(signs[left]) * int(signs[right]))
    diagnostics["corr_after_median"] = (
        float(np.median(retained_after)) if retained_after else math.nan
    )
    if retained_before and retained_after:
        diagnostics["agreement_gain"] = float(
            np.median(retained_after) - np.median(retained_before)
        )
    diagnostics["alignment_applied"] = int(np.any(signs < 0))
    if not retained_before:
        diagnostics["status"] = "no_reliable_pairwise_links"
    elif diagnostics["alignment_applied"]:
        diagnostics["status"] = "aligned_with_flips"
    else:
        diagnostics["status"] = "reliable_links_no_flip_needed"

    aligned *= signs[None, :]
    aligned_reference = safe_row_median(features * signs[None, :])
    for column in range(n_tracklets):
        candidate = features[:, column] * signs[column]
        valid = np.isfinite(candidate) & np.isfinite(aligned_reference)
        if np.count_nonzero(valid) < min_overlap:
            continue
        x = candidate[valid].astype(np.float64)
        y = aligned_reference[valid].astype(np.float64)
        if np.std(x) >= 1e-12 and np.std(y) >= 1e-12:
            correlations[column] = float(np.corrcoef(x, y)[0, 1])
    return aligned, signs, correlations, diagnostics, pairwise_corr, pairwise_overlap

def interpolate_nan(signal: np.ndarray) -> tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.isfinite(signal)
    valid_ratio = float(np.mean(valid)) if signal.size else 0.0
    if np.count_nonzero(valid) < 2:
        return np.asarray([], dtype=np.float32), valid_ratio
    indices = np.arange(signal.size)
    output = signal.copy()
    output[~valid] = np.interp(indices[~valid], indices[valid], signal[valid])
    return output.astype(np.float32), valid_ratio

def filter_signal(signal: np.ndarray, fps: float, args) -> tuple[np.ndarray, float]:
    observed = np.flatnonzero(np.isfinite(signal))
    interpolated, valid_ratio = interpolate_nan(signal)
    if observed.size < 2 or observed[-1] - observed[0] + 1 < math.ceil(fps * args.min_signal_s):
        return np.asarray([], dtype=np.float32), valid_ratio
    clean = detrend(interpolated, type="linear")
    kernel = args.median_kernel if args.median_kernel % 2 == 1 else args.median_kernel + 1
    if kernel >= 3 and len(clean) >= kernel:
        clean = medfilt(clean, kernel_size=kernel)
    nyquist = fps / 2.0
    low = max(args.rr_min_hz / nyquist, 1e-4)
    high = min(args.rr_max_hz / nyquist, 0.999)
    if high <= low:
        return np.asarray([], dtype=np.float32), valid_ratio
    b, a = butter(args.filter_order, [low, high], btype="bandpass")
    try:
        filtered = filtfilt(b, a, clean)
    except ValueError:
        return np.asarray([], dtype=np.float32), valid_ratio
    std = float(np.std(filtered))
    if not np.isfinite(std) or std < 1e-12:
        return np.asarray([], dtype=np.float32), valid_ratio
    return ((filtered - np.mean(filtered)) / std).astype(np.float32), valid_ratio

def fft_rr(filtered: np.ndarray, fps: float, args) -> dict[str, Any]:
    result = {"rr_bpm": math.nan, "peak_hz": math.nan, "snr_db": math.nan, "ok": False, "reason": ""}
    if filtered.size < max(25, int(round(fps * args.min_signal_s))):
        result["reason"] = "signal_too_short_or_invalid"
        return result
    frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    power = np.abs(np.fft.rfft(filtered)) ** 2
    band = (frequencies >= args.rr_min_hz) & (frequencies <= args.rr_max_hz)
    if not np.any(band):
        result["reason"] = "empty_frequency_band"
        return result
    band_f = frequencies[band]
    band_p = power[band]
    peak_index = int(np.argmax(band_p))
    peak_power = float(band_p[peak_index])
    noise = float((np.sum(band_p) - peak_power) / max(1, len(band_p) - 1))
    peak_hz = float(band_f[peak_index])
    result.update(
        {
            "rr_bpm": peak_hz * 60.0,
            "peak_hz": peak_hz,
            "snr_db": 10.0 * math.log10((peak_power + 1e-12) / (noise + 1e-12)),
            "ok": True,
            "reason": "ok",
        }
    )
    return result

def analyze_signal(signal: np.ndarray, fps: float, args, timer: TimerBook) -> dict[str, Any]:
    with timer.measure("filtering_s"):
        filtered, valid_ratio = filter_signal(signal, fps, args)
    if valid_ratio < args.min_signal_valid_ratio:
        result = {
            "rr_bpm": math.nan,
            "peak_hz": math.nan,
            "snr_db": math.nan,
            "ok": False,
            "reason": f"valid_ratio_too_low:{valid_ratio:.3f}<{args.min_signal_valid_ratio:.3f}",
        }
    else:
        with timer.measure("fft_s"):
            result = fft_rr(filtered, fps, args)
    result["signal_samples"] = int(len(signal))
    result["signal_valid_ratio"] = valid_ratio
    return result
