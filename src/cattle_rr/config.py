from dataclasses import dataclass, asdict
from math import isfinite
@dataclass
class RRConfig:
    fb_pyr_scale: float = 0.5
    fb_levels: int = 3
    fb_window: int = 15
    fb_iterations: int = 3
    fb_poly_n: int = 5
    fb_poly_sigma: float = 1.2
    motion_percentile: float = 72.0
    min_region_pixels: int = 20
    max_points: int = 50
    max_tracklets: int = 500
    min_tracklet_samples: int = 10
    quality_level: float = 0.01
    min_point_distance: int = 8
    block_size: int = 7
    lk_window: int = 21
    lk_levels: int = 3
    lk_iterations: int = 30
    lk_epsilon: float = 0.01
    lk_fb_error: float = 1.5
    max_pair_gap: int = 3
    align_min_overlap: int = 30
    align_min_overlap_s: float = 2.0
    align_min_abs_corr: float = 0.3
    align_iterations: int = 3
    align_max_z: float = 4.0
    rr_min_hz: float = 0.43
    rr_max_hz: float = 2.0
    min_signal_s: float = 5.0
    min_signal_valid_ratio: float = 0.35
    median_kernel: int = 5
    filter_order: int = 3
    def validate(self,fps):
        if not isfinite(fps) or fps <= 0 or fps <= 2*self.rr_max_hz: raise ValueError('FPS must be finite and satisfy the Nyquist condition')
        if not 0 < self.rr_min_hz < self.rr_max_hz: raise ValueError('Invalid RR band')
    def to_dict(self): return asdict(self)
