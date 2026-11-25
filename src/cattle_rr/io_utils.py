import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_paths(directory):
    paths = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in EXT:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate image stem in {directory.name}/: {path.stem}")
        paths[path.stem] = path
    return paths


def _read_image(path, mode):
    # Python handles Unicode paths on Windows before OpenCV decodes the bytes.
    data = path.read_bytes()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), mode) if data else None


def load_sequence(root):
    root = Path(root)
    fd, md = root / "frames", root / "masks"
    if not fd.is_dir() or not md.is_dir():
        raise FileNotFoundError("Expected frames/ and masks/ subdirectories")
    fs, ms = _image_paths(fd), _image_paths(md)
    if not fs or fs.keys() != ms.keys():
        raise ValueError("Frame and mask filename stems must match exactly")

    names = sorted(fs)
    index_file = root / "frame_indices.csv"
    if index_file.exists():
        with index_file.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"filename", "source_frame_index"}.issubset(reader.fieldnames or []):
                raise ValueError("frame_indices.csv must contain filename,source_frame_index columns")
            rows = list(reader)
        index_map = {row["filename"]: int(row["source_frame_index"]) for row in rows}
        if len(rows) != len(index_map) or index_map.keys() != fs.keys():
            raise ValueError("frame_indices.csv must contain exactly one row for every frame stem")
        if min(index_map.values()) < 0 or len(set(index_map.values())) != len(index_map):
            raise ValueError("source_frame_index values must be nonnegative and unique")
        names = sorted(names, key=index_map.__getitem__)
        indices = [index_map[name] for name in names]
    else:
        indices = list(range(len(names)))

    frames, masks = [], []
    for name in names:
        frame = _read_image(fs[name], cv2.IMREAD_COLOR)
        mask = _read_image(ms[name], cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            raise IOError(f"Cannot read paired frame/mask {name}")
        frames.append(frame)
        masks.append(mask)
    return frames, masks, indices


def _json_value(value):
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def save_outputs(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signal_keys = ["raw_signal", "pca_signal", "filtered_signal"]
    excluded = set(signal_keys) | {
        "g2l_region_mask",
        "g2l_region_frequency",
        "displacement_cube", "shared_axis", "eigenvalues", "tracklet_signs",
        "tracklet_alignment_correlations", "pairwise_alignment_correlations", "pairwise_alignment_overlaps",
    }
    summary = {key: value for key, value in result.items() if key not in excluded}
    (out / "rr_result.json").write_text(json.dumps(_json_value(summary), indent=2, allow_nan=False), encoding="utf-8")
    if result.get("g2l_region_mask") is not None:
        ok, encoded = cv2.imencode(".png", result["g2l_region_mask"])
        if not ok:
            raise IOError("Cannot encode the fixed G2L region")
        (out / "g2l_region_mask.png").write_bytes(encoded.tobytes())
    if result.get("g2l_region_frequency") is not None:
        frequency = np.clip(np.rint(result["g2l_region_frequency"] * 255), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", frequency)
        if not ok:
            raise IOError("Cannot encode the region selection frequency")
        (out / "g2l_region_frequency.png").write_bytes(encoded.tobytes())

    arrays = [np.asarray(result[key]).ravel() if result.get(key) is not None else None for key in signal_keys]
    sample_count = max((len(array) for array in arrays if array is not None), default=0)
    source_start = result.get("first_source_frame_index", 0)
    with (out / "signals.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", *signal_keys, "source_frame_index"])
        for i in range(sample_count):
            values = [float(a[i]) if a is not None and i < len(a) and np.isfinite(a[i]) else "" for a in arrays]
            writer.writerow([i, *values, source_start + i])
