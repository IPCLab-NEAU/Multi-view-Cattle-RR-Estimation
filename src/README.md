## Requirements
Python 3.10+.

```bash
python -m pip install -e .
```

Alternatively, install `requirements.txt` and run with `PYTHONPATH=src` (Linux/macOS) or `$env:PYTHONPATH="src"` (PowerShell).

## Input

```text
roi_segment/
  frames/
    000001.png
    000002.png
    000005.png
  masks/
    000001.png
    000002.png
    000005.png
  frame_indices.csv  # optional
```

Frame and mask **stems must match**, image dimensions must remain constant, and masks must be nonempty where applicable. The optional `frame_indices.csv` has two columns:

```csv
filename,source_frame_index
000001,0
000002,1
000005,4
```

## Run

```bash
python -m cattle_rr --input /path/to/roi_segment --fps 25 --output outputs/run --config configs/default.json
```

Output files are `rr_result.json` and `signals.csv`. The CSV records the raw polarity-aligned median, the median without APA, the filtered signal, and indices in the ROI segment and the source video. Unavailable rates are recorded as JSON `null` with a failure status.
