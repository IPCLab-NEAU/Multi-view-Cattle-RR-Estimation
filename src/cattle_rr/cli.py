import argparse,json
from pathlib import Path
from .config import RRConfig
from .pipeline import G2LFlowRREstimator
from .io_utils import load_sequence,save_outputs
def main():
    p=argparse.ArgumentParser(description="Core RR estimator for precomputed abdominal ROI frames and masks")
    p.add_argument("--input",required=True); p.add_argument("--fps",required=True,type=float); p.add_argument("--output",required=True); p.add_argument("--config")
    a=p.parse_args(); c=RRConfig()
    if a.config:
        values=json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("The configuration must be a JSON object")
        known_parameters = c.to_dict()
        for k,v in values.items():
            if k not in known_parameters: raise ValueError(f"Unknown parameter: {k}")
            setattr(c,k,v)
    frames,masks,frame_indices=load_sequence(a.input); result=G2LFlowRREstimator(c).estimate(frames,masks,a.fps,frame_indices=frame_indices); save_outputs(result,a.output)
    print("Status:",result["status"]); print("RR (bpm):",result.get("rr_bpm"))
if __name__=="__main__": main()
