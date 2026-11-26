"""Capture one warmed-up RGB/depth pair and basic depth diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import RealSense


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial")
    p.add_argument("--output", default="runs/camera_check")
    args = p.parse_args()
    camera = RealSense(args.serial)
    try:
        for _ in range(30):
            rgb, depth, intrinsics = camera.read()
    finally:
        camera.close()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth_m.npy", depth)
    valid = np.isfinite(depth) & (depth > 0)
    center = depth[190:290, 270:370]
    center_valid = center[np.isfinite(center) & (center > 0)]
    report = {
        "shape_hw": list(depth.shape),
        "valid_depth_fraction": float(valid.mean()),
        "center_valid_fraction": float((center > 0).mean()),
        "center_median_depth_m": float(np.median(center_valid))
        if center_valid.size
        else None,
        "intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "distortion": str(intrinsics.model),
            "coefficients": list(intrinsics.coeffs),
        },
        "note": "Depth values are in metres and aligned to RGB; camera-to-base calibration is still required.",
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
