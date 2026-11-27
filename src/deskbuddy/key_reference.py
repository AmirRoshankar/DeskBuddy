"""Create a local YOLOE visual key reference from a boxed example, no hardware."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", choices=["s", "m"], default="m")
    parser.add_argument(
        "--box", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2")
    )
    args = parser.parse_args()
    root = Path("runs/deskbuddy/perception").resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "settings"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLOE

    from .learned.data import check_budget

    check_budget(root, 50_000_000)
    weights = root / f"yoloe-26{args.size}-seg.pt"
    if not weights.exists():
        parser.error(
            f"Install the official YOLOE-26 Small promptable weights at {weights}"
        )
    rgb = np.array(Image.open(args.image).convert("RGB"))
    x1, y1, x2, y2 = args.box
    if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
        parser.error("Reference box must be inside the image")
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    model = YOLOE(str(weights))
    model.predict(
        bgr,
        refer_image=bgr,
        visual_prompts=dict(bboxes=np.array([args.box]), cls=np.array([0])),
        device=0 if torch.cuda.is_available() else "cpu",
        imgsz=960,
        verbose=False,
    )
    torch.save(
        dict(names=["keys"], embeddings=model.model.pe.cpu(), reference_roi=args.box),
        root / f"keys-visual-prompts-{args.size}.pt",
    )
    print("Saved local key reference. Restart DeskBuddy to load it. No hardware used.")


if __name__ == "__main__":
    main()
