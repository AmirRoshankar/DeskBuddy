"""Best-effort diagnostics, isolated from live acquisition and capture."""

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Unique names avoid stale temporary files and concurrent-writer collisions.
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def publish(directory, rgb, depth, report):
    directory = Path(directory)
    png = io.BytesIO()
    Image.fromarray(rgb).save(png, format="PNG")
    array = io.BytesIO()
    np.save(array, depth)
    payloads = {
        "rgb.png": png.getvalue(),
        "depth_m.npy": array.getvalue(),
        "state.json": (json.dumps(report, indent=2) + "\n").encode("utf-8"),
    }
    errors = {}
    for name, payload in payloads.items():
        try:
            atomic_bytes(directory / name, payload)
        except OSError as exc:
            errors[name] = str(exc)
    return errors


def pickup_images(directory, rgb, secondary):
    """Raw RGB for pickup evidence; never reuse annotated UI pixels."""
    import time

    directory = Path(directory).resolve()
    images = [rgb]
    if secondary is not None and time.time() - secondary[1] < 1:
        images.append(secondary[0])
    paths = []
    for index, image in enumerate(images):
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=92)
        path = directory / f"pickup-rgb-{index}.jpg"
        atomic_bytes(path, stream.getvalue())
        paths.append(str(path))
    return dict(pickup_rgb_unix_s=time.time(), pickup_rgb_paths=paths)
