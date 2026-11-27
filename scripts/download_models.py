"""Install the optional synthetic checkpoints after verifying every SHA-256.

Never opens cameras or motors. Third-party YOLO weights are not in this bundle.
"""

import argparse
import hashlib
import io
import json
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def install_bundle(data, manifest, root, replace=False):
    root = Path(root).resolve()
    if digest(data) != manifest["sha256"]:
        raise ValueError("Bundle checksum mismatch")
    payloads = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("Bundle contents do not match the manifest")
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or not entry.filename.startswith(("runs/deskbuddy/", "runs/flowreach/"))
                or name.suffix not in (".pt", ".json", ".npz")
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError("Unsafe checkpoint path")
            destination = (root / Path(*name.parts)).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Checkpoint would leave the project")
            content = archive.read(entry)
            if digest(content) != manifest["files"][entry.filename]:
                raise ValueError("Checkpoint checksum mismatch")
            if (
                destination.exists()
                and digest(destination.read_bytes()) != digest(content)
                and not replace
            ):
                raise ValueError(
                    f"Existing checkpoint differs: {entry.filename}; use --replace explicitly"
                )
            payloads[destination] = content
    # Validate every entry before changing any existing checkpoint.
    for destination, content in payloads.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return len(payloads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "models/manifest.json").read_text(encoding="utf-8"))
    with urllib.request.urlopen(manifest["url"], timeout=60) as response:
        data = response.read(50_000_001)
    if len(data) > 50_000_000:
        raise ValueError("Unexpectedly large model bundle")
    count = install_bundle(data, manifest, root, args.replace)
    print(
        f"Installed {count} verified benchmark/model artifacts. No hardware accessed."
    )


if __name__ == "__main__":
    main()
