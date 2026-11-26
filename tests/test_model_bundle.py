import io
import zipfile

import pytest

from scripts.download_models import digest, install_bundle


def bundle(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    data = stream.getvalue()
    return data, dict(
        sha256=digest(data),
        files={name: digest(content) for name, content in files.items()},
    )


def test_bundle_installs_only_verified_files_and_preserves_local_models(tmp_path):
    data, manifest = bundle({"runs/flowreach/demo/imf.pt": b"example checkpoint"})
    assert install_bundle(data, manifest, tmp_path) == 1
    target = tmp_path / "runs/flowreach/demo/imf.pt"
    target.write_bytes(b"local training result")
    with pytest.raises(ValueError, match="Existing checkpoint differs"):
        install_bundle(data, manifest, tmp_path)
    assert target.read_bytes() == b"local training result"


@pytest.mark.parametrize(
    "name",
    [
        "../outside.pt",
        "runs/flowreach/../../../outside.pt",
        "runs/flowreach/..\\..\\outside.pt",
    ],
)
def test_bundle_rejects_traversal_before_writing(tmp_path, name):
    data, manifest = bundle({name: b"bad"})
    # Windows ZIP writers can normalize backslashes before validation, causing
    # an earlier manifest mismatch; either rejection must leave disk untouched.
    with pytest.raises(ValueError):
        install_bundle(data, manifest, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_bundle_rejects_tampered_download(tmp_path):
    data, manifest = bundle({"runs/flowreach/demo/imf.pt": b"example"})
    with pytest.raises(ValueError, match="Bundle checksum mismatch"):
        install_bundle(data + b"tampered", manifest, tmp_path)
