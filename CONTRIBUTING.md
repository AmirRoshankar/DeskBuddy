# Contributing

Start with [the maintainer handoff](docs/MAINTAINER.md). Keep hardware actions explicit, numerical and learned model roles clear, and geometric/simulated/physical results separately labeled.

From a virtual environment:

```bash
python -m pip install -e ".[dev,flow,simulation]" opencv-python
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python -m deskbuddy demo --headless --output runs/offline
```

Changes to feedback, process ownership, camera freshness, retries or collision behavior need meaningful regression coverage. Do not run physical motion as a test-suite side effect. Describe the trigger, changed behavior, verification and limitations in a pull request.

Keep raw frames, local configuration, serial identifiers, generated datasets and large weights out of commits. Optional synthetic weights belong in the versioned checksummed release. Preserve upstream notices and dependency licensing terms.
