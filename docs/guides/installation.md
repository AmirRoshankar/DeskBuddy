# Installation

## CPU demo

From the repository root, create/activate Python 3.12+ environment and install:

```bash
python -m pip install -e ".[dev]"
python -m deskbuddy demo --headless --output runs/offline
python -m deskbuddy demo
```

The synthetic numerical demo needs no robot, camera, model downloads or CUDA. Linux interactive use needs Tk and a display. The application is developed on Windows; CI checks offline behavior on Windows and Linux.

## Optional components

| Extra | Purpose |
| --- | --- |
| `flow` | PyTorch training/inference |
| `camera` | RealSense, OpenCV and webcam enumeration |
| `robot` | LeRobot/Feetech hardware transport |
| `perception` | Ultralytics YOLOE and pinned OpenCV |
| `simulation` | MuJoCo gravity/contact experiments |
| `viz` | Matplotlib and Plotly research figures |

For an NVIDIA Windows setup, `scripts/setup-windows.ps1` creates a local environment and installs the recorded CUDA packages and optional stack. It does not open motors, calibrate or change the system driver. Dependency records are in `requirements/`; a separate CUDA toolkit is not needed for the recorded PyTorch wheels.

## Local robot setup

Copy `configs/local.example.json` to ignored `configs/local.json`; follow [hardware commissioning](hardware.md). Do not copy someone else's measured transforms/joint calibration. Learn which command accesses hardware before running it.

Once commissioned, `Start DeskBuddy.cmd` launches the native UI from any working directory. It first tries this repository's `.venv`, then a prepared parent `.venv`. `python -m deskbuddy.desktop_launcher` is equivalent from the root. Startup reads cameras/encoders and does not restore torque or execute motion. A second launcher detects the existing viewer.

The learned planners need separately downloaded or reproduced [synthetic checkpoints](models.md). External detector weights follow their upstream license and are not included in the model bundle.
