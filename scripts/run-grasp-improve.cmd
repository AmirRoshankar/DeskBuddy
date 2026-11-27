@echo off
rem Offline grasp-improvement run: MuJoCo toy physics + PyTorch only. Never moves the arm.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist runs\offline-jobs mkdir runs\offline-jobs
set LOG=runs\offline-jobs\grasp-improve.log
echo started %DATE% %TIME% > %LOG%
%PY% -X utf8 -m pytest -q tests/test_grasp_improve.py tests/test_can_physics.py tests/test_can_discard.py tests/test_joint_planner.py tests/test_reachability.py tests/test_path_overlays.py tests/test_route_execution.py >> %LOG% 2>&1
echo tests-exit %ERRORLEVEL% >> %LOG%
%PY% -X utf8 -m deskbuddy.learned.grasp_improve --folder runs\flowreach\can-necked-improve-smoke --smoke >> %LOG% 2>&1
echo smoke-exit %ERRORLEVEL% >> %LOG%
if errorlevel 1 goto done
%PY% -X utf8 -m deskbuddy.learned.grasp_improve --folder runs\flowreach\can-necked-improve-v1 >> %LOG% 2>&1
echo full-exit %ERRORLEVEL% >> %LOG%
:done
echo finished %DATE% %TIME% >> %LOG%
