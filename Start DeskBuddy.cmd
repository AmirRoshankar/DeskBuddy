@echo off
setlocal
cd /d "%~dp0"
set "DESKBUDDY_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DESKBUDDY_PYTHON%" set "DESKBUDDY_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%DESKBUDDY_PYTHON%" (
  echo Create a Python environment first. See README.md.
  pause
  exit /b 1
)
"%DESKBUDDY_PYTHON%" -X utf8 -u -m deskbuddy.desktop_launcher
if errorlevel 1 pause
