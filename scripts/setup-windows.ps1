$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $DeskBuddyRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12+ is required.' }
    }
    $projectPython = Join-Path $DeskBuddyRoot '.venv\Scripts\python.exe'
    & $projectPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
    & $projectPython -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    & $projectPython -m pip install -c requirements/constraints-cuda.txt -e '.[dev,camera,robot,perception,simulation,viz]'
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
    & $projectPython -m deskbuddy doctor --require-cuda
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed; inspect runs/environment.json' }
} finally {
    Pop-Location
}
