param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^COM[0-9]+$')]
    [string]$Port,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$RobotId = 'deskbuddy'
)

# Interactive official LeRobot calibration. Run with the operator at the arm.
# Motor IDs must already be configured; this script does not assign IDs.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $DeskBuddyRoot
try {
    $armPython = Get-DeskBuddyPython
    $calibrationDirectory = Join-Path $DeskBuddyRoot 'runs\motor-calibration'
    New-Item -ItemType Directory -Force -Path $calibrationDirectory | Out-Null
    Write-Host "Interactive calibration for $RobotId on $Port. This changes motor settings and torque state."
    Write-Host 'Keep the arm supported; follow the SDK prompts to position and move joints by hand.'
    & $armPython -u -m deskbuddy.calibrate_arm `
        '--port' $Port `
        '--id' $RobotId `
        '--directory' $calibrationDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Calibration did not finish successfully.' }
    Write-Host "Calibration saved under $calibrationDirectory. Verify joint readings before commanding motion."
} finally {
    Pop-Location
}
