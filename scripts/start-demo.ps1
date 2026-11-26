$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $DeskBuddyRoot
try {
    $demoPython = Get-DeskBuddyPython
    & $demoPython -m deskbuddy demo
    if ($LASTEXITCODE -ne 0) { throw 'Preview exited with an error.' }
} finally {
    Pop-Location
}
