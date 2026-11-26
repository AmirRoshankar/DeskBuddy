# Shared paths only. Sourcing this file does not access robot hardware.
$DeskBuddyRoot = Split-Path -Parent $PSScriptRoot
function Get-DeskBuddyPython {
    $candidates = @(
        (Join-Path $DeskBuddyRoot '.venv\Scripts\python.exe'),
        (Join-Path (Split-Path -Parent $DeskBuddyRoot) '.venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw 'Create a project environment first. See README.md.'
}
