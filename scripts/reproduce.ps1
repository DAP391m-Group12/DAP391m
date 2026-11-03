# Runs the Dataset B reproduction CLI from the repository root.
# Example: .\scripts\reproduce.ps1 verify-assets
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repositoryRoot
try {
    & python -m src.dataset_b.reproduce @args
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
