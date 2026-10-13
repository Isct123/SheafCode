$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1';$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$lock=$null
try {
    $lock=[System.IO.File]::Open((Join-Path $releaseRoot 'expanded_logs\running.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    & $pythonExe -m pytest extensions/joint_decisions -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {throw 'Joint decision tests failed'}
    $ErrorActionPreference='Continue'
    & $pythonExe -u extensions/joint_decisions/run.py
    $stageExit=$LASTEXITCODE;$ErrorActionPreference='Stop'
    if ($stageExit -ne 0) {throw "Additional audit failed: exit $stageExit"}
} finally {if ($null -ne $lock) {$lock.Dispose()}}
