param([int]$Hours=40)
$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1';$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR=Join-Path $workspaceRoot 'audit_tools\mpl_cache'
New-Item -ItemType Directory -Force -Path 'constrained_logs' | Out-Null
$lock=$null
try {
    $lock=[System.IO.File]::Open((Join-Path $releaseRoot 'expanded_logs\running.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    'Started '+(Get-Date -Format o) | Out-File 'constrained_logs\status.txt' -Append
    foreach ($step in @(
        @('-m','pytest','tests','extensions/functional','extensions/structured','extensions/strong_baselines','extensions/followup','-q','-p','no:cacheprovider'),
        @('-u','extensions/followup/run_constrained.py','--hours',"$Hours"),
        @('-u','extensions/followup/matched_alerts.py'),
        @('-u','extensions/followup/benchmark_budget_dual.py'),
        @('-u','extensions/followup/verify_certificate_study.py')
    )) {
        $ErrorActionPreference='Continue'
        & $pythonExe @step *>&1 | Tee-Object -FilePath 'constrained_logs\run.log' -Append
        $stageExit=$LASTEXITCODE;$ErrorActionPreference='Stop'
        if ($stageExit -ne 0) {throw "Stage failed: $($step -join ' '), exit=$stageExit"}
    }
    'Completed '+(Get-Date -Format o) | Out-File 'constrained_logs\status.txt' -Append
} catch {
    ('Failed '+(Get-Date -Format o)+': '+$_) | Out-File 'constrained_logs\status.txt' -Append
    Write-Error $_;exit 1
} finally {if ($null -ne $lock) {$lock.Dispose()}}
