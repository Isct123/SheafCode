$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1';$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR=Join-Path $workspaceRoot 'audit_tools\mpl_cache'
New-Item -ItemType Directory -Force -Path 'final_verification' | Out-Null
$lock=$null
try {
    $lock=[System.IO.File]::Open((Join-Path $releaseRoot 'expanded_logs\running.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    'Started '+(Get-Date -Format o) | Out-File 'final_verification\status.txt' -Append
    foreach ($step in @(
        @('-m','pytest','tests','extensions/functional','extensions/structured','extensions/strong_baselines','extensions/followup','-q','-p','no:cacheprovider'),
        @('-u','extensions/followup/review_expanded.py'),
        @('-u','extensions/followup/report_constrained.py'),
        @('-u','extensions/followup/build_figure_pack.py'),
        @('-u','extensions/followup/finalize.py'),
        @('-u','extensions/followup/final_readiness.py')
    )) {
        $ErrorActionPreference='Continue'
        & $pythonExe @step *>&1 | Tee-Object -FilePath 'final_verification\run.log' -Append
        $stageExit=$LASTEXITCODE;$ErrorActionPreference='Stop'
        if ($stageExit -ne 0) {throw "Verification failed: $($step -join ' '), exit=$stageExit"}
    }
    'Completed '+(Get-Date -Format o)+'; see READINESS.md for scope and limitations' | Out-File 'final_verification\status.txt' -Append
} catch {
    ('FAILED '+(Get-Date -Format o)+': '+$_) | Out-File 'final_verification\status.txt' -Append
    Write-Error $_;exit 1
} finally {if ($null -ne $lock) {$lock.Dispose()}}
