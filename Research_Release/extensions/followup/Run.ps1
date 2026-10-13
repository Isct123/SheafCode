$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1';$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
New-Item -ItemType Directory -Force -Path 'followup_logs' | Out-Null
$lock=$null
try {
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "Python not found: $pythonExe" }
    New-Item -ItemType Directory -Force -Path (Join-Path $releaseRoot 'expanded_logs') | Out-Null
                                                                              
    $lock=[System.IO.File]::Open((Join-Path $releaseRoot 'expanded_logs\running.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    $auditPath=Join-Path $releaseRoot 'expanded_docs\audit.json'
    if (-not (Test-Path -LiteralPath $auditPath)) { throw 'Finish the expanded run and its audit before this follow-up.' }
    foreach ($region in @('delhi2025','bengaluru_raw','hyderabad','bengaluru','delhi','ncr','mumbai','kolkata','india')) {
        $folder=Join-Path $releaseRoot "results_expanded\$region"
        $manifestPath=Join-Path $folder 'manifest.json'
        if (-not (Test-Path -LiteralPath $manifestPath)) {throw "Missing expanded manifest: $region"}
        $manifest=Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ($manifest.status -notin @('complete','excluded')) {throw "Expanded stage incomplete: $region"}
        if (($manifest.status -eq 'complete') -and (-not (Test-Path -LiteralPath (Join-Path $folder 'fault_structure_summary.json')))) {throw "Expanded fault study missing: $region"}
    }
    'Started '+(Get-Date -Format o) | Out-File 'followup_logs\status.txt' -Append
    foreach ($step in @(
        @('-m','pytest','tests','extensions/functional','extensions/structured','extensions/strong_baselines','extensions/followup','-q','-p','no:cacheprovider'),
        @('-u','scripts/build_expanded_audit.py'),
        @('-u','extensions/followup/review_expanded.py'),
        @('-u','extensions/followup/run.py'),
        @('-u','extensions/followup/run_robust_quantity.py'),
        @('-u','extensions/followup/benchmark_subspaces.py'),
        @('-u','extensions/followup/neural.py'),
        @('-u','extensions/followup/external.py'),
        @('-u','extensions/strong_baselines/run.py'),
        @('-u','extensions/strong_baselines/horizon_stress.py'),
        @('-u','extensions/strong_baselines/report.py'),
        @('-u','extensions/followup/finalize.py')
    )) {
        $ErrorActionPreference='Continue'
        & $pythonExe @step *>&1 | Tee-Object -FilePath 'followup_logs\run.log' -Append
        $stageExit=$LASTEXITCODE;$ErrorActionPreference='Stop'
        if ($stageExit -ne 0) { throw "Follow-up stage failed: $($step -join ' '), exit=$stageExit" }
    }
    'Completed '+(Get-Date -Format o)+'; see followup_results and remaining limitations in extensions/followup/README.md' | Out-File 'followup_logs\status.txt' -Append
} catch {
    ('Stopped/failed '+(Get-Date -Format o)+': '+$_) | Out-File 'followup_logs\status.txt' -Append
    Write-Error $_
    exit 1
} finally { if ($null -ne $lock) {$lock.Dispose()} }
