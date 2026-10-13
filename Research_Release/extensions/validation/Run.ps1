param([switch]$SkipGraph)
$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1';$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR=Join-Path $workspaceRoot 'audit_tools\mpl_cache'
New-Item -ItemType Directory -Force -Path 'validation_logs' | Out-Null
$lock=$null
try {
    $lock=[System.IO.File]::Open((Join-Path $releaseRoot 'expanded_logs\running.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    $PID | Out-File 'validation_logs\runner_pid.txt'
    'Started '+(Get-Date -Format o) | Out-File 'validation_logs\status.txt' -Append
    $steps=@(
        @('-m','pytest','tests','extensions/functional','extensions/structured','extensions/strong_baselines','extensions/followup','extensions/validation/test_validation.py','-q','-p','no:cacheprovider'),
        @('-u','extensions/validation/resume.py','--skip-graph')
    )
    if (-not $SkipGraph) {
        $steps+=,@('-m','pytest','extensions/validation/test_graph.py','-q','-p','no:cacheprovider')
        $steps+=,@('-u','extensions/validation/graph_benchmark.py')
    }
    foreach ($step in $steps) {
        ('Stage '+($step -join ' ')+' '+(Get-Date -Format o)) | Out-File 'validation_logs\status.txt' -Append
        $ErrorActionPreference='Continue'
        & $pythonExe @step *>&1 | Tee-Object -FilePath 'validation_logs\run.log' -Append
        $stageExit=$LASTEXITCODE;$ErrorActionPreference='Stop'
        if ($stageExit -ne 0) {throw "Validation stage failed: $($step -join ' '), exit=$stageExit"}
    }
    ('Completed '+(Get-Date -Format o)+'; '+ $(if ($SkipGraph) {'non-graph studies only'} else {'all configured studies'})) | Out-File 'validation_logs\status.txt' -Append
} catch {
    ('Stopped/failed '+(Get-Date -Format o)+': '+$_) | Out-File 'validation_logs\status.txt' -Append
    Write-Error $_
    exit 1
} finally {if ($null -ne $lock) {$lock.Dispose()}}
