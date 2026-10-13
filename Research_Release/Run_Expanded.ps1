param([switch]$Force)
$ErrorActionPreference='Stop'
$releaseRoot=$PSScriptRoot
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
New-Item -ItemType Directory -Force -Path 'expanded_logs' | Out-Null
$statusPath=Join-Path $releaseRoot 'expanded_logs\status.txt'
$lockPath=Join-Path $releaseRoot 'expanded_logs\running.lock'
$lock=$null
try {
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "Python not found: $pythonExe" }
    $lock=[System.IO.File]::Open($lockPath,[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    "Started $(Get-Date -Format o); controller PID=$PID" | Out-File -LiteralPath $statusPath -Append
    $ErrorActionPreference='Continue'
    & $pythonExe -m pytest tests -q *>&1 | Tee-Object -FilePath 'expanded_logs\tests.log'
    $stageExit=$LASTEXITCODE; $ErrorActionPreference='Stop'
    if ($stageExit -ne 0) { throw 'Tests failed; expanded experiments not started.' }
    $failures=0
    foreach ($region in @('delhi2025','bengaluru_raw','hyderabad','bengaluru','delhi','ncr','mumbai','kolkata','india')) {
        $folder=Join-Path $releaseRoot "results_expanded\$region"
        $manifestPath=Join-Path $folder 'manifest.json'; $done=$false
        if ((Test-Path -LiteralPath $manifestPath) -and (-not $Force)) {
            $m=Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
            $done=($m.status -in @('complete','excluded')) -and ($m.args.days -eq 120) -and ($m.args.run_days -eq 60) -and ($m.args.stations -eq 1000) -and ($m.args.run_stations -eq 1000)
            foreach ($entry in $m.code_sha256.PSObject.Properties) {
                $p=Join-Path $releaseRoot $entry.Name
                if ((-not (Test-Path -LiteralPath $p)) -or ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value)) { $done=$false; break }
            }
        }
        if (-not $done) {
            "Starting core $region $(Get-Date -Format o)" | Out-File -LiteralPath $statusPath -Append
            $ErrorActionPreference='Continue'
            & $pythonExe -u scripts/run_corrected.py --region $region --days 120 --run-days 60 --stations 1000 --run-stations 1000 --output results_expanded *>&1 | Tee-Object -FilePath "expanded_logs\$region-core.log"
            $stageExit=$LASTEXITCODE; $ErrorActionPreference='Stop'
            if ($stageExit -ne 0) { $failures++; "FAILED core $region exit=$stageExit" | Out-File -LiteralPath $statusPath -Append; continue }
        }
        if ((Test-Path -LiteralPath (Join-Path $folder 'model.npz')) -and (($Force) -or (-not $done) -or (-not (Test-Path -LiteralPath (Join-Path $folder 'fault_structure_summary.json'))))) {
            $ErrorActionPreference='Continue'
            & $pythonExe -u scripts/run_fault_structure.py --region $region --hours 360 --output results_expanded *>&1 | Tee-Object -FilePath "expanded_logs\$region-fault.log"
            $stageExit=$LASTEXITCODE; $ErrorActionPreference='Stop'
            if ($stageExit -ne 0) { $failures++; "FAILED faults $region exit=$stageExit" | Out-File -LiteralPath $statusPath -Append }
        }
        "Finished $region $(Get-Date -Format o)" | Out-File -LiteralPath $statusPath -Append
    }
    $ErrorActionPreference='Continue'
    & $pythonExe -u scripts/build_expanded_audit.py *>&1 | Tee-Object -FilePath 'expanded_logs\audit.log'
    $stageExit=$LASTEXITCODE; $ErrorActionPreference='Stop'
    if ($stageExit -ne 0) { $failures++ }
    "Finished $(Get-Date -Format o); failures=$failures" | Out-File -LiteralPath $statusPath -Append
    if ($failures -gt 0) { exit 1 }
} catch {
    "FAILED controller $(Get-Date -Format o): $_" | Out-File -LiteralPath $statusPath -Append
    Write-Error $_
    exit 1
} finally { if ($null -ne $lock) { $lock.Dispose() } }
