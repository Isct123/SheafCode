param([switch]$Force, [int]$Days=24, [int]$RunDays=12, [int]$Stations=12, [int]$RunStations=6, [int]$FaultHours=120)
$ErrorActionPreference='Stop'
$releaseRoot=$PSScriptRoot
$workspaceRoot=Split-Path -Parent $releaseRoot
$pythonExe=Join-Path $workspaceRoot 'audit_tools\python\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Portable Python is missing from audit_tools/python.' }
Set-Location -LiteralPath $releaseRoot
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR=Join-Path $releaseRoot 'matplotlib_cache'
New-Item -ItemType Directory -Force -Path (Join-Path $releaseRoot 'logs') | Out-Null
$statusPath=Join-Path $releaseRoot 'logs\run_status.txt'
"Started $(Get-Date -Format o)" | Out-File -LiteralPath $statusPath -Append
& $pythonExe -m pytest tests -q
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed; experiments were not started.' }
$regions=@('bengaluru_raw','delhi2025','hyderabad','bengaluru','delhi','ncr','mumbai','kolkata','india')
$failures=0
foreach ($region in $regions) {
    $folder=Join-Path $releaseRoot "results\$region"
    $manifestPath=Join-Path $folder 'manifest.json'
    $done=$false
    if ((Test-Path -LiteralPath $manifestPath) -and (-not $Force)) {
        $manifest=Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
                                                                
        $sameArgs=($manifest.args.days -eq $Days -and $manifest.args.run_days -eq $RunDays -and $manifest.args.stations -eq $Stations -and $manifest.args.run_stations -eq $RunStations)
        $sameCode=$true
        foreach ($entry in $manifest.code_sha256.PSObject.Properties) {
            $sourcePath=Join-Path $releaseRoot $entry.Name
            if ((-not (Test-Path -LiteralPath $sourcePath)) -or ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value)) { $sameCode=$false; break }
        }
        if ($sameArgs -and $sameCode -and ($manifest.status -eq 'complete' -or $manifest.status -eq 'excluded')) { $done=$true }
    }
    if (-not $done) {
        Write-Host "Running $region; log: logs\$region-core.log"
        $ErrorActionPreference='Continue'
        & $pythonExe -u scripts/run_corrected.py --region $region --days $Days --run-days $RunDays --stations $Stations --run-stations $RunStations *>&1 |
            Tee-Object -FilePath (Join-Path $releaseRoot "logs\$region-core.log")
        $stageExitCode=$LASTEXITCODE
        $ErrorActionPreference='Stop'
        if ($stageExitCode -ne 0) {
            $failures++
            "FAILED core $region $(Get-Date -Format o)" | Out-File -LiteralPath $statusPath -Append
            Write-Warning "Core experiment failed for $region. Continuing other networks."
            continue
        }
    }
    $modelPath=Join-Path $folder 'model.npz'
    $faultPath=Join-Path $folder 'fault_structure_summary.json'
    if ((Test-Path -LiteralPath $modelPath) -and (($Force) -or (-not $done) -or (-not (Test-Path -LiteralPath $faultPath)))) {
        $ErrorActionPreference='Continue'
        & $pythonExe -u scripts/run_fault_structure.py --region $region --hours $FaultHours *>&1 |
            Tee-Object -FilePath (Join-Path $releaseRoot "logs\$region-fault.log")
        $stageExitCode=$LASTEXITCODE
        $ErrorActionPreference='Stop'
        if ($stageExitCode -ne 0) {
            $failures++
            "FAILED faults $region $(Get-Date -Format o)" | Out-File -LiteralPath $statusPath -Append
            Write-Warning "Fault experiment failed for $region. Continuing."
        }
    }
    "Finished $region $(Get-Date -Format o)" | Out-File -LiteralPath $statusPath -Append
}
& $pythonExe scripts/build_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed. Experiment outputs remain saved.' }
if ($failures -eq 0) {
    foreach ($script in @('finalize_results.py','run_weather.py','replay_interval_details.py','run_sensor_repair.py','build_technical_extensions.py','build_final_report.py')) {
        $ErrorActionPreference='Continue'
        & $pythonExe (Join-Path 'scripts' $script)
        $stageExitCode=$LASTEXITCODE
        $ErrorActionPreference='Stop'
        if ($stageExitCode -ne 0) { throw "Supplementary audit failed: $script. Existing outputs remain saved." }
    }
    $tectonic=Join-Path $workspaceRoot 'audit_tools\tectonic\tectonic.exe'
    if (Test-Path -LiteralPath $tectonic) {
        $env:TECTONIC_CACHE_DIR=Join-Path $workspaceRoot 'audit_tools\tex_cache'
        $ErrorActionPreference='Continue'
        & $tectonic --only-cached --keep-logs docs/SCIENTIFIC_RESULTS_AND_AUDIT.tex
        $stageExitCode=$LASTEXITCODE
        $ErrorActionPreference='Stop'
        if ($stageExitCode -ne 0) { throw 'Scientific report PDF failed to compile; Markdown and results remain saved.' }
        & $pythonExe scripts/build_final_report.py
        if ($LASTEXITCODE -ne 0) { throw 'Scientific handoff archive failed.' }
    }
}
"Finished $(Get-Date -Format o); failures=$failures" | Out-File -LiteralPath $statusPath -Append
Write-Host 'Finished. Read Research_Release/docs/OVERNIGHT_REPORT.md and logs/run_status.txt.'
Write-Host 'Final scientific audit: Research_Release/docs/SCIENTIFIC_RESULTS_AND_AUDIT.md'
Write-Host 'Results handoff: Research_Release/Scientific_Results.zip'
if ($failures -gt 0) { exit 1 }
