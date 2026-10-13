$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$researchPython = Join-Path $PSScriptRoot 'audit_tools/python/python.exe'
$researchScripts = Join-Path $PSScriptRoot 'Research_Release/extensions/replication_20261012'

                                                                               
$researchResults = Join-Path $PSScriptRoot 'Research_Release/replication_20261012_results'
$cohortFile = Join-Path $researchResults 'cohorts.json'
$acquisitionComplete = Test-Path -LiteralPath $cohortFile
foreach ($researchMarker in @('EPA_acquired.json', 'NSW_acquired.json', 'UK_acquired.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $researchResults $researchMarker))) { $acquisitionComplete = $false }
}
$ukMarker = Join-Path $researchResults 'UK_acquired.json'
if ((Test-Path -LiteralPath $ukMarker) -and ((Get-Content -LiteralPath $ukMarker -Raw | ConvertFrom-Json).status -ne 'completed')) { $acquisitionComplete = $false }
if (-not $acquisitionComplete) {
    foreach ($researchStage in @('acquire.py', 'acquire_uk.py')) {
        & $researchPython -u (Join-Path $researchScripts $researchStage)
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $researchStage. Its logs and partial results are retained." }
    }
}
$no2Archive = Join-Path $PSScriptRoot 'Research_Release/replication_20261012_results/extra_epa_sources/hourly_42602_2025.zip'
if ((Test-Path -LiteralPath $no2Archive) -and ((-not $acquisitionComplete) -or (-not (Test-Path -LiteralPath (Join-Path $researchResults 'PMNO2_acquired.json'))))) {
    & $researchPython -u (Join-Path $researchScripts 'acquire_pmno2.py')
    if ($LASTEXITCODE -ne 0) { throw 'PM2.5/NO2 acquisition failed. Review its source ledger.' }
}
foreach ($researchStage in @('run.py', 'random_forest.py', 'audit_report.py', 'plot_results.py', 'package.py')) {
    & $researchPython -u (Join-Path $researchScripts $researchStage)
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $researchStage. Successful stages remain available for resume." }
}
Write-Output 'Completed. Read Research_Release/replication_20261012_results/RESULTS.md and ALL_COHORT_RESULTS.csv.'
