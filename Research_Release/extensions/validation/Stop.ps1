$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runnerPath=Join-Path $PSScriptRoot 'Run.ps1'
$pidPath=Join-Path $releaseRoot 'validation_logs\runner_pid.txt'
$progressPath=Join-Path $releaseRoot 'validation_results\graph_progress.json'
if (Test-Path -LiteralPath $progressPath) {
    $progress=Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
    if (($progress.status -eq 'running') -and $progress.pid) {
        $graphProcess=Get-CimInstance Win32_Process -Filter "ProcessId = $($progress.pid)"
        if ($graphProcess -and ($graphProcess.CommandLine -like '*extensions/validation/graph_benchmark.py*') -and ($graphProcess.Name -eq 'python.exe')) {Stop-Process -Id $progress.pid}
    }
}
if (Test-Path -LiteralPath $pidPath) {
    $runnerId=[int](Get-Content -Raw -LiteralPath $pidPath)
    $runnerProcess=Get-CimInstance Win32_Process -Filter "ProcessId = $runnerId"
    if ($runnerProcess -and $runnerProcess.CommandLine.Contains($runnerPath) -and ($runnerProcess.Name -eq 'powershell.exe')) {Stop-Process -Id $runnerId}
}
('Stopped by user '+(Get-Date -Format o)+'; completed epoch checkpoints preserved') | Out-File (Join-Path $releaseRoot 'validation_logs\status.txt') -Append
Write-Host 'Stopped the validation runner. Restart Run_Validation_Research.cmd to resume the saved epoch.'
