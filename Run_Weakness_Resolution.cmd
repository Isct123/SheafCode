@echo off
setlocal
cd /d "%~dp0"
echo Resume the isolated empirical follow-up and rebuild the submission.
echo Do not start this launcher while another follow-up training run is active.
"audit_tools\python\python.exe" "Research_Release\extensions\weakness_resolution\run.py"
if errorlevel 1 (
  echo Follow-up stopped with an error. Completed checkpoints are retained.
  pause
  exit /b 1
)
echo Research validation completed.
pause
