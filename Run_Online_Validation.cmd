@echo off
cd /d "%~dp0"
"%~dp0audit_tools\python\python.exe" "%~dp0Research_Release\extensions\online_validation\run.py"
if errorlevel 1 (
  echo Validation stopped. Read Research_Release\online_validation_results\STAGE_STATUS.json and the named stage log.
  pause
  exit /b 1
)
echo Research stages completed.
pause
