@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist "%~dp0audit_tools\python\python.exe" (
  echo Project Python is missing.
  pause
  exit /b 1
)
"%~dp0audit_tools\python\python.exe" "%~dp0audit_tools\verify_operational_validation.py"
set "verificationExit=%ERRORLEVEL%"
pause
exit /b %verificationExit%
