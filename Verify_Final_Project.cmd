@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist "%~dp0audit_tools\python\python.exe" (
  echo The project Python is missing: audit_tools\python\python.exe
  pause
  exit /b 1
)
"%~dp0audit_tools\python\python.exe" "%~dp0audit_tools\verify_final_project.py"
set "verificationExit=%ERRORLEVEL%"
pause
exit /b %verificationExit%
