@echo off
setlocal
cd /d "%~dp0"
"%~dp0audit_tools\python\python.exe" "%~dp0audit_tools\verify_math_integration.py"
set "mathVerificationExit=%ERRORLEVEL%"
pause
exit /b %mathVerificationExit%
