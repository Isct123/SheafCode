@echo off
setlocal
cd /d "%~dp0"
"%~dp0audit_tools\python\python.exe" "%~dp0audit_tools\verify_joint_certification.py"
set "jointVerificationExit=%ERRORLEVEL%"
pause
exit /b %jointVerificationExit%
