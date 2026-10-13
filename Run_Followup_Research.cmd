@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Research_Release\extensions\followup\Run.ps1"
set "researchExit=%ERRORLEVEL%"
if not "%researchExit%"=="0" echo Follow-up failed. Read Research_Release\followup_logs\status.txt and run.log.
pause
exit /b %researchExit%
