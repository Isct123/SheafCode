@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Research_Release\Run_Expanded.ps1" %*
set "researchExit=%ERRORLEVEL%"
echo.
echo Expanded status and logs: Research_Release\expanded_logs
echo Results: Research_Release\expanded_docs\EXPANDED_RESULTS.md
pause
exit /b %researchExit%
