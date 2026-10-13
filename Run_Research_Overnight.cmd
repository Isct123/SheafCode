@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Research_Release\Run_Overnight.ps1" %*
echo.
echo Results and logs are in Research_Release.
pause

