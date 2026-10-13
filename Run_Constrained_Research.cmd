@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Research_Release\extensions\followup\Run_Constrained.ps1" %*
pause
