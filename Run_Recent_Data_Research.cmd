@echo off
setlocal
cd /d "%~dp0"
"%~dp0audit_tools\python\python.exe" "%~dp0Research_Release\extensions\recent_data\run.py"
set "RESEARCH_EXIT=%ERRORLEVEL%"
echo.
if not "%RESEARCH_EXIT%"=="0" echo Research stopped with an error. Read Research_Release\recent_data_results\logs and status.json.
if "%RESEARCH_EXIT%"=="0" echo Completed. Read IRIS_Submission_Recent_Data\README.md.
pause
exit /b %RESEARCH_EXIT%
