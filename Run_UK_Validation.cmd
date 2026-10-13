@echo off
setlocal
cd /d "%~dp0"
"%~dp0audit_tools\python\python.exe" "%~dp0Research_Release\extensions\uk_validation\run.py"
set "UK_RESEARCH_EXIT=%ERRORLEVEL%"
if not "%UK_RESEARCH_EXIT%"=="0" echo UK validation stopped. Read Research_Release\uk_validation_results\status.json and logs.
if "%UK_RESEARCH_EXIT%"=="0" echo UK validation completed. Read IRIS_Submission_Recent_Data\Review_Notes\UK_VALIDATION_REPORT.md.
pause
exit /b %UK_RESEARCH_EXIT%
