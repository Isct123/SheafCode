@echo off
setlocal
cd /d "%~dp0"
set EXPANDED_WORKERS=2
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PY=audit_tools\python\python.exe
set TASK=Research_Release\extensions\expanded_validation
"%PY%" "%TASK%\acquire.py"
if errorlevel 1 goto failed
"%PY%" "%TASK%\acquire_nsw.py"
if errorlevel 1 goto failed
"%PY%" "%TASK%\finalize.py"
if errorlevel 1 goto failed
"%PY%" "%TASK%\run.py"
if errorlevel 1 goto failed
"%PY%" "%TASK%\audit.py"
if errorlevel 1 goto failed
"%PY%" "Research_Release\extensions\weakness_resolution\run.py"
if errorlevel 1 goto failed
echo Broad validation, audit and manuscript completed.
exit /b 0
:failed
echo Validation stopped on an error. Successful frozen stages are retained for resume.
exit /b 1
