@echo off
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set "MPLCONFIGDIR=%~dp0audit_tools\mpl_cache"
"%~dp0audit_tools\python\python.exe" "%~dp0Research_Release\extensions\followup\build_figure_pack.py"
pause
