@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "submissionCompiler=%~dp0..\..\audit_tools\tectonic\tectonic.exe"
set "TECTONIC_CACHE_DIR=%~dp0..\..\audit_tools\tex_cache"
if not exist "%submissionCompiler%" (
  echo Local compiler missing. Use Open_in_Overleaf.zip with XeLaTeX instead.
  pause
  exit /b 1
)
if not exist "%CD%\manuscript.tex" (
  echo manuscript.tex is missing from Source.
  pause
  exit /b 1
)
if not exist "%CD%\synopsis.tex" (
  echo synopsis.tex is missing from Source.
  pause
  exit /b 1
)
"%submissionCompiler%" --only-cached --keep-logs --outdir "%CD%" "%CD%\manuscript.tex"
if errorlevel 1 goto compilation_failed
"%submissionCompiler%" --only-cached --keep-logs --outdir "%CD%" "%CD%\synopsis.tex"
if errorlevel 1 goto compilation_failed
copy /y "%CD%\manuscript.pdf" "%~dp0..\Manuscript_For_Upload.pdf" >nul
if errorlevel 1 goto copying_failed
copy /y "%CD%\synopsis.pdf" "%~dp0..\Synopsis.pdf" >nul
if errorlevel 1 goto copying_failed
echo Manuscript_For_Upload.pdf and Synopsis.pdf updated in SUBMISSION_WORKSPACE.
echo Review both PDFs and synchronize Abstract.txt, Portal_Text.txt and Video_Script.txt before submitting.
pause
exit /b 0
:compilation_failed
echo Compilation failed. Read manuscript.log or synopsis.log in Source.
echo Submission-folder PDFs were not replaced.
pause
exit /b 1
:copying_failed
echo Could not update a submission PDF. Close it in your viewer and rerun this script.
pause
exit /b 1
