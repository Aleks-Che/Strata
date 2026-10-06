@echo off
setlocal
cd /d "%~dp0"
rem Step-3.7-Flash optimized web chat: expert cache, pipeline, allocation reuse, batch copies.
rem MTP is off; the separate MTP probe is limited to 480 total tokens.
set "STEP_PROFILE=%~dp0build-local\step35-cuda\step18-fast-profile\step37.json"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ERROR: Python environment .venv is missing.
    pause
    exit /b 1
)
if not exist "%STEP_PROFILE%" (
    echo ERROR: Step optimized profile is missing. See docs\Step-3.7-Flash\STEP37_FLASH_FAST_LAUNCH.md
    pause
    exit /b 1
)
if not exist "%~dp0build-local\step35-cuda\bin\strata-step35-fast.exe" (
    echo ERROR: strata-step35-fast.exe is missing. See the launch guide.
    pause
    exit /b 1
)
echo Step-3.7-Flash optimized web chat - default URL http://127.0.0.1:8093
echo MTP off; context 4096; expert cache 8192 MiB; reuse and batch copies on.
"%~dp0.venv\Scripts\python.exe" -X utf8 -m serve.server --engine strata --config "%STEP_PROFILE%" --host 127.0.0.1 --port 8093 --open %*
set "STEP_EXIT_CODE=%ERRORLEVEL%"
if not "%STEP_EXIT_CODE%"=="0" pause
exit /b %STEP_EXIT_CODE%
