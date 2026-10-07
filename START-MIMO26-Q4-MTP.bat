@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "MIMO_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%MIMO_PYTHON%" (
    echo ERROR: .venv\Scripts\python.exe is missing. See docs\mimo-v2.6-flash\MIMO26_FLASH_FAST_LAUNCH.md
    pause
    exit /b 1
)
"%MIMO_PYTHON%" -X utf8 -m tools.run_mimo2_profile --profile "%~dp0strata-mimo26-q4-mtp-10_93.json" %*
set "MIMO_EXIT_CODE=%ERRORLEVEL%"
if not "%MIMO_EXIT_CODE%"=="0" pause
exit /b %MIMO_EXIT_CODE%
