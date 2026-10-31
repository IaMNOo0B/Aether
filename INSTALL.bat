@echo off
setlocal
cd /d "%~dp0"
title AETHER - Setup
echo ========================================
echo          AETHER - WINDOWS SETUP
echo ========================================
where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher was not found.
    echo Install Python 3.12+ from https://www.python.org/downloads/windows/
    echo Make sure "Add Python to PATH" is enabled.
    pause
    exit /b 1
)
py -3.12 --version
if errorlevel 1 (
    echo Python 3.12 was not found. Install Python 3.12 or newer.
    pause
    exit /b 1
)
if not exist .venv (
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -e ".[dev]"
if errorlevel 1 goto :failed
if not exist data mkdir data
if not exist logs mkdir logs
echo.
echo Setup completed.
echo Next: double-click START_AETHER.bat
pause
exit /b 0
:failed
echo.
echo Setup failed. Read the error above.
pause
exit /b 1
