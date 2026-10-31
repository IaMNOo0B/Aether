@echo off
setlocal
cd /d "%~dp0"
title AETHER
if not exist .venv\Scripts\python.exe (
    echo AETHER is not installed yet. Run INSTALL.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo AETHER exited with an error. Check logs/aether.log.
    pause
)
