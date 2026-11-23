@echo off
rem Double-click to run. Uses the project interpreter: Prototype\.venv, then WildFireDetection\.venv,
rem then python on PATH.
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
rem Website + API on http://127.0.0.1:8000 (opens the browser). Close this window to stop.
"%PY%" scripts\app\start_app.py %*
if errorlevel 1 pause
