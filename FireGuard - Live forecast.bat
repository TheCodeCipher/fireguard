@echo off
rem Double-click to run. Uses the project interpreter: Prototype\.venv, then WildFireDetection\.venv,
rem then python on PATH.
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
rem Tomorrow's forecast with the best model, then score every past forecast whose day is over.
"%PY%" scripts\live\run_live_forecast.py
if errorlevel 1 (pause & exit /b 1)
"%PY%" scripts\live\score_live_forecasts.py
pause
