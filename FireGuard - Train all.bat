@echo off
rem Double-click to run. Uses the project interpreter: Prototype\.venv, then WildFireDetection\.venv,
rem then python on PATH.
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
rem Cache, baselines, M1-M4 on all folds, evaluation, report, publish. Safe to re-run after an
rem interruption: it continues where it stopped. Keep the laptop plugged in with sleep off.
"%PY%" scripts\train_all.py --data-root ..\WildfireSpreadTS\WildfireSpreadTS %*
pause
