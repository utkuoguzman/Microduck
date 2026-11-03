@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "src\xgoduck_sim_and_hil.py" %*
) else (
    python "src\xgoduck_sim_and_hil.py" %*
)

