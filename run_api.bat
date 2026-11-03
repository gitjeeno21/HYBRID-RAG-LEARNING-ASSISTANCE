@echo off
cd /d "%~dp0"
rem Prefer the project's virtual environment when it exists.
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
"%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8000 --loop asyncio
