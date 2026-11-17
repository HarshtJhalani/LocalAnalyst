@echo off
REM FastAPI backend: http://127.0.0.1:8000  (Swagger UI: /docs)
cd /d "%~dp0"
title Text-to-SQL API

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting the API (Ollama must be running)...
echo Swagger UI: http://127.0.0.1:8000/docs
".venv\Scripts\python.exe" -m uvicorn src.main.main:app --host 127.0.0.1 --port 8000

echo.
echo The API stopped. If it failed at startup, check that Ollama is running
echo and the models are pulled.
pause
