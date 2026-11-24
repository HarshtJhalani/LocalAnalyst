@echo off
REM FastAPI backend: http://127.0.0.1:8765  (Swagger UI: /docs; port = API_PORT in .env)
cd /d "%~dp0"
title Text-to-SQL API
set "API_PORT=8765"
if exist ".env" for /f "tokens=1,* delims==" %%a in ('findstr /b "API_PORT=" ".env"') do set "API_PORT=%%b"

if not exist ".venv\Scripts\python.exe" (
    echo The app is not set up yet. Run setup.bat first.
    pause
    exit /b 1
)

call "scripts\ensure_ollama.bat"
if errorlevel 1 (
    echo The API needs Ollama. Run setup.bat, or install it from https://ollama.com/download
    pause
    exit /b 1
)

echo Starting the API (loading the model can take a minute)...
echo Swagger UI: http://127.0.0.1:%API_PORT%/docs
".venv\Scripts\python.exe" -m uvicorn src.main.main:app --host 127.0.0.1 --port %API_PORT%

echo.
echo The API stopped. If it failed at startup, the message above says why
echo (most often: the model is not downloaded - run setup.bat again).
pause
