@echo off
REM One-time setup: virtual environment, dependencies, .env, Ollama models.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example
)

where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo Ollama was not found. Install it from https://ollama.com, then run:
    echo     ollama pull qwen2.5:7b-instruct
    echo     ollama pull qwen2.5:3b-instruct
) else (
    echo Pulling Ollama models...
    ollama pull qwen2.5:3b-instruct
    ollama pull qwen2.5:7b-instruct
)

echo.
echo Setup complete. Start the app with run_all.bat
pause
exit /b 0

:error
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
