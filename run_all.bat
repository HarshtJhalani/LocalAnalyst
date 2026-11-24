@echo off
REM Starts the API and the UI in their own windows and opens the app.
cd /d "%~dp0"
set "API_PORT=8765"
if exist ".env" for /f "tokens=1,* delims==" %%a in ('findstr /b "API_PORT=" ".env"') do set "API_PORT=%%b"

if not exist ".venv\Scripts\python.exe" (
    echo The app is not set up yet. Running setup.bat first...
    call setup.bat || exit /b 1
)

REM The API port already answering: an API from an earlier start is still open.
curl -s -o nul http://127.0.0.1:%API_PORT%/ && (
    echo An API is already running on port %API_PORT%. Close its window to restart it with
    echo the latest code, or press a key to use it as it is.
    pause
    goto :ui
)

start "Text-to-SQL API" cmd /c run_api.bat

echo Waiting for the API to start (loading the model can take a minute)...
set /a tries=0
:wait
curl -s -o nul http://127.0.0.1:%API_PORT%/ && goto :ui
set /a tries+=1
if %tries% geq 90 (
    echo The API did not start. Check the "Text-to-SQL API" window for the error.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait

:ui
echo API is up. Starting the UI...
start "Text-to-SQL UI" cmd /c run_ui.bat
echo.
echo App:        http://localhost:8501  (opens by itself)
echo API:        http://127.0.0.1:%API_PORT%
echo Swagger UI: http://127.0.0.1:%API_PORT%/docs
echo Close the API and UI windows to stop.
timeout /t 10 >nul
