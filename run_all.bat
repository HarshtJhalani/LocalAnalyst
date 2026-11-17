@echo off
REM Starts the API and the UI in their own windows and opens Swagger UI.
cd /d "%~dp0"

start "Text-to-SQL API" cmd /c run_api.bat

echo Waiting for the API to start (model loading can take a minute)...
set /a tries=0
:wait
curl -s -o nul http://127.0.0.1:8000/ && goto :ready
set /a tries+=1
if %tries% geq 90 (
    echo The API did not start. Check the "Text-to-SQL API" window for the error.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait

:ready
echo API is up. Starting the UI...
start "Text-to-SQL UI" cmd /c run_ui.bat
start "" http://127.0.0.1:8000/docs
echo.
echo API:        http://127.0.0.1:8000
echo Swagger UI: http://127.0.0.1:8000/docs
echo UI:         http://localhost:8501  (opens by itself)
echo Close the API and UI windows to stop.
timeout /t 10 >nul
