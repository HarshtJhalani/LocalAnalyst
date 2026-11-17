@echo off
REM Streamlit frontend: http://localhost:8501  (needs the API running)
cd /d "%~dp0"
title Text-to-SQL UI

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting the UI: http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run src/ui/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

echo.
echo The UI stopped.
pause
