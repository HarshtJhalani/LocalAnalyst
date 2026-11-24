@echo off
REM Makes sure Ollama is installed and its server is running.
REM Used by setup.bat and the run_*.bat scripts. Sets OLLAMA_EXE.
REM Exit code 0 = running, 1 = not installed or did not start.

set "OLLAMA_EXE="
for /f "delims=" %%i in ('where ollama 2^>nul') do if not defined OLLAMA_EXE set "OLLAMA_EXE=%%i"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

if not defined OLLAMA_EXE (
    echo Ollama is not installed.
    exit /b 1
)

REM Already running?
"%OLLAMA_EXE%" list >nul 2>&1
if not errorlevel 1 exit /b 0

echo Starting Ollama...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
) else (
    start "Ollama server" /min "%OLLAMA_EXE%" serve
)

set /a ollama_tries=0
:wait_ollama
timeout /t 2 /nobreak >nul
"%OLLAMA_EXE%" list >nul 2>&1
if not errorlevel 1 exit /b 0
set /a ollama_tries+=1
if %ollama_tries% lss 20 goto :wait_ollama

echo Ollama did not start. Open the Ollama app from the Start menu, then try again.
exit /b 1
