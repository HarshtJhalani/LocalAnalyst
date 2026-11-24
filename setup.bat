@echo off
REM One-time setup on a fresh Windows computer:
REM   1. Python 3.11+       (checked; how to install it if missing)
REM   2. .venv + packages   (requirements.txt)
REM   3. .env               (from .env.example)
REM   4. Ollama             (installed with winget if you agree, then started)
REM   5. The model that fits this computer:
REM        NVIDIA GPU with 6 GB or more -> qwen2.5:7b-instruct (MODEL_PROFILE=gpu)
REM        otherwise                    -> qwen2.5:3b-instruct (MODEL_PROFILE=cpu)
REM   6. A quick self-test  (python -m pytest)
REM Safe to run again: steps already done are skipped.
setlocal
cd /d "%~dp0"

echo === 1/6 Python ===
set "PY="
for %%c in ("py -3.13" "py -3.12" "py -3.11" "py -3" "python") do (
    if not defined PY (
        %%~c -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=%%~c"
    )
)
if not defined PY (
    echo Python 3.11 or newer was not found.
    echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
    echo or run:  winget install Python.Python.3.12
    echo Then open a new terminal and run setup.bat again.
    goto :error
)
for /f "delims=" %%v in ('%PY% -c "import sys; print(sys.version.split()[0])"') do echo Using Python %%v ^(%PY%^)

echo.
echo === 2/6 Virtual environment and packages ===
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    %PY% -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet --retries 10 --timeout 60 || goto :error
echo Installing packages ^(first time: a few minutes^)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet --retries 10 --timeout 60
if errorlevel 1 (
    echo Retrying without pip's download cache...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet --retries 10 --timeout 60 --no-cache-dir || goto :error
)

echo.
echo === 3/6 Settings ===
set "NEW_ENV="
if not exist ".env" (
    copy ".env.example" ".env" >nul
    set "NEW_ENV=1"
    echo Created .env from .env.example
) else (
    echo .env already exists; kept as it is.
)

echo.
echo === 4/6 Ollama ===
call "scripts\ensure_ollama.bat"
if errorlevel 1 (
    if not defined OLLAMA_EXE (
        echo Ollama runs the AI model on this computer. Install it now with winget?
        choice /c YN /m "Install Ollama"
        if errorlevel 2 (
            echo Install it from https://ollama.com/download, then run setup.bat again.
            goto :error
        )
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements || goto :error
        call "scripts\ensure_ollama.bat" || goto :error
    ) else (
        goto :error
    )
)

echo.
echo === 5/6 AI model ===
set "VRAM=0"
for /f "delims=" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul') do if "%VRAM%"=="0" set "VRAM=%%m"
set /a VRAM_NUM=%VRAM% 2>nul
if %VRAM_NUM% geq 6000 (
    set "MODEL=qwen2.5:7b-instruct"
    set "PROFILE=gpu"
    echo NVIDIA GPU with %VRAM_NUM% MB: using the 7B model ^(about 4.7 GB download^).
) else (
    set "MODEL=qwen2.5:3b-instruct"
    set "PROFILE=cpu"
    echo No NVIDIA GPU with 6 GB or more: using the 3B model ^(about 1.9 GB download^).
)
"%OLLAMA_EXE%" pull %MODEL% || goto :error

REM A fresh .env gets the profile that matches the downloaded model.
if defined NEW_ENV (
    powershell -NoProfile -Command "(Get-Content .env) -replace '^MODEL_PROFILE=.*', 'MODEL_PROFILE=%PROFILE%' | Set-Content .env -Encoding utf8"
    echo Set MODEL_PROFILE=%PROFILE% in .env
) else (
    echo Your .env was kept; make sure MODEL_PROFILE fits the model ^(%PROFILE%^).
)

echo.
echo === 6/6 Self-test ===
".venv\Scripts\python.exe" -m pytest -q -p no:warnings
if errorlevel 1 (
    echo Some tests failed; the app may still work. See the messages above.
) else (
    echo All tests passed.
)

echo.
echo Setup complete. Start the app with run_all.bat
pause
exit /b 0

:error
echo.
echo Setup stopped. See the messages above, fix the problem and run setup.bat again.
pause
exit /b 1
