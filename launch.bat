@echo off
title Mold Design Tool
echo Starting Mold Design Tool...
echo.

cd /d "%~dp0backend"

:: Activate virtual environment if present
if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
) else if exist "%~dp0backend\venv\Scripts\activate.bat" (
    call "%~dp0backend\venv\Scripts\activate.bat"
)

:: Open the browser after a short delay (runs in background while uvicorn starts)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

echo Tool will open in your browser automatically.
echo Close this window to stop the server.
echo.

:: Run uvicorn in the foreground — closing this window stops the server
uvicorn app:app --host 127.0.0.1 --port 8000
