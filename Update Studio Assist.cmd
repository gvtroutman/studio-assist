@echo off
rem Double-click once: pull the latest main from GitHub now, then register the
rem scheduled task that keeps pulling every 5 minutes (studio_update.py).
rem Safe to run again - it only fast-forwards, and re-registering the task
rem replaces the old one.
setlocal
cd /d "%~dp0"

where git.exe >nul 2>&1
if not %errorlevel%==0 (
    echo Git is not installed. Get it from https://git-scm.com/download/win
    pause
    exit /b 1
)

rem studio_update.py follows main and pauses on other branches. It never
rem switches this folder's branch. Use the same checks as the app's Update button.
echo Pulling the latest main from GitHub, then checking every 5 minutes...
where py.exe >nul 2>&1
if %errorlevel%==0 (
    py studio_update.py --install
) else (
    python studio_update.py --install
)
echo.
pause
