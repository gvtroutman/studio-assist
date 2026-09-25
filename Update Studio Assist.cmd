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

rem studio_update.py does the pull: it follows main, and moves this folder
rem off an old, merged branch onto main when that loses nothing. A plain
rem "git pull" here pulled whatever branch was checked out, and a merged
rem branch never changes again.
echo Pulling the latest main from GitHub, then checking every 5 minutes...
where py.exe >nul 2>&1
if %errorlevel%==0 (
    py studio_update.py --install
) else (
    python studio_update.py --install
)
echo.
pause
