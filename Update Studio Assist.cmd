@echo off
rem Double-click once: pull the latest from GitHub now, then register the
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

echo Pulling the latest from GitHub...
git pull --ff-only
if not %errorlevel%==0 (
    echo.
    echo The pull was refused - this folder has its own commits or edits that
    echo GitHub's version would overwrite. Nothing was changed.
    pause
    exit /b 1
)

echo.
where py.exe >nul 2>&1
if %errorlevel%==0 (
    py studio_update.py --install
) else (
    python studio_update.py --install
)
echo.
pause
