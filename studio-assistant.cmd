@echo off
rem The CLI. `py` is the Windows Python launcher, and unlike a bare `python`
rem it is not whatever a conda shell or a Store alias put on the PATH first.
setlocal

where py.exe >nul 2>&1
if %errorlevel%==0 (
    py "%~dp0studio_agent.py" %*
    exit /b %errorlevel%
)

where python.exe >nul 2>&1
if %errorlevel%==0 (
    python "%~dp0studio_agent.py" %*
    exit /b %errorlevel%
)

echo Studio Assist needs Python 3 and could not find it on this PC.
exit /b 1
