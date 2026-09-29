@echo off
rem Serve Studio Assist to a phone: chat and pictures as a web page, over the
rem tailnet. The window stays open - it shows the link to open on the phone,
rem and closing it stops the server. Add --lan after the file name below to
rem serve the home network too, behind a passcode.
rem
rem `py` is the Windows Python launcher; see "Studio Assist.cmd" for why no
rem Python is named by its install path.
setlocal
title Studio Assist Phone

where py.exe >nul 2>&1
if %errorlevel%==0 (
    py.exe "%~dp0apps\phone\server.py" %*
    goto done
)

where python.exe >nul 2>&1
if %errorlevel%==0 (
    python.exe "%~dp0apps\phone\server.py" %*
    goto done
)

echo Studio Assist needs Python 3 and could not find it on this PC.
echo.
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH", then run this again.

:done
echo.
pause
