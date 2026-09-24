@echo off
rem Start the app with no console window.
rem
rem This line used to name a Python by its full install path
rem (...\Programs\Python\Python312\pythonw.exe). That is exactly one Python
rem upgrade away from a shortcut that does nothing at all when you click it -
rem no window, no error, because pythonw.exe has no console to complain in.
rem `pyw` is the Windows Python launcher: it ships with Python, lives in
rem System32, and keeps working when the version behind it changes.
setlocal

where pyw.exe >nul 2>&1
if %errorlevel%==0 (
    start "" pyw.exe "%~dp0studio_chat.py"
    exit /b 0
)

where pythonw.exe >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw.exe "%~dp0studio_chat.py"
    exit /b 0
)

echo Studio Assist needs Python 3 and could not find it on this PC.
echo.
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH", then run this again.
echo.
pause
exit /b 1
