@echo off
rem LoRA Casino on its own (Windows): starts its small server and opens the page in your browser.
rem Start ComfyUI too (before or after): the pictures are drawn through it. Close this window to stop.
setlocal
title LoRA Casino
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" call :try ".venv\Scripts\python.exe"
if not defined PY call :try py -3
if not defined PY call :try python
rem ComfyUI Desktop ships a Python that already has numpy
if not defined PY for /d %%D in ("%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Installs\*") do if not defined PY if exist "%%D\standalone-env\python.exe" call :try "%%D\standalone-env\python.exe"
if defined PY goto run

rem a Python without numpy: give LoRA Casino its own small environment (once)
set "BASEPY="
call :plain py -3
if not defined BASEPY call :plain python
if not defined BASEPY goto nopython
echo Setting up LoRA Casino (once): a private Python environment with numpy...
%BASEPY% -m venv .venv || goto nopython
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto nopython
set PY=".venv\Scripts\python.exe"

:run
%PY% standalone.py 8777 --open
if errorlevel 1 pause
exit /b

:nopython
echo.
echo LoRA Casino needs Python 3.10 or newer with numpy.
echo Install Python from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run this again.
pause
exit /b 1

:try
%* -c "import sys, numpy; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set PY=%*
exit /b 0

:plain
%* -c "import sys, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set BASEPY=%*
exit /b 0
