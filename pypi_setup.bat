@echo off
rem ======================================================================
rem pyvosklivesubtitle PyPI build script (Windows 10/11)
rem
rem Builds the source distribution and the Windows wheel.
rem Linux, Android/Termux and macOS wheels are built with pypi_setup.sh.
rem
rem Python selection priority:
rem   1. %PYTHON%        explicit override, for example: set PYTHON=C:\Python314\python.exe
rem   2. %VIRTUAL_ENV%   active virtual environment
rem   3. python / python3 on PATH, then the "py -3" launcher
rem
rem Python 3.10 or newer is required.
rem ======================================================================

setlocal EnableExtensions

rem Always work from the folder that contains this script
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem Find Python
rem ----------------------------------------------------------------------
set "PYTHON_CMD="
set "PY_EXE="

if defined PYTHON goto :use_override
if defined VIRTUAL_ENV goto :use_venv
goto :use_system

:use_override
call :python_ok "%PYTHON%"
if errorlevel 1 goto :err_override
set "PYTHON_CMD=%PYTHON%"
goto :python_found

:err_override
echo.
echo ERROR: "%PYTHON%" was not found or is older than Python 3.10.
goto :fail

:use_venv
set "VENV_PY=%VIRTUAL_ENV%\Scripts\python.exe"
if not exist "%VENV_PY%" goto :err_venv_missing
call :python_ok "%VENV_PY%"
if errorlevel 1 goto :err_venv_old
set "PYTHON_CMD=%VENV_PY%"
goto :python_found

:err_venv_missing
echo.
echo ERROR: VIRTUAL_ENV is set but no python.exe was found in:
echo        %VIRTUAL_ENV%\Scripts
goto :fail

:err_venv_old
echo.
echo ERROR: The active virtual environment uses Python older than 3.10:
echo        %VIRTUAL_ENV%
goto :fail

:use_system
for %%C in (python python3) do (
    if not defined PYTHON_CMD (
        call :python_ok "%%C"
        if not errorlevel 1 set "PYTHON_CMD=%%C"
    )
)
if defined PYTHON_CMD goto :python_found

rem Fall back to the Windows "py" launcher
where py >nul 2>&1
if errorlevel 1 goto :err_no_python
for /f "usebackq delims=" %%P in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do set "PY_EXE=%%P"
if not defined PY_EXE goto :err_no_python
call :python_ok "%PY_EXE%"
if errorlevel 1 goto :err_no_python
set "PYTHON_CMD=%PY_EXE%"
goto :python_found

:err_no_python
echo.
echo ERROR: Python 3.10 or newer was not found.
goto :fail

:python_found

echo.
echo Using Python:
"%PYTHON_CMD%" --version
if errorlevel 1 goto :fail

rem ----------------------------------------------------------------------
rem Python runtime detection
rem ----------------------------------------------------------------------
echo.
echo ========================
echo Python runtime detection
echo ========================
"%PYTHON_CMD%" -c "import os, platform, sys; print('Operating system : ' + platform.system()); print('Architecture     : ' + platform.machine()); print('Python executable: ' + os.path.realpath(sys.executable)); sys.exit(0 if platform.system() == 'Windows' else 1)"
if errorlevel 1 goto :err_not_windows
goto :platform_ok

:err_not_windows
echo.
echo ERROR: This script builds the Windows wheel and must run on Windows.
echo        Use pypi_setup.sh on Linux, Termux and macOS.
goto :fail

:platform_ok

rem ----------------------------------------------------------------------
rem Clean previous build
rem ----------------------------------------------------------------------
echo.
echo Cleaning previous build files...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%D in (*.egg-info) do rmdir /s /q "%%D"

rem ----------------------------------------------------------------------
rem Build tools
rem ----------------------------------------------------------------------
echo.
echo Updating setuptools and wheel...
"%PYTHON_CMD%" -m pip install --upgrade setuptools wheel
if errorlevel 1 goto :fail

rem ----------------------------------------------------------------------
rem Build source distribution
rem setup.py deliberately includes ALL native libraries for sdist.
rem ----------------------------------------------------------------------
echo.
echo Building source distribution...
"%PYTHON_CMD%" setup.py sdist
if errorlevel 1 goto :fail

rem ----------------------------------------------------------------------
rem Build Windows wheel
rem ----------------------------------------------------------------------
echo.
echo Detected Windows.
echo Building Windows wheel...
"%PYTHON_CMD%" setup.py bdist_wheel
if errorlevel 1 goto :fail

rem ----------------------------------------------------------------------
rem Verify wheel contents:
rem   - must contain libvosk.dll
rem   - must NOT contain Linux, Android or macOS Vosk libraries
rem ----------------------------------------------------------------------
echo.
echo Checking Windows wheel contents...
"%PYTHON_CMD%" -c "import glob,sys,zipfile; w=sorted(glob.glob('dist/*.whl')); sys.exit('ERROR: No wheel was created.') if not w else None; n=zipfile.ZipFile(w[0]).namelist(); sys.exit('ERROR: Windows wheel does not contain libvosk.dll') if not any(x.endswith('libvosk.dll') for x in n) else None; bad=[x for x in n if 'libvosk_linux_' in x or 'libvosk_android-' in x or x.endswith('libvosk.so') or x.endswith('libvosk.dyld')]; sys.exit('ERROR: Non-Windows Vosk library detected inside Windows wheel: ' + ', '.join(bad)) if bad else None; print('Wheel OK: ' + w[0])"
if errorlevel 1 goto :fail

rem ----------------------------------------------------------------------
rem Show resulting distributions
rem ----------------------------------------------------------------------
echo.
echo =======================
echo Generated distributions
echo =======================
dir dist

rem ----------------------------------------------------------------------
rem Verify distributions with twine
rem ----------------------------------------------------------------------
echo.
echo Checking distributions with twine...
"%PYTHON_CMD%" -m twine --version >nul 2>&1
if errorlevel 1 goto :no_twine
"%PYTHON_CMD%" -m twine check dist/*
if errorlevel 1 goto :fail
goto :success

:no_twine
echo WARNING: twine is not installed.
echo Install it with:
echo     pip install twine

:success
echo.
echo ================
echo BUILD SUCCESSFUL
echo ================
endlocal
exit /b 0

:fail
echo.
echo ================
echo BUILD FAILED
echo ================
endlocal
exit /b 1

rem ----------------------------------------------------------------------
rem Subroutine: succeed if the given command exists and is Python >= 3.10
rem ----------------------------------------------------------------------
:python_ok
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
exit /b %errorlevel%
