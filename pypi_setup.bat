@echo off
setlocal EnableExtensions

set "Py_DEBUG=1"

echo ==============================================
echo  pyvosklivesubtitle PyPI build
echo  Platform : Windows
echo ==============================================
echo.


REM ============================================================
REM Python detection
REM ============================================================

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3.10"
    goto :python_found
)

where python3.10 >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python3.10"
    goto :python_found
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :python_found
)

echo ERROR: Python 3.10+ was not found.
echo.
exit /b 1


:python_found

echo Python   : %PYTHON%
echo.


REM ============================================================
REM Check Python version
REM ============================================================

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"

if errorlevel 1 (
    echo ERROR: Python 3.10 or newer is required.
    echo.
    exit /b 1
)


REM ============================================================
REM Clean previous build
REM ============================================================

echo [1/5] Cleaning old build files...
echo.

if exist ".\build" (
    rmdir /s /q ".\build"
    if errorlevel 1 (
        echo ERROR: Failed to delete .\build
        exit /b 1
    )
)

if exist ".\dist" (
    rmdir /s /q ".\dist"
    if errorlevel 1 (
        echo ERROR: Failed to delete .\dist
        exit /b 1
    )
)

if exist ".\pyvosklivesubtitle.egg-info" (
    rmdir /s /q ".\pyvosklivesubtitle.egg-info"
    if errorlevel 1 (
        echo ERROR: Failed to delete .\pyvosklivesubtitle.egg-info
        exit /b 1
    )
)

if exist ".\vosk_autosrt.egg-info" (
    rmdir /s /q ".\vosk_autosrt.egg-info"
    if errorlevel 1 (
        echo ERROR: Failed to delete .\vosk_autosrt.egg-info
        exit /b 1
    )
)


REM ============================================================
REM Update build tools
REM ============================================================

echo.
echo [2/5] Updating build tools...
echo.

%PYTHON% -m pip install --upgrade pip setuptools wheel build

if errorlevel 1 (
    echo.
    echo ERROR: Failed to update build tools.
    exit /b 1
)


REM ============================================================
REM Build source distribution
REM ============================================================

echo.
echo [3/5] Building source distribution...
echo.

%PYTHON% -m build --sdist

if errorlevel 1 (
    echo.
    echo ERROR: Failed to build source distribution.
    exit /b 1
)


REM ============================================================
REM Build Windows wheel
REM
REM Do NOT use manylinux here.
REM Windows wheels are automatically tagged win_amd64
REM when using 64-bit Python.
REM ============================================================

echo.
echo [4/5] Building Windows wheel...
echo.

%PYTHON% -m build --wheel

if errorlevel 1 (
    echo.
    echo ERROR: Failed to build Windows wheel.
    exit /b 1
)


REM ============================================================
REM Verify output
REM ============================================================

echo.
echo [5/5] Build completed.
echo.

if not exist ".\dist" (
    echo ERROR: dist directory was not created.
    exit /b 1
)

echo ==============================================
echo  Generated packages:
echo ==============================================
echo.

dir /b ".\dist"

echo.
echo ==============================================
echo  Expected Windows wheel:
echo ==============================================
echo.
echo  pyvosklivesubtitle-X.Y.Z-cp310-cp310-win_amd64.whl
echo.
echo Upload the files from .\dist\
echo ==============================================
echo.

endlocal
exit /b 0
