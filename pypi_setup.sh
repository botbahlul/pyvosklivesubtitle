#!/bin/sh

set -e

# ============================================================
# pyvosklivesubtitle PyPI build script
#
# Linux  -> build linux wheel -> auditwheel repair -> manylinux
# macOS  -> build native macOS wheel
# Windows -> build native Windows wheel
# ============================================================


# ------------------------------------------------------------
# Detect platform
# ------------------------------------------------------------

UNAME="$(uname -s 2>/dev/null || echo unknown)"

case "$UNAME" in
    Linux*)
        PLATFORM="Linux"
        ;;

    Darwin*)
        PLATFORM="Darwin"
        ;;

    MINGW*|MSYS*|CYGWIN*)
        PLATFORM="Windows"
        ;;

    *)
        echo "ERROR: Unsupported platform: $UNAME"
        exit 1
        ;;
esac


echo "=============================================="
echo " pyvosklivesubtitle PyPI build"
echo " Platform : $PLATFORM"
echo "=============================================="


# ------------------------------------------------------------
# Select Python
# ------------------------------------------------------------

if [ "$PLATFORM" = "Windows" ]; then

    if command -v py >/dev/null 2>&1; then
        PYTHON="py -3.10"
    elif command -v python3.10 >/dev/null 2>&1; then
        PYTHON="python3.10"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    else
        echo "ERROR: Python 3.10+ not found."
        exit 1
    fi

else

    if command -v python3.10 >/dev/null 2>&1; then
        PYTHON="python3.10"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    else
        echo "ERROR: Python 3.10+ not found."
        exit 1
    fi

fi


echo "Python   : $PYTHON"


# ------------------------------------------------------------
# Clean previous build
# ------------------------------------------------------------

echo
echo "[1/5] Cleaning old build files..."

rm -rf build
rm -rf dist
rm -rf pyvosklivesubtitle.egg-info
rm -rf vosk_autosrt.egg-info


# ------------------------------------------------------------
# Upgrade packaging tools
# ------------------------------------------------------------

echo
echo "[2/5] Updating build tools..."

$PYTHON -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    build


# ------------------------------------------------------------
# Build source distribution
# ------------------------------------------------------------

echo
echo "[3/5] Building source distribution..."

$PYTHON -m build --sdist


# ------------------------------------------------------------
# Build platform wheel
# ------------------------------------------------------------

echo
echo "[4/5] Building platform wheel..."

$PYTHON -m build --wheel


# ------------------------------------------------------------
# Linux
#
# Convert linux_x86_64 wheel to manylinux.
# auditwheel does the actual compatibility check and repair.
# ------------------------------------------------------------

if [ "$PLATFORM" = "Linux" ]; then

    echo
    echo "[5/5] Linux detected."

    if ! command -v auditwheel >/dev/null 2>&1; then
        echo
        echo "ERROR: auditwheel is not installed."
        echo
        echo "Install it with:"
        echo
        echo "    $PYTHON -m pip install auditwheel"
        echo
        exit 1
    fi

    mkdir -p dist/repaired

    echo
    echo "Running auditwheel..."

    auditwheel show dist/*.whl

    auditwheel repair \
        --plat manylinux_2_17_x86_64 \
        -w dist/repaired \
        dist/*.whl

    echo
    echo "=============================================="
    echo " Linux wheel:"
    echo "=============================================="

    ls -lh dist/repaired/

    echo
    echo "IMPORTANT:"
    echo "Upload the wheel from dist/repaired/"
    echo "NOT the original wheel in dist/."

    exit 0
fi


# ------------------------------------------------------------
# macOS
#
# Do NOT use manylinux on macOS.
# Native macOS platform tag is generated automatically.
# ------------------------------------------------------------

if [ "$PLATFORM" = "Darwin" ]; then

    echo
    echo "[5/5] macOS detected."

    echo
    echo "macOS wheels:"
    ls -lh dist/*.whl

    echo
    echo "No auditwheel required."
    echo "Upload the macOS wheel from dist/."

    exit 0
fi


# ------------------------------------------------------------
# Windows
#
# Do NOT use manylinux on Windows.
# Native Windows platform tag is generated automatically.
# ------------------------------------------------------------

if [ "$PLATFORM" = "Windows" ]; then

    echo
    echo "[5/5] Windows detected."

    echo
    echo "Windows wheels:"
    ls -lh dist/*.whl

    echo
    echo "No auditwheel required."
    echo "Upload the Windows wheel from dist/."

    exit 0
fi
