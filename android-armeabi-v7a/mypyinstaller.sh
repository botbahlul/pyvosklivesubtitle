#!/bin/sh

# ======================================================================
# pyvosklivesubtitle PyInstaller build script
#
# Supported: Linux (x86_64, aarch64, armv7l), Android (native Termux),
#            macOS, Windows (Git Bash / MSYS2 / Cygwin)
#
# Termux vs Linux (proot-distro) is detected from the ELF PT_INTERP of
# the Python executable, NOT from /data/data/com.termux or sys.prefix,
# because those are also visible inside proot-distro.
# ======================================================================

# ----------------------------------------------------------------------
# Find Python
# ----------------------------------------------------------------------
# Return success if the given command exists and is Python >= 3.10
# Return success if the given command exists and is Python >= 3.10
python_ok() {
    command -v "$1" >/dev/null 2>&1 || return 1
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

PYTHON_CMD=""

if [ -n "$PYTHON" ]; then
    # 1. Explicit override
    if python_ok "$PYTHON"; then
        PYTHON_CMD="$PYTHON"
    else
        echo "ERROR: '$PYTHON' was not found or is older than Python 3.10."
        exit 1
    fi

elif [ -n "$VIRTUAL_ENV" ]; then
    # 2. Active virtual environment (venv / virtualenv)
    for candidate in \
        "$VIRTUAL_ENV/bin/python" \
        "$VIRTUAL_ENV/bin/python3" \
        "$VIRTUAL_ENV/Scripts/python.exe"
    do
        if [ -x "$candidate" ]; then
            if python_ok "$candidate"; then
                PYTHON_CMD="$candidate"
                break
            else
                echo "ERROR: The active virtual environment ($VIRTUAL_ENV)"
                echo "       uses Python older than 3.10."
                exit 1
            fi
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        echo "ERROR: VIRTUAL_ENV is set ($VIRTUAL_ENV) but no Python executable was found in it."
        exit 1
    fi

else
    # 3. System Python, in priority order
    for candidate in python3.10 python3 python; do
        if python_ok "$candidate"; then
            PYTHON_CMD="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10 or newer was not found."
    exit 1
fi

# ----------------------------------------------------------------------
# Check Python version
# ----------------------------------------------------------------------
echo
echo "Using Python:"
"$PYTHON_CMD" --version

"$PYTHON_CMD" -c '
import sys
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10 or newer is required.")
    sys.exit(1)
' || exit 1

# ----------------------------------------------------------------------
# Detect operating system
# ----------------------------------------------------------------------
OS_NAME="$(uname -s)"
CPU_ARCH="$(uname -m)"

# Normalize Windows uname output (Git Bash / MSYS2 / Cygwin)
case "$OS_NAME" in
    MINGW*|MSYS*|CYGWIN*) OS_NAME="Windows" ;;
esac

# Native Termux (Bionic) vs proot-distro / normal Linux (glibc)
if [ "$OS_NAME" = "Linux" ]; then
    PY_IS_ANDROID="$("$PYTHON_CMD" -c '
import os, struct, sys
def interp(path):
    try:
        d = open(path, "rb").read()
        if d[:4] != b"\x7fELF":
            return ""
        is64 = d[4] == 2
        bo = "<" if d[5] == 1 else ">"
        if is64:
            phoff = struct.unpack_from(bo + "Q", d, 32)[0]
            phentsize, phnum = struct.unpack_from(bo + "HH", d, 54)
        else:
            phoff = struct.unpack_from(bo + "I", d, 28)[0]
            phentsize, phnum = struct.unpack_from(bo + "HH", d, 42)
        for i in range(phnum):
            o = phoff + i * phentsize
            if struct.unpack_from(bo + "I", d, o)[0] != 3:
                continue
            if is64:
                po = struct.unpack_from(bo + "Q", d, o + 8)[0]
                ps = struct.unpack_from(bo + "Q", d, o + 32)[0]
            else:
                po = struct.unpack_from(bo + "I", d, o + 4)[0]
                ps = struct.unpack_from(bo + "I", d, o + 16)[0]
            return d[po:po + ps].rstrip(b"\0").decode("utf-8", "replace")
    except Exception:
        pass
    return ""
p = interp(os.path.realpath(sys.executable))
print("yes" if p in ("/system/bin/linker", "/system/bin/linker64") else "no")
' 2>/dev/null)"

    if [ "$PY_IS_ANDROID" = "yes" ]; then
        OS_NAME="Android"
    fi
fi

echo
echo "Operating system : $OS_NAME"
echo "Architecture     : $CPU_ARCH"

# ----------------------------------------------------------------------
# PyInstaller uses ':' as src/dest separator on POSIX, ';' on Windows
# ----------------------------------------------------------------------
if [ "$OS_NAME" = "Windows" ]; then
    PYI_SEP=";"
else
    PYI_SEP=":"
fi

# ----------------------------------------------------------------------
# Clean previous build
# ----------------------------------------------------------------------
echo
echo "Cleaning previous build files..."
rm -rf ./build ./dist
rm -f ./pyvosklivesubtitle.spec

# ----------------------------------------------------------------------
# Native library selection per platform / architecture
# ----------------------------------------------------------------------
case "$OS_NAME" in

    Windows)
        echo
        echo "Detected Windows ($CPU_ARCH)."
        LIBVOSK_FILES="libgcc_s_seh-1.dll libstdc++-6.dll libvosk.dll libwinpthread-1.dll"
        ;;

    Darwin)
        echo
        echo "Detected macOS ($CPU_ARCH)."
        case "$CPU_ARCH" in
            x86_64|arm64)
                LIBVOSK_FILES="libvosk.dyld"
                ;;
            *)
                echo "ERROR: Unsupported macOS architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    Linux)
        echo
        echo "Detected Linux ($CPU_ARCH)."
        case "$CPU_ARCH" in
            x86_64|amd64)
                LIBVOSK_FILES="libvosk.so"
                ;;
            aarch64|arm64)
                LIBVOSK_FILES="libvosk_linux_aarch64.so"
                ;;
            armv7l|armv7|armhf)
                LIBVOSK_FILES="libvosk_linux_armv7l.so"
                ;;
            *)
                echo "ERROR: Unsupported Linux architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    Android)
        echo
        echo "Detected Termux Android ($CPU_ARCH)."
        case "$CPU_ARCH" in
            aarch64|arm64)
                LIBVOSK_FILES="libvosk_android-arm64-v8a.so"
                ;;
            armv7l|armv7|armhf)
                LIBVOSK_FILES="libvosk_android-armeabi-v7a.so"
                ;;
            x86_64|amd64)
                LIBVOSK_FILES="libvosk_android-x86_64.so"
                ;;
            x86|i686|i386)
                LIBVOSK_FILES="libvosk_android-x86.so"
                ;;
            *)
                echo "ERROR: Unsupported Android architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    *)
        echo
        echo "ERROR: Unsupported operating system: $OS_NAME"
        exit 1
        ;;
esac

# Make sure every listed native library actually exists
for f in $LIBVOSK_FILES; do
    if [ ! -f "./$f" ]; then
        echo "ERROR: expected native library not found: ./$f"
        exit 1
    fi
done

# ----------------------------------------------------------------------
# Build the PyInstaller argument list (single run)
# ----------------------------------------------------------------------
set --

for f in $LIBVOSK_FILES; do
    set -- "$@" --add-data "./${f}${PYI_SEP}."
done

set -- "$@" \
    --add-data "./streamlink/plugins/*${PYI_SEP}streamlink/plugins" \
    --add-data "./streamlink/*${PYI_SEP}streamlink" \
    --collect-submodules streamlink.plugins \
    --hidden-import argparse \
    --hidden-import pysrt \
    --hidden-import six \
    --hidden-import progressbar \
    --hidden-import tqdm \
    --hidden-import requests \
    --hidden-import _cffi_backend \
    --hidden-import sounddevice \
    --noconsole \
    --windowed \
    --additional-hooks-dir=./ \
    --onefile pyvls.pyw

echo
echo "Running: $PYTHON_CMD -m PyInstaller $*"
echo

"$PYTHON_CMD" -m PyInstaller "$@"
