#!/bin/sh

# ======================================================================
# pyvosklivesubtitle PyPI build script
#
# Supported:
#   Linux:
#       x86_64
#       aarch64
#       armv7l
#
#   Android:
#       Termux native
#
#   macOS
#   Windows
#
# Linux wheels are repaired to manylinux.
# Android wheels use an Android platform tag.
#
# IMPORTANT:
#   Termux vs Linux is NOT detected from:
#       /data/data/com.termux
#       sys.prefix
#
#   The Python executable's ELF PT_INTERP is inspected instead.
#
#   Native Termux:
#       Bionic linker
#
#   Linux / proot-distro:
#       glibc linker
# ======================================================================

set -e


# ======================================================================
# Find Python
# ======================================================================

if [ -n "$PYTHON" ]; then
    PYTHON_CMD="$PYTHON"
#elif command -v python3.10 >/dev/null 2>&1; then
#    PYTHON_CMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 was not found."
    exit 1
fi


# ======================================================================
# Check Python version
# ======================================================================

echo
echo "Using Python:"
"$PYTHON_CMD" --version

"$PYTHON_CMD" -c '
import sys

if sys.version_info < (3, 10):
    print("ERROR: Python 3.10 or newer is required.")
    sys.exit(1)
'


# ======================================================================
# Detect Python runtime
#
# This uses the ELF PT_INTERP of the Python executable.
#
# Android/Bionic:
#     /system/bin/linker
#     /system/bin/linker64
#
# Linux/glibc:
#     /lib/ld-linux-*.so.*
#     /lib64/ld-linux-*.so.*
#     /usr/lib/.../ld-linux-*.so.*
#
# The detection is performed by the SAME Python executable that will
# run setup.py.
# ======================================================================

PLATFORM_INFO="$("$PYTHON_CMD" -c '
import os
import platform
import struct
import sys


def get_elf_interpreter(filename):
    try:
        with open(filename, "rb") as f:
            data = f.read()
    except Exception:
        return ""


    if len(data) < 64:
        return ""


    if data[:4] != b"\x7fELF":
        return ""


    elf_class = data[4]
    endian = data[5]

    if endian == 1:
        byte_order = "<"
    elif endian == 2:
        byte_order = ">"
    else:
        return ""


    try:
        if elf_class == 2:
            # ELF64
            e_phoff = struct.unpack_from(
                byte_order + "Q",
                data,
                32
            )[0]

            e_phentsize = struct.unpack_from(
                byte_order + "H",
                data,
                54
            )[0]

            e_phnum = struct.unpack_from(
                byte_order + "H",
                data,
                56
            )[0]

            PT_INTERP = 3

            for i in range(e_phnum):
                offset = e_phoff + i * e_phentsize

                if offset + e_phentsize > len(data):
                    break

                p_type = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset
                )[0]

                if p_type != PT_INTERP:
                    continue

                p_offset = struct.unpack_from(
                    byte_order + "Q",
                    data,
                    offset + 8
                )[0]

                p_filesz = struct.unpack_from(
                    byte_order + "Q",
                    data,
                    offset + 32
                )[0]

                value = data[
                    p_offset:p_offset + p_filesz
                ]

                return value.rstrip(b"\0").decode(
                    "utf-8",
                    "replace"
                )


        elif elf_class == 1:
            # ELF32
            e_phoff = struct.unpack_from(
                byte_order + "I",
                data,
                28
            )[0]

            e_phentsize = struct.unpack_from(
                byte_order + "H",
                data,
                42
            )[0]

            e_phnum = struct.unpack_from(
                byte_order + "H",
                data,
                44
            )[0]

            PT_INTERP = 3

            for i in range(e_phnum):
                offset = e_phoff + i * e_phentsize

                if offset + e_phentsize > len(data):
                    break

                p_type = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset
                )[0]

                if p_type != PT_INTERP:
                    continue

                p_offset = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset + 4
                )[0]

                p_filesz = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset + 16
                )[0]

                value = data[
                    p_offset:p_offset + p_filesz
                ]

                return value.rstrip(b"\0").decode(
                    "utf-8",
                    "replace"
                )

    except Exception:
        return ""

    return ""


executable = os.path.realpath(sys.executable)
interpreter = get_elf_interpreter(executable)

system = platform.system()
machine = platform.machine().lower()


if system == "Darwin":
    os_name = "Darwin"

elif system == "Windows":
    os_name = "Windows"

elif system == "Linux":

    # Native Android/Bionic ELF loader.
    if interpreter in (
        "/system/bin/linker",
        "/system/bin/linker64",
    ):
        os_name = "Android"

    # A glibc Linux interpreter.
    elif (
        "ld-linux" in os.path.basename(interpreter)
        or "ld-linux" in interpreter
    ):
        os_name = "Linux"

    else:
        # Do not silently classify an unknown Linux runtime as Android.
        # This prevents accidentally putting an Android binary into
        # a Linux wheel.
        os_name = "Linux"

else:
    os_name = system


print("OS_NAME=" + os_name)
print("CPU_ARCH=" + machine)
print("PYTHON_EXECUTABLE=" + executable)
print("ELF_INTERPRETER=" + (interpreter or "UNKNOWN"))
' )"


# ======================================================================
# Read platform information
# ======================================================================

OS_NAME="$(printf '%s\n' "$PLATFORM_INFO" | sed -n 's/^OS_NAME=//p')"
CPU_ARCH="$(printf '%s\n' "$PLATFORM_INFO" | sed -n 's/^CPU_ARCH=//p')"
PYTHON_EXECUTABLE="$(printf '%s\n' "$PLATFORM_INFO" | sed -n 's/^PYTHON_EXECUTABLE=//p')"
ELF_INTERPRETER="$(printf '%s\n' "$PLATFORM_INFO" | sed -n 's/^ELF_INTERPRETER=//p')"


echo
echo "========================"
echo "Python runtime detection"
echo "========================"
echo "Operating system : $OS_NAME"
echo "Architecture     : $CPU_ARCH"
echo "Python executable: $PYTHON_EXECUTABLE"
echo "ELF interpreter  : $ELF_INTERPRETER"
#echo "========================"


# ======================================================================
# Safety check
#
# If Linux is detected but the Python executable has no recognizable
# ELF interpreter, do not build a Linux wheel automatically.
#
# This is intentionally conservative.
# ======================================================================

if [ "$OS_NAME" = "Linux" ] && [ "$ELF_INTERPRETER" = "UNKNOWN" ]; then
    echo
    echo "ERROR: Unable to identify the Linux Python ELF interpreter."
    echo "Refusing to build a Linux wheel because the runtime could not"
    echo "be verified as a normal Linux/glibc environment."
    exit 1
fi


# ======================================================================
# Clean previous build
# ======================================================================

echo
echo "Cleaning previous build files..."
rm -rf build dist *.egg-info

# ======================================================================
# Build tools
# ======================================================================

echo
echo "Updating setuptools and wheel..."

"$PYTHON_CMD" -m pip install --upgrade setuptools wheel


# ======================================================================
# Build source distribution
#
# setup.py deliberately includes ALL native libraries for sdist.
# ======================================================================

echo
echo "Building source distribution..."

"$PYTHON_CMD" setup.py sdist


# ======================================================================
# Build platform-specific wheel
# ======================================================================

case "$OS_NAME" in


    # ==================================================================
    # macOS
    # ==================================================================

    Darwin)

        echo
        echo "Detected macOS."

        if [ "$CPU_ARCH" = "x86_64" ]; then

            echo "Building macOS 10.15 x86_64 wheel..."

            "$PYTHON_CMD" setup.py bdist_wheel \
                --plat-name macosx_10_15_x86_64

        else

            echo "Building automatic macOS wheel..."

            "$PYTHON_CMD" setup.py bdist_wheel

        fi

        ;;


    # ==================================================================
    # Windows
    # ==================================================================

    Windows)

        echo
        echo "Detected Windows."

        "$PYTHON_CMD" setup.py bdist_wheel

        ;;


    # ==================================================================
    # Linux
    # ==================================================================

    Linux)

        echo
        echo "Detected Linux ($CPU_ARCH)."
        echo "Linux wheel will be repaired to manylinux."

        case "$CPU_ARCH" in

            x86_64|amd64)

                MANYLINUX_PLAT="manylinux_2_17_x86_64"
                NATIVE_WHEEL="dist/*linux_x86_64.whl"

                ;;


            aarch64|arm64)

                MANYLINUX_PLAT="manylinux_2_17_aarch64"
                NATIVE_WHEEL="dist/*linux_aarch64.whl"

                ;;


            armv7l|armv7|armhf)

                MANYLINUX_PLAT="manylinux_2_17_armv7l"
                NATIVE_WHEEL="dist/*linux_armv7l.whl"

                ;;


            *)

                echo
                echo "ERROR: Unsupported Linux architecture: $CPU_ARCH"
                exit 1

                ;;

        esac


        # --------------------------------------------------------------
        # Build native Linux wheel.
        # --------------------------------------------------------------

        echo
        echo "Building native Linux wheel..."

        "$PYTHON_CMD" setup.py bdist_wheel


        # --------------------------------------------------------------
        # Verify that the wheel contains the Linux Vosk library.
        # --------------------------------------------------------------

        echo
        echo "Checking Linux wheel contents..."

        WHEEL_FILE="$(ls $NATIVE_WHEEL 2>/dev/null | head -n 1 || true)"

        if [ -z "$WHEEL_FILE" ]; then
            echo "ERROR: Native Linux wheel was not created."
            exit 1
        fi

        case "$CPU_ARCH" in

            x86_64|amd64)
                EXPECTED_LIB="libvosk.so"
                ;;

            aarch64|arm64)
                EXPECTED_LIB="libvosk_linux_aarch64.so"
                ;;

            armv7l|armv7|armhf)
                EXPECTED_LIB="libvosk_linux_armv7l.so"
                ;;

        esac


        if ! unzip -l "$WHEEL_FILE" | grep -q "$EXPECTED_LIB"; then
            echo
            echo "ERROR: Linux wheel does not contain:"
            echo "  $EXPECTED_LIB"
            echo
            echo "Wheel contents:"
            unzip -l "$WHEEL_FILE"
            exit 1
        fi


        # Android libraries must NEVER enter a Linux wheel.
        if unzip -l "$WHEEL_FILE" | grep -q 'libvosk_android-'; then
            echo
            echo "ERROR: Android Vosk library detected inside Linux wheel!"
            echo
            unzip -l "$WHEEL_FILE" | grep 'libvosk'
            exit 1
        fi


        # --------------------------------------------------------------
        # auditwheel
        # --------------------------------------------------------------

        echo
        echo "Checking auditwheel..."

        if ! command -v auditwheel >/dev/null 2>&1; then
            "$PYTHON_CMD" -m pip install --upgrade auditwheel
        fi


        # --------------------------------------------------------------
        # patchelf
        # --------------------------------------------------------------

        echo
        echo "Checking patchelf..."

        if ! command -v patchelf >/dev/null 2>&1; then
            echo
            echo "ERROR: patchelf is required by auditwheel."
            echo
            echo "Install it with:"
            echo "    apt install patchelf"
            exit 1
        fi


        # --------------------------------------------------------------
        # Show auditwheel analysis before repair.
        # --------------------------------------------------------------

        echo
        echo "Running auditwheel show..."

        "$PYTHON_CMD" -m auditwheel show "$WHEEL_FILE"


        # --------------------------------------------------------------
        # Repair wheel.
        # --------------------------------------------------------------

        echo
        echo "Running auditwheel repair..."
        echo "Target platform : $MANYLINUX_PLAT"
        echo "Input wheel     : $WHEEL_FILE"

        rm -rf dist/repaired
        mkdir -p dist/repaired

        "$PYTHON_CMD" -m auditwheel repair \
            --plat "$MANYLINUX_PLAT" \
            --wheel-dir dist/repaired \
            "$WHEEL_FILE"


        # --------------------------------------------------------------
        # Replace native wheel with repaired manylinux wheel.
        # --------------------------------------------------------------

        echo
        echo "Replacing native Linux wheel with repaired manylinux wheel..."

        rm -f "$WHEEL_FILE"

        REPAIRED_WHEEL="$(ls dist/repaired/*.whl | head -n 1)"

        if [ -z "$REPAIRED_WHEEL" ]; then
            echo "ERROR: auditwheel did not produce a repaired wheel."
            exit 1
        fi

        mv "$REPAIRED_WHEEL" dist/

        rm -rf dist/repaired

        ;;


    # ==================================================================
    # Android / native Termux
    # ==================================================================

    Android)

        echo
        echo "Detected native Android / Termux ($CPU_ARCH)."
        echo "Building Android wheel."

        case "$CPU_ARCH" in

            aarch64|arm64)

                ANDROID_PLAT="android_24_arm64_v8a"
                EXPECTED_LIB="libvosk_android-arm64-v8a.so"

                ;;


            armv7l|armv7|armhf)

                ANDROID_PLAT="android_24_armeabi_v7a"
                EXPECTED_LIB="libvosk_android-armeabi-v7a.so"

                ;;


            x86_64|amd64)

                ANDROID_PLAT="android_24_x86_64"
                EXPECTED_LIB="libvosk_android-x86_64.so"

                ;;


            x86|i686|i386)

                ANDROID_PLAT="android_24_x86"
                EXPECTED_LIB="libvosk_android-x86.so"

                ;;


            *)

                echo
                echo "ERROR: Unsupported Android architecture: $CPU_ARCH"
                exit 1

                ;;

        esac


        echo
        echo "Android platform tag : $ANDROID_PLAT"
        echo "Expected Vosk library: $EXPECTED_LIB"


        "$PYTHON_CMD" setup.py bdist_wheel \
            --plat-name "$ANDROID_PLAT"


        # --------------------------------------------------------------
        # Verify Android wheel contents.
        # --------------------------------------------------------------

        WHEEL_FILE="$(ls dist/*.whl | head -n 1)"

        echo
        echo "Checking Android wheel contents..."

        if ! unzip -l "$WHEEL_FILE" | grep -q "$EXPECTED_LIB"; then
            echo
            echo "ERROR: Android wheel does not contain:"
            echo "  $EXPECTED_LIB"
            echo
            unzip -l "$WHEEL_FILE"
            exit 1
        fi


        # Linux libraries must NEVER enter an Android wheel.
        if unzip -l "$WHEEL_FILE" | grep -q 'libvosk_linux_'; then
            echo
            echo "ERROR: Linux Vosk library detected inside Android wheel!"
            echo
            unzip -l "$WHEEL_FILE" | grep 'libvosk'
            exit 1
        fi

        ;;


    # ==================================================================
    # Unsupported
    # ==================================================================

    *)

        echo
        echo "ERROR: Unsupported operating system: $OS_NAME"
        exit 1

        ;;

esac


# ======================================================================
# Show resulting distributions
# ======================================================================

echo
echo "======================="
echo "Generated distributions"
echo "======================="

ls -lh dist/


# ======================================================================
# Verify wheel metadata
# ======================================================================

echo
echo "Checking distributions with twine..."

if command -v twine >/dev/null 2>&1; then
    twine check dist/*
else
    echo "WARNING: twine is not installed."
    echo "Install it with:"
    echo "    pip install twine"
fi


echo
echo "================"
echo "BUILD SUCCESSFUL"
echo "================"
