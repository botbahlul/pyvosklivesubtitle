from __future__ import unicode_literals

import os
import platform
import re
import struct
import sys
import warnings


# ======================================================================
# Suppress non-essential setuptools warnings
# ======================================================================

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="setuptools",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="setuptools",
)

warnings.filterwarnings(
    "ignore",
    message=".*is deprecated.*",
)


# ======================================================================
# setuptools
# ======================================================================

try:
    from setuptools import setup, find_packages
    from setuptools.dist import Distribution
except ImportError:
    print("ERROR: setuptools is required to build pyvosklivesubtitle.")
    print()
    print("Install it with:")
    print("    python -m pip install setuptools wheel")
    sys.exit(1)


# ======================================================================
# Python version
# ======================================================================

MIN_PYTHON = (3, 10)

if sys.version_info < MIN_PYTHON:
    print("THIS MODULE REQUIRES PYTHON 3.10+.")
    print(
        "YOU ARE CURRENTLY USING PYTHON {0}".format(
            platform.python_version()
        )
    )
    sys.exit(1)


# ======================================================================
# Package version
#
# DO NOT import pyvosklivesubtitle here.
#
# This is important for Python 3.13+, because audioop was removed from
# the standard library and the package may require audioop-lts.
# ======================================================================

PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pyvosklivesubtitle",
)

INIT_FILE = os.path.join(
    PACKAGE_DIR,
    "__init__.py",
)

with open(INIT_FILE, encoding="utf-8") as f:
    INIT_CONTENT = f.read()


VERSION_MATCH = re.search(
    r'^VERSION\s*=\s*[\'"]([^\'"]+)[\'"]',
    INIT_CONTENT,
    re.MULTILINE,
)

if not VERSION_MATCH:
    print(
        "ERROR: Unable to find VERSION in "
        "pyvosklivesubtitle/__init__.py"
    )
    sys.exit(1)


VERSION = VERSION_MATCH.group(1)


# ======================================================================
# ELF PT_INTERP detection
#
# This is the important part.
#
# We do NOT use:
#
#   /data/data/com.termux
#   sys.prefix
#   PREFIX
#
# as the primary Termux detector.
#
# Instead we inspect the ELF interpreter of the Python executable that
# is actually running setup.py.
#
# Native Android:
#
#   /system/bin/linker
#   /system/bin/linker64
#
# Linux/glibc:
#
#   /lib/ld-linux-*.so.*
#   /lib64/ld-linux-*.so.*
#   /usr/lib/.../ld-linux-*.so.*
#
# Therefore:
#
#   native Termux Python -> Android
#   proot-distro Python  -> Linux
#
# even if /data/data/com.termux is visible inside proot.
# ======================================================================

def get_elf_interpreter(filename):
    """
    Return the ELF PT_INTERP path of an executable.

    Supports ELF32 and ELF64, little-endian and big-endian.

    Returns:
        str
            Interpreter path.

        ""
            If the file is not a usable ELF executable or PT_INTERP
            cannot be found.
    """

    try:
        with open(filename, "rb") as f:
            data = f.read()
    except Exception:
        return ""


    if len(data) < 20:
        return ""


    # ELF magic.
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


    PT_INTERP = 3


    try:

        # --------------------------------------------------------------
        # ELF64
        # --------------------------------------------------------------

        if elf_class == 2:

            if len(data) < 64:
                return ""

            e_phoff = struct.unpack_from(
                byte_order + "Q",
                data,
                32,
            )[0]

            e_phentsize = struct.unpack_from(
                byte_order + "H",
                data,
                54,
            )[0]

            e_phnum = struct.unpack_from(
                byte_order + "H",
                data,
                56,
            )[0]


            for index in range(e_phnum):

                offset = (
                    e_phoff
                    + index * e_phentsize
                )

                if offset + e_phentsize > len(data):
                    break


                p_type = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset,
                )[0]


                if p_type != PT_INTERP:
                    continue


                p_offset = struct.unpack_from(
                    byte_order + "Q",
                    data,
                    offset + 8,
                )[0]

                p_filesz = struct.unpack_from(
                    byte_order + "Q",
                    data,
                    offset + 32,
                )[0]


                value = data[
                    p_offset:p_offset + p_filesz
                ]


                return value.rstrip(
                    b"\0"
                ).decode(
                    "utf-8",
                    "replace",
                )


        # --------------------------------------------------------------
        # ELF32
        # --------------------------------------------------------------

        elif elf_class == 1:

            if len(data) < 52:
                return ""

            e_phoff = struct.unpack_from(
                byte_order + "I",
                data,
                28,
            )[0]

            e_phentsize = struct.unpack_from(
                byte_order + "H",
                data,
                42,
            )[0]

            e_phnum = struct.unpack_from(
                byte_order + "H",
                data,
                44,
            )[0]


            for index in range(e_phnum):

                offset = (
                    e_phoff
                    + index * e_phentsize
                )

                if offset + e_phentsize > len(data):
                    break


                p_type = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset,
                )[0]


                if p_type != PT_INTERP:
                    continue


                p_offset = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset + 4,
                )[0]

                p_filesz = struct.unpack_from(
                    byte_order + "I",
                    data,
                    offset + 16,
                )[0]


                value = data[
                    p_offset:p_offset + p_filesz
                ]


                return value.rstrip(
                    b"\0"
                ).decode(
                    "utf-8",
                    "replace",
                )

    except Exception:
        return ""


    return ""


# ======================================================================
# Platform detection
# ======================================================================

SYSTEM = platform.system()
MACHINE = platform.machine().lower()

PYTHON_EXECUTABLE = os.path.realpath(
    sys.executable
)

ELF_INTERPRETER = get_elf_interpreter(
    PYTHON_EXECUTABLE
)


def detect_os():
    """
    Detect the actual runtime used by Python.

    Returns:

        Android
            Native Android/Bionic Python.

        Linux
            Linux/glibc or unknown Linux runtime.

        Darwin
            macOS.

        Windows
            Windows.
    """

    if SYSTEM == "Darwin":
        return "Darwin"


    if SYSTEM == "Windows":
        return "Windows"


    if SYSTEM != "Linux":
        return SYSTEM


    # --------------------------------------------------------------
    # Native Android/Bionic
    # --------------------------------------------------------------

    if ELF_INTERPRETER in (
        "/system/bin/linker",
        "/system/bin/linker64",
    ):
        return "Android"


    # --------------------------------------------------------------
    # Linux
    #
    # A Linux/glibc loader normally contains ld-linux.
    #
    # We intentionally default unknown Linux to Linux rather than
    # Android. This prevents an accidental Android library from being
    # selected merely because the environment is unusual.
    # --------------------------------------------------------------

    return "Linux"


OS_NAME = detect_os()


# ======================================================================
# Platform information
# ======================================================================

def get_platform_name():
    if OS_NAME == "Android":
        return "Android (Termux)"

    if OS_NAME == "Linux":
        return "Linux"

    if OS_NAME == "Darwin":
        return "macOS"

    if OS_NAME == "Windows":
        return "Windows"

    return OS_NAME


# ======================================================================
# Native library maps
# ======================================================================

# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------

WINDOWS_LIBS = [
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libvosk.dll",
    "libwinpthread-1.dll",
]


# ----------------------------------------------------------------------
# Linux / glibc
# ----------------------------------------------------------------------

LINUX_ARCH_LIB_MAP = {

    "x86_64":
        "libvosk.so",

    "amd64":
        "libvosk.so",

    "aarch64":
        "libvosk_linux_aarch64.so",

    "arm64":
        "libvosk_linux_aarch64.so",

    "armv7l":
        "libvosk_linux_armv7l.so",

    "armv7":
        "libvosk_linux_armv7l.so",

    "armhf":
        "libvosk_linux_armv7l.so",
}


# ----------------------------------------------------------------------
# Android / Bionic
# ----------------------------------------------------------------------

ANDROID_ARCH_LIB_MAP = {

    "aarch64":
        "libvosk_android-arm64-v8a.so",

    "arm64":
        "libvosk_android-arm64-v8a.so",

    "armv7l":
        "libvosk_android-armeabi-v7a.so",

    "armv7":
        "libvosk_android-armeabi-v7a.so",

    "armhf":
        "libvosk_android-armeabi-v7a.so",

    "x86_64":
        "libvosk_android-x86_64.so",

    "amd64":
        "libvosk_android-x86_64.so",

    "x86":
        "libvosk_android-x86.so",

    "i686":
        "libvosk_android-x86.so",

    "i386":
        "libvosk_android-x86.so",
}


# ----------------------------------------------------------------------
# macOS
# ----------------------------------------------------------------------

DARWIN_LIBS = [
    "libvosk.dyld",
]


# ======================================================================
# All native libraries
#
# sdist contains ALL binaries so that the source archive remains
# capable of building wheels for the supported platforms.
# ======================================================================

LINUX_LIBS_ALL = list(
    dict.fromkeys(
        LINUX_ARCH_LIB_MAP.values()
    )
)

ANDROID_LIBS_ALL = list(
    dict.fromkeys(
        ANDROID_ARCH_LIB_MAP.values()
    )
)

ALL_LIBS = (
    WINDOWS_LIBS
    + LINUX_LIBS_ALL
    + ANDROID_LIBS_ALL
    + DARWIN_LIBS
)


# ======================================================================
# Select native libraries
# ======================================================================

def get_lib_files():
    """
    Return native libraries required by the current build.

    sdist:
        ALL native libraries.

    wheel:
        Only the library corresponding to the current runtime and
        architecture.
    """

    # --------------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------------

    if "sdist" in sys.argv:
        return ALL_LIBS


    # --------------------------------------------------------------
    # Android / native Termux
    # --------------------------------------------------------------

    if OS_NAME == "Android":

        lib = ANDROID_ARCH_LIB_MAP.get(MACHINE)

        if lib is None:
            raise NotImplementedError(
                "Unsupported Android/Termux architecture: {}".format(
                    MACHINE
                )
            )

        return [lib]


    # --------------------------------------------------------------
    # Windows
    # --------------------------------------------------------------

    if OS_NAME == "Windows":
        return WINDOWS_LIBS


    # --------------------------------------------------------------
    # macOS
    # --------------------------------------------------------------

    if OS_NAME == "Darwin":
        return DARWIN_LIBS


    # --------------------------------------------------------------
    # Linux / glibc
    # --------------------------------------------------------------

    if OS_NAME == "Linux":

        lib = LINUX_ARCH_LIB_MAP.get(MACHINE)

        if lib is None:
            raise NotImplementedError(
                "Unsupported Linux architecture: {}".format(
                    MACHINE
                )
            )

        return [lib]


    raise NotImplementedError(
        "Platform '{}' is not supported.".format(
            OS_NAME
        )
    )


# ======================================================================
# Binary distribution
# ======================================================================

class BinaryDistribution(Distribution):
    """
    Tell setuptools that this package contains platform-specific
    native libraries.
    """

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


# ======================================================================
# Verify native libraries
# ======================================================================

def check_lib_files():

    package_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pyvosklivesubtitle",
    )


    expected = get_lib_files()

    missing = []


    for filename in expected:

        filepath = os.path.join(
            package_dir,
            filename,
        )

        if not os.path.isfile(filepath):
            missing.append(filepath)


    if missing:

        print()
        print(
            "ERROR: Required native library file(s) not found:"
        )

        for filepath in missing:
            print(
                "  - {}".format(filepath)
            )

        print()
        print(
            "Platform          : {}".format(
                get_platform_name()
            )
        )
        print(
            "System            : {}".format(
                SYSTEM
            )
        )
        print(
            "Architecture      : {}".format(
                MACHINE
            )
        )
        print(
            "Python executable : {}".format(
                PYTHON_EXECUTABLE
            )
        )
        print(
            "ELF interpreter   : {}".format(
                ELF_INTERPRETER or "UNKNOWN"
            )
        )
        print()

        sys.exit(1)


check_lib_files()


# ======================================================================
# Long description
# ======================================================================

long_description = (
    "pyvosklivesubtitle is a PySimpleGUI based DESKTOP APP that can "
    "RECOGNIZE any live streaming in 23 languages that supported by "
    "VOSK then TRANSLATE (using unofficial online Google Translate  "
    "API) and display it as LIVE CAPTION / LIVE SUBTITLE            "
)


# ======================================================================
# Runtime dependencies
# ======================================================================

INSTALL_REQUIRES = [
    "pysimplegui>=4.60.1",
    "sounddevice>=0.4.4",
    "vosk>=0.3.44",
    "requests>=2.27.1",
    "httpx>=0.13.3",
    "streamlink>=5.3.1",
    "urllib3>=1.26.0,<3.0",
    "six>=1.16.0",
    "pysrt>=1.1.2",
    "tqdm>=4.64.0",
    "audioop-lts>=0.2.1; python_version>='3.13'",
]


# ======================================================================
# Setup
# ======================================================================

setup(

    name="pyvosklivesubtitle",

    version=VERSION,

    description=(
        "pyvosklivesubtitle is a PySimpleGUI based DESKTOP APP that can "
        "RECOGNIZE any live streaming in 23 languages that supported by "
        "VOSK then TRANSLATE (using unofficial online Google Translate  "
        "API) and display it as LIVE CAPTION / LIVE SUBTITLE            "
    ),

    long_description=long_description,

    author="Bot Bahlul",

    author_email="bot.bahlul@gmail.com",

    url="https://github.com/botbahlul/pyvosklivesubtitle",

    packages=find_packages(),

    entry_points={
        "console_scripts": [
            "pyvosklivesubtitle=pyvosklivesubtitle:main",
        ],
    },

    install_requires=INSTALL_REQUIRES,

    license=open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "LICENSE",
        ),
        encoding="utf-8",
    ).read(),

    include_package_data=False,

    package_data={
        "pyvosklivesubtitle": get_lib_files(),
    },

    distclass=BinaryDistribution,
)
