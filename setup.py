from __future__ import unicode_literals

import os
import platform
import sys
import warnings
import re

from setuptools import setup, find_packages
from setuptools.dist import Distribution

#from pyvosklivesubtitle import VERSION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(BASE_DIR, "pyvosklivesubtitle", "__init__.py"),
    encoding="utf-8",
) as f:
    VERSION = re.search(
        r'^VERSION\s*=\s*[\'"]([^\'"]+)[\'"]',
        f.read(),
        re.MULTILINE,
    ).group(1)


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
    message=".*is deprecated*",
)


# ----------------------------------------------------------------------
# Python version
# ----------------------------------------------------------------------

if sys.version_info < (3, 10):
    print(
        "THIS MODULE REQUIRES PYTHON 3.10+. "
        "YOU ARE CURRENTLY USING PYTHON {0}".format(sys.version)
    )
    sys.exit(1)


# ----------------------------------------------------------------------
# Platform detection
# ----------------------------------------------------------------------

SYSTEM = platform.system()

if SYSTEM not in ("Windows", "Linux", "Darwin"):
    raise NotImplementedError(
        "Platform '{}' is not supported.".format(SYSTEM)
    )


# ----------------------------------------------------------------------
# Binary distribution
#
# The package contains native Vosk libraries (.so/.dll/.dyld), therefore
# it MUST NOT be built as a pure Python wheel.
# ----------------------------------------------------------------------

class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


# ----------------------------------------------------------------------
# Platform-specific binary files
# ----------------------------------------------------------------------

def get_lib_files():
    if SYSTEM == "Linux":
        return [
            "libvosk.so",
        ]

    if SYSTEM == "Darwin":
        # Vosk Python binding currently loads libvosk.dyld.
        return [
            "libvosk.dyld",
        ]

    if SYSTEM == "Windows":
        return [
            "libvosk.dll",
            "libgcc_s_seh-1.dll",
            "libstdc++-6.dll",
            "libwinpthread-1.dll",
        ]

    raise NotImplementedError(
        "Platform '{}' is not supported.".format(SYSTEM)
    )


# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------

install_requires = [
    "pysimplegui>=4.60.1",
    "sounddevice>=0.4.4",
    "vosk>=0.3.44",
    "requests>=2.27.1",
    "httpx>=0.13.3",
    "streamlink>=5.3.1",
    "urllib3>=1.26.0,<3.0",
    "six>=1.16.0",
    "pysrt>=1.1.2",
    "av==12.2.0",
    "tqdm>=4.64.0",
]


# Windows-only dependency
if SYSTEM == "Windows":
    install_requires.append("pywin32>=306")


# ----------------------------------------------------------------------
# Long description
# ----------------------------------------------------------------------

long_description = (
    "pyvosklivesubtitle is a Python based desktop application "
    "which can recognize live streaming in 21 languages supported "
    "by VOSK, then translate and display it as LIVE SUBTITLES."
)


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

setup(
    name="pyvosklivesubtitle",

    version=VERSION,

    description=(
        "A Python based desktop application that can RECOGNIZE "
        "live streaming in 21 languages supported by VOSK, "
        "then TRANSLATE and display it as LIVE SUBTITLES"
    ),

    long_description=long_description,

    author="Bot Bahlul",
    author_email="bot.bahlul@gmail.com",

    url="https://github.com/botbahlul/pyvosklivesubtitle",

    packages=find_packages(),

    include_package_data=True,

    package_data={
        "pyvosklivesubtitle": get_lib_files(),
    },

    install_requires=install_requires,

    entry_points={
        "console_scripts": [
            "pyvosklivesubtitle=pyvosklivesubtitle:main",
        ],
    },

    license=open(
        os.path.join(os.path.dirname(__file__), "LICENSE"),
        encoding="utf-8",
    ).read(),

    distclass=BinaryDistribution,
)
