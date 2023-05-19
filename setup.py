from __future__ import unicode_literals
import sys
import platform
import os
import stat
from pyvosklivesubtitle import VERSION

try:
    from setuptools import setup, find_packages
    from setuptools.dist import Distribution
except ImportError:
    from distutils.core import setup
    from distutils.dist import Distribut

if sys.version_info <= (3, 8):
    print("THIS MODULE REQUIRES PYTHON 3.8+. YOU ARE CURRENTLY USING PYTHON {0}".format(sys.version))
    sys.exit(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

    def get_ext_modules(self):
        if platform.system() == 'Windows':
            return []
        else:
            return super().get_ext_modules()


def get_lib_files():
    if platform.system() == 'Linux':
        return ['libvosk.so']
    elif platform.system() == 'Darwin':
        return ['libvosk.dyld']
    elif platform.system() == 'Windows':
        return ['libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libvosk.dll', 'libwinpthread-1.dll']
    else:
        raise NotImplementedError(f"Platform '{platform.system()}' is not supported.")


long_description = (
    'pyvosklivesubtitle is a python based desktop aplication which can recognize any live streaming'
    'in 21 languages that supported by VOSK then translate and display it as LIVE SUBTITLES'
)

install_requires = [
    "sounddevice>=0.4.4",
    "vosk>=0.3.44",
    "pysimplegui>=4.60.1",
    "httpx>=0.13.3",
    "streamlink>=5.3.1",
    "six>=1.16.0",
    "pysrt>=1.1.2",
    "requests>=2.27.1",
    "tqdm>=4.64.0",
]

if platform.system() == "Windows":
    install_requires.append("pywin32>=306")

setup(
    name="pyvosklivesubtitle",
    description="A Python based desktop aplication that can RECOGNIZE any live streaming in 21 languages that supported by VOSK then TRANSLATE and display it as LIVE SUBTITLES",
    version=VERSION,
    author='Bot Bahlul',
    author_email='bot.bahlul@gmail.com',
    url='https://github.com/botbahlul/pyvosklivesubtitle',
    packages=[str('pyvosklivesubtitle')],
    entry_points={
        'console_scripts': [
            'pyvosklivesubtitle = pyvosklivesubtitle:main',
        ],
    },
    install_requires=install_requires,
    license=open("LICENSE").read(),
    include_package_data=True,
    package_data={'': get_lib_files()},
    distclass=BinaryDistribution,
)
