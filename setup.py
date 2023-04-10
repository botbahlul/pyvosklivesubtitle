    #!/usr/bin/env python3
from __future__ import unicode_literals

import sys
import os
import stat

#from setuptools import setup
#from setuptools.command.install import install
#from distutils import log

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.version_info < (2, 6):
    print("THIS MODULE REQUIRES PYTHON 2.6, 2.7, OR 3.3+. YOU ARE CURRENTLY USING PYTHON {0}".format(sys.version))
    sys.exit(1)


long_description = (
    'pyvosklivesubtitle is a python based desktop aplication which can recognize any live streaming'
    'in 21 languages that supported by VOSK then translate and display it as LIVE SUBTITLES'
    )

setup(
    name="pyvosklivesubtitle",
    description="A Python based desktop aplication that can RECOGNIZE any live streaming in 21 languages that supported by VOSK then TRANSLATE and display it as LIVE SUBTITLES",
    version="0.1.2",
    include_package_data=True,
    author='Bot Bahlul',
    author_email='bot.bahlul@gmail.com',
    url='https://github.com/botbahlul/pyvosklivesubtitle',
    packages=[str('pyvosklivesubtitle')],
    entry_points={
        'console_scripts': [
            'pyvosklivesubtitle = pyvosklivesubtitle:main',
        ],
    },
    install_requires=[
        'sounddevice>=0.4.4',
        'vosk>=0.3.44',
        'pysimplegui>=4.60.1',
        "httpx>=0.13.3",
        "streamlink>=5.3.1",
        "six>=1.16.0",
        "pysrt>=1.1.2",
    ],
    license=open("LICENSE").read()
)
