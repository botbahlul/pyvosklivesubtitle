# vosklivesubtitle <a href="https://pypi.org/project/vosklivesubtitle/0.0.1/"><img src="https://img.shields.io/pypi/v/vosklivesubtitle.svg"></img></a>

### A Python based desktop aplication similar to Google Translate with an extra feature LIVE SUBTITLES

the translation part is using googletrans==4.0.0-rc1

the GUI part developed with pysimplegui https://github.com/PySimpleGUI/PySimpleGUI

### Installation

If you don't have python on your system you can get compiled version from this release page assets.

if python has already installed on your system you can install this script with pip

```
pip install vosklivesubtitle
```

you can compile this script into a single executable file with pyinstaller by renaming
```__init__.py``` file to ```vosklivesubtitle.pyw``` and type :

```
pip install pyinstaller
pyinstaller --onefile vosklivesubtitle.pyw
```

The executable compiled file will be placed by pyinstaller into dist subfolder of your current working folder, then you can just
rename and put that compiled file into a folder that has been added to your PATH ENVIRONTMENT so you can execute it from anywhere.

I was succesfuly compiled it in Windows 10 with pyinstaller-5.1 and Pyhton-3.10.4, and python-3.8.12 in Debian 9

Another alternative way you can install this script with python by cloning this git (or downloading this git as zip then extract it into 
a folder), and then just type :

```
python setup.py build
python setup.py install
```
``  

If you failed when installing any python modules you can always use this way, get that module from github then build them manually


### Usage

```
vosklivesubtitle [-S SRC_LANGUAGE] [-D DST_LANGUAGE]

for example : vosklivesubtitle -S ja -D en

options:
  -h, --help            show this help message and exit
  -S SRC_LANGUAGE, --src-language SRC_LANGUAGE
                        Spoken language
  -D DST_LANGUAGE, --dst-language DST_LANGUAGE
                        Desired language for translation
  -ll, --list-languages
                        List all available source/destination languages
  -ld, --list-devices   show list of audio devices and exit
  -f FILENAME, --filename FILENAME
                        audio file to store recording to
  -d DEVICE, --device DEVICE
                        input device (numeric ID or substring)
  -r SAMPLERATE, --samplerate SAMPLERATE
                        sampling rate in Hertz for example 8000, 16000, 44100, or 48000
```

Those command line switches are just to make this application directly select the language in combo box, so it's okay if you don't use them

The language codes are just as same as language codes in googletrans


### License

MIT
