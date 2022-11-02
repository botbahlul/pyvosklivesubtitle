# pyvosklivesubtitle <a href="https://pypi.org/project/pyvosklivesubtitle/0.0.1/"><img src="https://img.shields.io/pypi/v/pyvosklivesubtitle.svg"></img></a>

### A Python based desktop aplication that can RECOGNIZE any live streaming in 21 languages that supported by VOSK then TRANSLATE and display it as LIVE SUBTITLES

The speech recognition part is using vosk

The translation part is using googletrans==4.0.0-rc1

The GUI part is using pysimplegui https://github.com/PySimpleGUI/PySimpleGUI

### Installation

If you don't have python on your system you can get compiled version from this release page assets.

If python has already installed on your system you can install this script with pip

```
pip install pyvosklivesubtitle
```

then you can run it by just type 
```
pyvosklivesubtitle
```

When you run this app for the very first time it may takes some times to download vosk language model, you can check  downloaded models in "C:\Username\\.cache\vosk\\" or "C:\Username\AppData\Local\vosk\\"

You can always downloaad those small models manually from https://alphacephei.com/vosk/models then extract it to that used folder

if you're windows user you can compile script in win folder into a single executable file with pyinstaller :

```
pip install pyinstaller
pyinstaller --onefile pyvls.py
```

The executable compiled file will be placed by pyinstaller into dist subfolder of your current working folder, then you can just
rename and put that compiled exe file into a folder that has been added to your PATH ENVIRONTMENT so you can execute it from anywhere.

DONT FORGET TO PLACE THOSE 4 LIBS FILES FROM VOSK (libgcc_s_seh-1.dll, libstdc++-6.dll, libvosk.dll, & libwinpthread-1.dll) into
same folder with compiled exe file

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
pyvosklivesubtitle [-S SRC_LANGUAGE] [-D DST_LANGUAGE]

for example : pyvosklivesubtitle -S ja -D en

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

### License

MIT
