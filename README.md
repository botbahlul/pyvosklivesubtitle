# pyvosklivesubtitle <a href="https://pypi.org/project/pyvosklivesubtitle/0.0.4/"><img src="https://img.shields.io/pypi/v/pyvosklivesubtitle.svg"></img></a>

### A Python based desktop aplication that can RECOGNIZE any live streaming in 21 languages that supported by VOSK then TRANSLATE and display it as LIVE SUBTITLES

The speech recognition part is using vosk https://alphacephei.com/vosk/

The translation part is using googletrans==4.0.0-rc1

The GUI part is using pysimplegui https://github.com/PySimpleGUI/PySimpleGUI

### Installation

If you don't have python on your OS you can get compiled version from this release page assets.

If python has already installed on your OS you can install this script with pip :

```
pip install pyvosklivesubtitle
```

then you can run it by just type :
```
pyvosklivesubtitle
```

When you run this app for the very first time it may takes some times to download vosk language model, you can check those  downloaded models in "/usr/share/vosk" (if you're on Linux) and "C:\Username\\.cache\vosk\\" or "C:\Username\AppData\Local\vosk\\" (if you're on Windows).

You can always download those small models manually from https://alphacephei.com/vosk/models then extract them to that used folder.

You can try to compile that script in win folder (if your OS is Windows) or linux folder (if your OS is Linux) into a single executable file with pyinstaller :
```
pip install pyinstaller
pyinstaller --onefile pyvls.py
```
The executable compiled file will be placed by pyinstaller into dist subfolder of your current working folder, then you can just put that compiled exe file into a folder that has been added to your PATH ENVIRONTMENT so you can execute it from anywhere.

DONT FORGET TO PLACE THOSE LIBS FILES FROM VOSK (libgcc_s_seh-1.dll, libstdc++-6.dll, libvosk.dll, & libwinpthread-1.dll if your OS is Windows, libvosk.so if your OS is Linux) into same folder with compiled exe file

I was succesfuly compiled it in Windows 10 with pyinstaller-5.1 and Pyhton-3.10.4, and python-3.8.12 in Debian 9

Another alternative way you can install this script with python by cloning this git (or downloading this git as zip then extract it into 
a folder), and then just type :

```
python setup.py build
python setup.py install
```

If you fail when installing any python modules using pip, you can always use this way, get that module from github then build them manually

For best recognizing performance, on windows you will need STEREO MIX or VIRTUAL AUDIO CABLE as RECORDING/INPUT DEVICE 
![image](https://user-images.githubusercontent.com/88623122/199527559-e2609d8c-3479-420d-8c52-806fa56a21f4.png)
![image](https://user-images.githubusercontent.com/88623122/199528286-1ab77dc4-38a9-41f2-9b92-25db352a1ed2.png)
![image](https://user-images.githubusercontent.com/88623122/199528861-22541706-3bdf-427c-8c2f-44174b114e34.png)

and on linux you willl need PAVUCONTROL (by choosing MONITOR of your audio device as INPUT DEVICE)
![image](https://user-images.githubusercontent.com/88623122/199517907-76d61acb-3f07-49b6-8f2f-4b6a2b787eff.png)

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
