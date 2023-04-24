# pyvosklivesubtitle <a href="https://pypi.org/project/pyvosklivesubtitle/"><img src="https://img.shields.io/pypi/v/pyvosklivesubtitle.svg"></img></a>

### PySimpleGUI based DESKTOP APP that can RECOGNIZE any live streaming in 21 languages that supported by VOSK then TRANSLATE (using unofficial online Google Translate API) and display it as LIVE CAPTION / LIVE SUBTITLE



https://user-images.githubusercontent.com/88623122/221994235-0ee3cfa5-26f7-4e53-881d-cdef06fe3f60.mp4



The speech recognition part is using vosk python module https://pypi.org/project/vosk/
```
pip install --upgrade vosk
```

On the older versions (bellow 0.0.11) the translation part is using googletrans=4.0.04c1 and pygoogletranslation module https://github.com/Saravananslb/py-googletranslation, but since version 0.0.11 I decide to use my own simple translate function.

The GUI part is using pysimplegui https://github.com/PySimpleGUI/PySimpleGUI
```
pip install --upgrade pysimplegui
```

### Installation
```
pip install --upgrade pyvosklivesubtitle
```

then you can run it by just type :
```
pyvosklivesubtitle
```

If you don't have python on your OS you can get compiled version from this release page assets.


When you run this app for the very first time it may takes some times to download vosk language model, you can check those  downloaded models in "/home/username/.cache/vosk" (if you're on Linux) and "C:\\Users\\Username\\.cache\vosk\\" (if you're on Windows).

You can always download those small models manually from https://alphacephei.com/vosk/models then extract them to that used folder.
Pay attension to those folders name, because their names should contain the language codes in \"\_\_init\_\_.py\", esspecially for Chinese language, which in \"\_\_init\_\_.py\" its code is \"zh\", so you should rename that extracted downloaded model to \"vosk-model-small-zh-0.22\". This is needed for GoogleTranslate funcion to work properly.
![image](https://user-images.githubusercontent.com/88623122/234000963-c2ab4c69-70fd-4374-9a1a-0cc1316791e8.png)
![image](https://user-images.githubusercontent.com/88623122/234001411-f06821c9-0b68-4414-b3c3-a7280da4d560.png)

You can try to compile that script in win folder (if your OS is Windows) or linux folder (if your OS is Linux) into a single executable file with pyinstaller:
```
pip install pyinstaller
pyinstaller --onefile pyvls.py
```
The executable compiled file will be placed by pyinstaller into dist subfolder of your current working folder, then you can just put that compiled exe file into a folder that has been added to your PATH ENVIRONTMENT so you can execute it from anywhere.

NOTES :
START FROM VERSION 0.1.2 YOU SHOULD USE THAT "mypyinstaller.bat" (or "mypyinstaller.sh" IF YOU'RE ON LINUX) TO COMPILE WITH pyinstaller.

THAT "streamlink" MODULE FOLDER MUST BE ON SAME FOLDER WITH "pyvls.pyw" and "mypyinstaller.bat", OR YOU CAN CHANGE THE LOCATION FOLDER BY EDITING "mypyinstaller.bat".

DONT FORGET TO PLACE THOSE LIBS FILES FROM VOSK (libgcc_s_seh-1.dll, libstdc++-6.dll, libvosk.dll, & libwinpthread-1.dll if your OS is Windows, libvosk.so if your OS is Linux) INTO SAME FOLDER WITH COMPILED EXE FILE!

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
![image](https://user-images.githubusercontent.com/88623122/231320153-e89bb21e-916f-4e82-a520-43e3112f6801.png)
![image](https://user-images.githubusercontent.com/88623122/230965838-40764e2a-8ce9-4a03-9f8b-901c9eae43ef.png)

To DRAG/MOVE that TRANSLATION TEXT anywhere on the screen, just USE CONTROL KEY + LEFT MOUSE combination.

### Usage

```
usage: pyvosklivesubtitle [-h] [-S SRC_LANGUAGE] [-D DST_LANGUAGE] [-lls] [-lld] [-sf SUBTITLE_FILENAME] [-F SUBTITLE_FORMAT] [-lsf]
                          [-ld] [-af AUDIO_FILENAME] [-d DEVICE] [-r SAMPLERATE] [-u URL] [-vf VIDEO_FILENAME] [-v]

options:
  -h, --help            show this help message and exit
  -S SRC_LANGUAGE, --src-language SRC_LANGUAGE
                        Spoken language
  -D DST_LANGUAGE, --dst-language DST_LANGUAGE
                        Desired language for translation
  -lls, --list-languages-src
                        List all available source languages
  -lld, --list-languages-dst
                        List all available destination languages
  -sf SUBTITLE_FILENAME, --subtitle-filename SUBTITLE_FILENAME
                        Subtitle file name for saved transcriptions
  -F SUBTITLE_FORMAT, --subtitle-format SUBTITLE_FORMAT
                        Desired subtitle format for saved transcriptions (default is "srt")
  -lsf, --list-subtitle-formats
                        List all available subtitle formats
  -ld, --list-devices   show list of audio devices and exit
  -af AUDIO_FILENAME, --audio-filename AUDIO_FILENAME
                        audio file to store recording to
  -d DEVICE, --device DEVICE
                        input device (numeric ID or substring)
  -r SAMPLERATE, --samplerate SAMPLERATE
                        sampling rate in Hertz for example 8000, 16000, 44100, or 48000
  -u URL, --url URL     URL of live streaming if you want to record the streaming
  -vf VIDEO_FILENAME, --video-filename VIDEO_FILENAME
                        video file to store recording to
  -v, --version         show program's version number and exit
```

Those command line switches are just to make this application directly select the language in combo box, so it's okay if you don't use them

### License

MIT

Check my other SPEECH RECOGNITIION + TRANSLATE PROJECTS in https://botbahlul.github.io
