# pyvosklivesubtitle <a href="https://pypi.org/project/pyvosklivesubtitle/"><img src="https://img.shields.io/pypi/v/pyvosklivesubtitle.svg"></img></a>

### PySimpleGUI based DESKTOP APP that can RECOGNIZE any LIVE STREAMING in 23 languages that supported by VOSK then TRANSLATE (using unofficial online Google Translate API) and display it as LIVE CAPTION / LIVE SUBTITLE

https://github.com/botbahlul/pyvosklivesubtitle/assets/88623122/d7f4b9ab-d333-475a-a0d6-c70b3f9b8621

The speech recognition part is using vosk python module https://pypi.org/project/vosk/
```
pip install --upgrade vosk
```

On the older versions (bellow 0.0.11) the translation part is using googletrans=4.0.04c1 and pygoogletranslation module https://github.com/Saravananslb/py-googletranslation, but since version 0.0.11 I decide to use my own simple translate function.

The GUI part is using pysimplegui https://github.com/PySimpleGUI/PySimpleGUI and tkinter

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

If you don't have python on your OS you can try compiled version from this release page assets.

If it doesn't run well then you need to install python on your Windows system.

When you run this app for the very first time it may takes some times to download vosk language model, you can check those  downloaded models in "/home/username/.cache/vosk" (if you're on Linux) and "C:\\Users\\Username\\.cache\vosk\\" (if you're on Windows).

You can always download those small models manually from https://alphacephei.com/vosk/models then extract them to that used folder.

You can try to compile that script in win folder (if your OS is Windows) or linux folder (if your OS is Linux) into a single executable file with pyinstaller:
```
pip install pyinstaller
pyinstaller --onefile pyvls.py
```

The executable compiled file will be placed by pyinstaller into dist subfolder of your current working folder, then you can just put that compiled exe file into a folder that has been added to your PATH ENVIRONTMENT so you can execute it from anywhere.

PLEASE NOTE THAT START FROM VERSION 0.1.2 YOU SHOULD USE THAT "mypyinstaller.bat" (or "mypyinstaller.sh" IF YOU'RE ON LINUX) TO COMPILE WITH pyinstaller.


UPDATE NOTES :

Please don't be confused with those android-aarch64 or linux-armv7l etc

They are just same pyvls.pyw scripts setup with different libvosk.so to run on [TERMUX](https://github.com/termux/termux-app/releases)

Those android-aarch64, android-armeabi-v7a, android-x86, and android-x86_64 are pure termux environment (which is running bionic like /system/bin/linker64)

And those linux-aarch64 and linux-armv7l are proot-distro environment installed inside termux (which is running linux glibc like /lib/ld-linux-aarch64.so.1)

If you didn't know yet, we can run a full xfce4 desktop environment on termux with the help of [termux-x11](https://github.com/termux/termux-x11/releases)

```
termux-setup-storage
pkg update && pkg upgrade -y
pkg install x11-repo -y
pkg install termux-x11-nightly termux-api xfce4 xfce4-terminal dbus pulseaudio pavucontrol -y
```

Don't forget to install python, python-pip and python-tkinter because this app use tkinter which can not be installed with pip (tkinter is part of termux python packages)

```
pkg update
pkg install python python-pip python-tkinter -y
```

And also [termux-api](https://github.com/termux/termux-api/releases) app, don't forget to give it microphone permission so vosk recognizer can listen to your audio

You have to install both the apk file and that termux packages for those termux-11 and termux-api

If you want to try to install and run pyvosklivesubtitle in xfce4 you can use these 2 commands

```
pulseaudio --start --exit-idle-time=-1 --load="module-native-protocol-tcp auth-anonymous=1"
termux-x11 :1 -xstartup "dbus-launch --exit-with-session xfce4-session" -dpi 120 &
```

You can save thsese 2 commands into a shell script for example can name it xfce4.sh so you can just type xfce4.sh every time you wan to run xfce4

```
cat << 'EOF' > xfce4.sh
#!/data/data/com.termux/files/usr/bin/bash
pulseaudio --start --exit-idle-time=-1 --load="module-native-protocol-tcp auth-anonymous=1"
termux-x11 :1 -xstartup "dbus-launch --exit-with-session xfce4-session" -dpi 120 &
EOF
chmod +x xfce4.sh
./xfce4.sh
```

Now you can open the termux-11 app you've installed before via the apk file, then open terminal application of xfce4 and then

```
pip install pyvosklivesubtutle
pyvosklivesubtutle
```

If you want to try pyvosklivesubtitle in proot-distro ubuntu environment, open that terminal app and type

```
pkg install proot-disto -y
proot-distro install ubuntu
proot-distro login ubuntu
apt update && apt upgrade -y
apt install python python3-pip python3-tk -y
python3 -m venv ~/venv
source ~/venv/bin/activate
python3 -m pip install --upgrade pyvosklivesubtitle
export DISPLAY=:1 # THIS IS MANDATORY TO RUN ANY PROOT-DISTRO UBUNTU GUI APPS IN XFCE4 ENVIRONMENT!
pyvosklivesubtitle
```

For audio routing in xfce4 you can use this vmic.sh script :

```
cat << 'EOF' > vmic.sh
#!/data/data/com.termux/files/usr/bin/bash

set -e

command -v pactl >/dev/null 2>&1 || {
    echo "ERROR: pactl not found, please install pulseaudio-utils package with sudo apt install pulseaudio-utils"
    exit 1
}

pactl info >/dev/null || {
    echo "ERROR: PipeWire/PulseAudio server unavailable"
    exit 1
}

echo "Resetting old audio routing if exist..."

DEFAULT_SINK=$(pactl list short sinks | awk '{print $2}')

MODULE_IDS=$(pactl list short modules | awk -v ds="$DEFAULT_SINK" '
    ($2 == "module-loopback" || $2 == "module-remap-source" || $2 == "module-null-sink") && \
    ($0 ~ "virtual_sink" || $0 ~ "virtual_mic" || $0 ~ ds) {
        print $1
    }
')


if [ -n "$MODULE_IDS" ]; then
    for id in $MODULE_IDS; do
        pactl unload-module "$id" 2>/dev/null || true
    done
fi


DEFAULT_SINK=$(pactl list short sinks | awk '{print $2}')
DEFAULT_SINK_MONITOR="${DEFAULT_SINK}.monitor"
VIRTUAL_SINK="virtual_sink"
VIRTUAL_MIC="virtual_mic"

echo "Creating $VIRTUAL_SINK..."

if ! pactl list short sinks | awk '{print $2}' | grep -Fxq "$VIRTUAL_SINK"; then
    pactl load-module module-null-sink \
        sink_name="$VIRTUAL_SINK" \
        sink_properties=device.description=VirtualSink >/dev/null
fi

echo "Creating $VIRTUAL_MIC from $DEFAULT_SINK_MONITOR..."

if ! pactl list short sources | awk '{print $2}' | grep -Fxq "$VIRTUAL_MIC"; then
    pactl load-module module-remap-source \
        "master=$DEFAULT_SINK_MONITOR" \
        source_name="$VIRTUAL_MIC" \
        source_properties=device.description=VirtualMic >/dev/null
fi

echo "Creating looopback $DEFAULT_SINK_MONITOR to $VIRTUAL_SINK"

LOOPBACK_ID=$(
    pactl list short modules |
    awk -v source="$VIRTUAL_MIC" -v sink="$VIRTUAL_SINK" '
        $2 == "module-loopback" &&
        index($0, "source=" source) &&
        index($0, "sink=" sink) {
            print $1
            exit
        }
    '
)

if [[ -z "$LOOPBACK_ID" ]]; then
    pactl load-module module-loopback \
        "source=$DEFAULT_SINK_MONITOR" \
        "sink=$VIRTUAL_SINK" >/dev/null
fi

echo "Set $VIRTUAL_MIC as default source..."
pactl set-default-source virtual_mic

echo
echo "Audio routing finished"
echo "Default playback : $DEFAULT_SINK"
echo "VirtualSink      : $VIRTUAL_SINK"
echo "VirtualMic       : $VIRTUAL_MIC"
echo "Default input    : $VIRTUAL_MIC"
echo "Loopback         : $DEFAULT_SINK_MONITOR -> $VIRTUAL_SINK"
EOF
chmod +x vmic.sh
./vmic.sh
```

If you're using older android like those Phoenix OS variants you can not use termux-11, instead you should use [bVNC Free](https://play.google.com/store/apps/details?id=com.iiordanov.freebVNC&hl=id) to run xfce4

```
pkg i -y tigervnc xorg-xhost 
xhost +local:
cat << 'EOF' > vnc-xfce4.sh
#!/data/data/com.termux/files/usr/bin/bash
vncserver -kill :1 >/dev/null 2>&1
rm -rf "$PREFIX/tmp/.X11-unix/X1" "$PREFIX/tmp/.X1-lock" /tmp/.X11-unix/X1 /tmp/.X1-lock 2>/dev/null
export HOSTNAME=localhost
touch "$HOME/.Xauthority"
vncserver -localhost -geometry 1024x600 -depth 16 :1 2>&1 | grep -v 'bad display name'
EOF
chmod +x vnc-xfce4.sh
./vnc-xfce4.sh
```

After execute that ./vnc-xfce4.sh script you can open bVNC app and setup your VNC settings from there.


START FROM VERSION 0.1.2 THIS APP USE streamlink MODULE WHICH SUPPORTS urllib3 >= 1.26, < 3.x BRANCH, SO IF YOU'RE STILL WANT TO INSTALL THIS APP TRY TO REINSTALL streamlink WITH --force-reinstall ARGUMENT.

```
pip install streamlink --force-reinstall
```

THAT "streamlink" MODULE FOLDER MUST BE ON SAME FOLDER WITH "pyvls.pyw" and "mypyinstaller.bat", OR YOU CAN CHANGE THE LOCATION FOLDER BY EDITING "mypyinstaller.bat".

DONT FORGET TO PLACE THOSE LIBS FILES FROM VOSK (libgcc_s_seh-1.dll, libstdc++-6.dll, libvosk.dll, & libwinpthread-1.dll if your OS is Windows, libvosk.so if your OS is Linux) INTO SAME FOLDER WITH COMPILED EXE FILE!

ALSO NOTE THAT SINCE VERSION 0.1.2 THIS APP NEED ffmpeg AND ffprobe, you can get it from https://www.ffmpeg.org/

IN DEBIAN BASED LINUX OS YOU CAN INSTALL IT WITH :

```
sudo apt update
sudo apt install -y ffmpeg
```

I was succesfuly compiled this script in Windows 10 with pyinstaller-5.1 and Pyhton-3.10.4, and python-3.10.4 in Debian 10

Another alternative way you can install this script with python by cloning this git (or downloading this git as zip then extract it into 
a folder), and then just type :

```
python setup.py build
python setup.py install
```

If you fail when installing any python modules using pip, you can always use this way, get that module from github then build them manually


For best recognizing performance, on windows you will need STEREO MIX or VIRTUAL AUDIO CABLE as RECORDING/INPUT DEVICE 
![image](https://github.com/botbahlul/pyvosklivesubtitle/assets/88623122/e08b4b85-45cb-477b-a7f8-297d94791839)
![image](https://user-images.githubusercontent.com/88623122/199527559-e2609d8c-3479-420d-8c52-806fa56a21f4.png)
![image](https://user-images.githubusercontent.com/88623122/199528286-1ab77dc4-38a9-41f2-9b92-25db352a1ed2.png)

LAUNCH Audio Repeater MME, NOT Audio Repeater KS! AND THEN CLICK THAT START BUTTON AND MINIMIZE IT, DON'T CLOSE IT!

![image](https://github.com/botbahlul/pyvosklivesubtitle/assets/88623122/6b09c622-7a76-42af-a776-738468466025)


On linux you willl need PAVUCONTROL (by choosing MONITOR of your audio device as INPUT DEVICE)

![image](https://user-images.githubusercontent.com/88623122/231320153-e89bb21e-916f-4e82-a520-43e3112f6801.png)
![image](https://user-images.githubusercontent.com/88623122/230965838-40764e2a-8ce9-4a03-9f8b-901c9eae43ef.png)

Or simply just use that [route_audio.sh](https://github.com/botbahlul/pyvosklivesubtitle/blob/main/route_audio.sh) script


On macOS you will need [blackhole](https://github.com/existentialaudio/blackhole) and [switchaudio-osx](https://github.com/deweller/switchaudio-osx)


To DRAG/MOVE that TRANSLATION TEXT anywhere on the screen, just USE CONTROL KEY + LEFT MOUSE combination.


ANOTHER NOTES, DON'T FORGET TO CHANGE YOUR CONSOLE (COMMAND PROMPT WINDOW) CODE PAGE TO 65001 (UTF-8)!

```
To change the codepage for the console, do the following:

Start -> Run -> regedit
Go to [HKEY_LOCAL_MACHINE\Software\Microsoft\Command Processor\Autorun]
Change the value to @chcp 65001>nul
If Autorun is not present, you can add a New String
```
![image](https://github.com/botbahlul/pyvosklivesubtitle/assets/88623122/f0a97679-d01a-4e55-b5cd-90530c4026cd)


### Usage

```
usage: pyvosklivesubtitle [-h] [-B] [-S SRC_LANGUAGE] [-D DST_LANGUAGE] [-lls] [-lld] [-sf SUBTITLE_FILENAME] [-F SUBTITLE_FORMAT]
                          [-lsf] [-ld] [-af AUDIO_FILENAME] [-d DEVICE] [-r SAMPLERATE] [-u URL] [-vf VIDEO_FILENAME] [-v]

options:
  -h, --help            show this help message and exit
  -B, --big-models      Use big models if exist
  -S SRC_LANGUAGE, --src-language SRC_LANGUAGE
                        Spoken language
  -D DST_LANGUAGE, --dst-language DST_LANGUAGE
                        Desired language for translation
  -lls, --list-src-languages
                        List all available source languages
  -lld, --list-dst-languages
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

Check my other SPEECH RECOGNITIION + TRANSLATE PROJECTS https://github.com/botbahlul?tab=repositories

Buy me coffee : https://sociabuzz.com/botbahlul/tribe
