#!/bin/sh

# ----------------------------------------------------------------------
# Find Python
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Check Python version
# ----------------------------------------------------------------------
echo
echo "Using Python:"
"$PYTHON_CMD" --version

"$PYTHON_CMD" -c '
import sys
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10 or newer is required.")
    sys.exit(1)
' || exit 1

# ----------------------------------------------------------------------
# Detect operating system
# ----------------------------------------------------------------------
OS_NAME="$(uname -s)"
CPU_ARCH="$(uname -m)"

# Normalize Windows uname output (Git Bash / MSYS2 / Cygwin) to "Windows"
case "$OS_NAME" in
    MINGW*|MSYS*|CYGWIN*) OS_NAME="Windows" ;;
esac

# Special detection for native Termux.
# /data/data/com.termux is also visible inside proot-distro,
# so its existence alone must NOT be used to identify Termux.
if [ "$OS_NAME" = "Linux" ]; then
    PYTHON_PREFIX="$("$PYTHON_CMD" -c 'import sys; print(sys.prefix)' 2>/dev/null)"

    case "$PYTHON_PREFIX" in
        /data/data/com.termux/files/usr*)
            OS_NAME="Android"
            ;;
    esac
fi

echo
echo "Operating system : $OS_NAME"
echo "Architecture     : $CPU_ARCH"

# ----------------------------------------------------------------------
# PyInstaller uses ':' as the src/dest separator on POSIX, ';' on Windows
# ----------------------------------------------------------------------
if [ "$OS_NAME" = "Windows" ]; then
    PYI_SEP=";"
else
    PYI_SEP=":"
fi

# ----------------------------------------------------------------------
# Clean previous build
# ----------------------------------------------------------------------
echo
echo "Cleaning previous build files..."

folder1="./build"
folder2="./dist"
file1="./pyvosklivesubtitle.spec"

if [ -d "$folder1" ]; then
    rm -rf "$folder1"
fi

if [ -d "$folder2" ]; then
    rm -rf "$folder2"
fi

if [ -f "$file1" ]; then
    rm -f "$file1"
fi

# ----------------------------------------------------------------------
# Native library selection per platform / architecture
#
# POSIX sh has no arrays/dicts, so each branch below fills LIBVOSK_FILES
# with a space-separated list of filenames (none of these contain
# spaces, so word-splitting on it later is safe).
# ----------------------------------------------------------------------
case "$OS_NAME" in

    Windows)
        echo
        echo "Detected Windows ($CPU_ARCH)."
        LIBVOSK_FILES="libgcc_s_seh-1.dll libstdc++-6.dll libvosk.dll libwinpthread-1.dll"
        ;;

    Darwin)
        echo
        echo "Detected macOS."
        case "$CPU_ARCH" in
            x86_64|arm64)
                echo "Building macOS $CPU_ARCH executable binary..."
                LIBVOSK_FILES="libvosk.dyld"
                ;;
            *)
                echo "ERROR: Unsupported macOS architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    Linux)
        echo
        echo "Detected Linux ($CPU_ARCH)."

        case "$CPU_ARCH" in
            x86_64|amd64)
                LIBVOSK_FILES="libvosk.so"
                ;;
            aarch64|arm64)
                LIBVOSK_FILES="libvosk_linux_aarch64.so"
                ;;
            armv7l|armv7|armhf)
                LIBVOSK_FILES="libvosk_linux_armv7l.so"
                ;;
            *)
                echo "ERROR: Unsupported Linux architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    Android)
        echo
        echo "Detected Termux Android ($CPU_ARCH)."

        case "$CPU_ARCH" in
            aarch64|arm64)
                LIBVOSK_FILES="libvosk_android-arm64-v8a.so"
                ;;
            armv7l|armv7|armhf)
                LIBVOSK_FILES="libvosk_android-armeabi-v7a.so"
                ;;
            x86_64|amd64)
                LIBVOSK_FILES="libvosk_android-x86_64.so"
                ;;
            x86|i686|i386)
                LIBVOSK_FILES="libvosk_android-x86.so"
                ;;
            *)
                echo "ERROR: Unsupported Android architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    *)
        echo
        echo "ERROR: Unsupported operating system: $OS_NAME"
        exit 1
        ;;
esac


# Make sure every listed native library actually exists before we build
for f in $LIBVOSK_FILES; do
    if [ ! -f "./$f" ]; then
        echo "ERROR: expected native library not found: ./$f"
        exit 1
    fi
done

# ----------------------------------------------------------------------
# Build the PyInstaller argument list
#
# We use "set --" to build the option list because POSIX sh has no
# arrays; this is the standard, quoting-safe way to accumulate
# variable-length argument lists in sh.
# ----------------------------------------------------------------------
set --

for f in $LIBVOSK_FILES; do
    set -- "$@" --add-data "./${f}${PYI_SEP}."
done

set -- "$@" \
--add-data "./streamlink/plugins/*:streamlink/plugins" \
--add-data "./streamlink/*:streamlink" \
--hidden-import streamlink.plugins.abematv \
--hidden-import streamlink.plugins.adultswim \
--hidden-import streamlink.plugins.afreeca \
--hidden-import streamlink.plugins.albavision \
--hidden-import streamlink.plugins.aloula \
--hidden-import streamlink.plugins.app17 \
--hidden-import streamlink.plugins.ard_live \
--hidden-import streamlink.plugins.ard_mediathek \
--hidden-import streamlink.plugins.artetv \
--hidden-import streamlink.plugins.atpchallenger \
--hidden-import streamlink.plugins.atresplayer \
--hidden-import streamlink.plugins.bbciplayer \
--hidden-import streamlink.plugins.bfmtv \
--hidden-import streamlink.plugins.bigo \
--hidden-import streamlink.plugins.bilibili \
--hidden-import streamlink.plugins.blazetv \
--hidden-import streamlink.plugins.bloomberg \
--hidden-import streamlink.plugins.booyah \
--hidden-import streamlink.plugins.brightcove \
--hidden-import streamlink.plugins.btv \
--hidden-import streamlink.plugins.cbsnews \
--hidden-import streamlink.plugins.cdnbg \
--hidden-import streamlink.plugins.ceskatelevize \
--hidden-import streamlink.plugins.cinergroup \
--hidden-import streamlink.plugins.clubbingtv \
--hidden-import streamlink.plugins.cmmedia \
--hidden-import streamlink.plugins.cnews \
--hidden-import streamlink.plugins.crunchyroll \
--hidden-import streamlink.plugins.dailymotion \
--hidden-import streamlink.plugins.dash \
--hidden-import streamlink.plugins.delfi \
--hidden-import streamlink.plugins.deutschewelle \
--hidden-import streamlink.plugins.dlive \
--hidden-import streamlink.plugins.dogan \
--hidden-import streamlink.plugins.dogus \
--hidden-import streamlink.plugins.drdk \
--hidden-import streamlink.plugins.earthcam \
--hidden-import streamlink.plugins.euronews \
--hidden-import streamlink.plugins.facebook \
--hidden-import streamlink.plugins.filmon \
--hidden-import streamlink.plugins.foxtr \
--hidden-import streamlink.plugins.galatasaraytv \
--hidden-import streamlink.plugins.goltelevision \
--hidden-import streamlink.plugins.goodgame \
--hidden-import streamlink.plugins.googledrive \
--hidden-import streamlink.plugins.gulli \
--hidden-import streamlink.plugins.hiplayer \
--hidden-import streamlink.plugins.hls \
--hidden-import streamlink.plugins.http \
--hidden-import streamlink.plugins.htv \
--hidden-import streamlink.plugins.huajiao \
--hidden-import streamlink.plugins.huya \
--hidden-import streamlink.plugins.idf1 \
--hidden-import streamlink.plugins.invintus \
--hidden-import streamlink.plugins.kugou \
--hidden-import streamlink.plugins.linelive \
--hidden-import streamlink.plugins.livestream \
--hidden-import streamlink.plugins.lnk \
--hidden-import streamlink.plugins.lrt \
--hidden-import streamlink.plugins.ltv_lsm_lv \
--hidden-import streamlink.plugins.mdstrm \
--hidden-import streamlink.plugins.mediaklikk \
--hidden-import streamlink.plugins.mediavitrina \
--hidden-import streamlink.plugins.mildom \
--hidden-import streamlink.plugins.mitele \
--hidden-import streamlink.plugins.mixcloud \
--hidden-import streamlink.plugins.mjunoon \
--hidden-import streamlink.plugins.mrtmk \
--hidden-import streamlink.plugins.n13tv \
--hidden-import streamlink.plugins.nbcnews \
--hidden-import streamlink.plugins.nhkworld \
--hidden-import streamlink.plugins.nicolive \
--hidden-import streamlink.plugins.nimotv \
--hidden-import streamlink.plugins.nos \
--hidden-import streamlink.plugins.nownews \
--hidden-import streamlink.plugins.nrk \
--hidden-import streamlink.plugins.ntv \
--hidden-import streamlink.plugins.okru \
--hidden-import streamlink.plugins.olympicchannel \
--hidden-import streamlink.plugins.oneplusone \
--hidden-import streamlink.plugins.onetv \
--hidden-import streamlink.plugins.openrectv \
--hidden-import streamlink.plugins.pandalive \
--hidden-import streamlink.plugins.picarto \
--hidden-import streamlink.plugins.piczel \
--hidden-import streamlink.plugins.pixiv \
--hidden-import streamlink.plugins.pluto \
--hidden-import streamlink.plugins.pluzz \
--hidden-import streamlink.plugins.qq \
--hidden-import streamlink.plugins.radiko \
--hidden-import streamlink.plugins.radionet \
--hidden-import streamlink.plugins.raiplay \
--hidden-import streamlink.plugins.reuters \
--hidden-import streamlink.plugins.rtbf \
--hidden-import streamlink.plugins.rtpa \
--hidden-import streamlink.plugins.rtpplay \
--hidden-import streamlink.plugins.rtve \
--hidden-import streamlink.plugins.rtvs \
--hidden-import streamlink.plugins.ruv \
--hidden-import streamlink.plugins.sbscokr \
--hidden-import streamlink.plugins.showroom \
--hidden-import streamlink.plugins.sportal \
--hidden-import streamlink.plugins.sportschau \
--hidden-import streamlink.plugins.ssh101 \
--hidden-import streamlink.plugins.stadium \
--hidden-import streamlink.plugins.steam \
--hidden-import streamlink.plugins.streamable \
--hidden-import streamlink.plugins.streann \
--hidden-import streamlink.plugins.stv \
--hidden-import streamlink.plugins.svtplay \
--hidden-import streamlink.plugins.swisstxt \
--hidden-import streamlink.plugins.telefe \
--hidden-import streamlink.plugins.tf1 \
--hidden-import streamlink.plugins.trovo \
--hidden-import streamlink.plugins.turkuvaz \
--hidden-import streamlink.plugins.tv360 \
--hidden-import streamlink.plugins.tv3cat \
--hidden-import streamlink.plugins.tv4play \
--hidden-import streamlink.plugins.tv5monde \
--hidden-import streamlink.plugins.tv8 \
--hidden-import streamlink.plugins.tv999 \
--hidden-import streamlink.plugins.tvibo \
--hidden-import streamlink.plugins.tviplayer \
--hidden-import streamlink.plugins.tvp \
--hidden-import streamlink.plugins.tvrby \
--hidden-import streamlink.plugins.tvrplus \
--hidden-import streamlink.plugins.tvtoya \
--hidden-import streamlink.plugins.twitcasting \
--hidden-import streamlink.plugins.twitch \
--hidden-import streamlink.plugins.useetv \
--hidden-import streamlink.plugins.ustreamtv \
--hidden-import streamlink.plugins.ustvnow \
--hidden-import streamlink.plugins.vidio \
--hidden-import streamlink.plugins.vimeo \
--hidden-import streamlink.plugins.vinhlongtv \
--hidden-import streamlink.plugins.vkplay \
--hidden-import streamlink.plugins.vk \
--hidden-import streamlink.plugins.vlive \
--hidden-import streamlink.plugins.vtvgo \
--hidden-import streamlink.plugins.wasd \
--hidden-import streamlink.plugins.webtv \
--hidden-import streamlink.plugins.welt \
--hidden-import streamlink.plugins.wwenetwork \
--hidden-import streamlink.plugins.youtube \
--hidden-import streamlink.plugins.yupptv \
--hidden-import streamlink.plugins.zattoo \
--hidden-import streamlink.plugins.zdf_mediathek \
--hidden-import streamlink.plugins.zeenews \
--hidden-import streamlink.plugins.zengatv \
--hidden-import streamlink.plugins.zhanqi \
--noconsole \
--windowed


set -- "$@" \
    --additional-hooks-dir=./ \
    --onefile pyvls.pyw

echo
echo "Running: $PYTHON_CMD -m PyInstaller $*"
echo

"$PYTHON_CMD" -m PyInstaller "$@"
the PyInstaller argument list
#
# We use "set --" to build the option list because POSIX sh has no
# arrays; this is the standard, quoting-safe way to accumulate
# variable-length argument lists in sh.
# ----------------------------------------------------------------------
set --

for f in $LIBVOSK_FILES; do
    set -- "$@" --add-data "./${f}${PYI_SEP}."
done

set -- "$@" \
    --hidden-import argparse \
    --hidden-import pysrt \
    --hidden-import six \
    --hidden-import progressbar \
    --hidden-import tqdm \
    --hidden-import requests \
    --hidden-import _cffi_backend \
    --hidden-import sounddevice

set -- "$@" \
    --additional-hooks-dir=./ \
    --onefile pyvls.pyw

echo
echo "Running: $PYTHON_CMD -m PyInstaller $*"
echo

"$PYTHON_CMD" -m PyInstaller "$@"
