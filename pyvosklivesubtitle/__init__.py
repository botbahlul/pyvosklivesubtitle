from __future__ import absolute_import, print_function, unicode_literals
import sys
import os
import time
from threading import Thread, Timer
import argparse
import sounddevice as sd
import json
import httpx
import PySimpleGUI as sg
import tkinter as tk
try:
    import queue  # Python 3 import
except ImportError:
    import Queue  # Python 2 import

import tempfile
from datetime import datetime, timedelta
import subprocess
import multiprocessing
import ctypes
import ctypes.wintypes
from streamlink import Streamlink
from streamlink.exceptions import NoPluginError, StreamlinkError, StreamError
import six
import pysrt
import shutil
import re
import select
import wave
import audioop
import math


#======================================================== ffmpeg_progress_yield ========================================================#


#import re
#import subprocess
from typing import Any, Callable, Iterator, List, Optional, Union


def to_ms(**kwargs: Union[float, int, str]) -> int:
    hour = int(kwargs.get("hour", 0))
    minute = int(kwargs.get("min", 0))
    sec = int(kwargs.get("sec", 0))
    ms = int(kwargs.get("ms", 0))

    return (hour * 60 * 60 * 1000) + (minute * 60 * 1000) + (sec * 1000) + ms


def _probe_duration(cmd: List[str]) -> Optional[int]:
    """
    Get the duration via ffprobe from input media file
    in case ffmpeg was run with loglevel=error.

    Args:
        cmd (List[str]): A list of command line elements, e.g. ["ffmpeg", "-i", ...]

    Returns:
        Optional[int]: The duration in milliseconds.
    """

    def _get_file_name(cmd: List[str]) -> Optional[str]:
        try:
            idx = cmd.index("-i")
            return cmd[idx + 1]
        except ValueError:
            return None

    file_name = _get_file_name(cmd)
    if file_name is None:
        return None

    try:
        if sys.platform == "win32":
            output = subprocess.check_output(
                [
                    "ffprobe",
                    "-loglevel",
                    "-1",
                    "-hide_banner",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    file_name,
                ],
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            output = subprocess.check_output(
                [
                    "ffprobe",
                    "-loglevel",
                    "-1",
                    "-hide_banner",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    file_name,
                ],
                universal_newlines=True,
            )

        return int(float(output.strip()) * 1000)
    except Exception:
        # TODO: add logging
        return None


def _uses_error_loglevel(cmd: List[str]) -> bool:
    try:
        idx = cmd.index("-loglevel")
        if cmd[idx + 1] == "error":
            return True
        else:
            return False
    except ValueError:
        return False


class FfmpegProgress:
    DUR_REGEX = re.compile(
        r"Duration: (?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})\.(?P<ms>\d{2})"
    )
    TIME_REGEX = re.compile(
        r"out_time=(?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})\.(?P<ms>\d{2})"
    )

    def __init__(self, cmd: List[str], dry_run: bool = False) -> None:
        """Initialize the FfmpegProgress class.

        Args:
            cmd (List[str]): A list of command line elements, e.g. ["ffmpeg", "-i", ...]
            dry_run (bool, optional): Only show what would be done. Defaults to False.
        """
        self.cmd = cmd
        self.stderr: Union[str, None] = None
        self.dry_run = dry_run
        self.process: Any = None
        self.stderr_callback: Union[Callable[[str], None], None] = None
        if sys.platform == "win32":
            self.base_popen_kwargs = {
                "stdin": subprocess.PIPE,  # Apply stdin isolation by creating separate pipe.
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "universal_newlines": False,
                "shell": True,
            }
        else:
            self.base_popen_kwargs = {
                "stdin": subprocess.PIPE,  # Apply stdin isolation by creating separate pipe.
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "universal_newlines": False,
            }

    def set_stderr_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set a callback function to be called on stderr output.
        The callback function must accept a single string argument.
        Note that this is called on every line of stderr output, so it can be called a lot.
        Also note that stdout/stderr are joined into one stream, so you might get stdout output in the callback.

        Args:
            callback (Callable[[str], None]): A callback function that accepts a single string argument.
        """
        if not callable(callback) or len(callback.__code__.co_varnames) != 1:
            raise ValueError(
                "Callback must be a function that accepts only one argument"
            )

        self.stderr_callback = callback

    def run_command_with_progress(
        self, popen_kwargs=None, duration_override: Union[float, None] = None
    ) -> Iterator[int]:
        """
        Run an ffmpeg command, trying to capture the process output and calculate
        the duration / progress.
        Yields the progress in percent.

        Args:
            popen_kwargs (dict, optional): A dict to specify extra arguments to the popen call, e.g. { creationflags: CREATE_NO_WINDOW }
            duration_override (float, optional): The duration in seconds. If not specified, it will be calculated from the ffmpeg output.

        Raises:
            RuntimeError: If the command fails, an exception is raised.

        Yields:
            Iterator[int]: A generator that yields the progress in percent.
        """
        if self.dry_run:
            return self.cmd

        total_dur: Union[None, int] = None
        if _uses_error_loglevel(self.cmd):
            total_dur = _probe_duration(self.cmd)

        cmd_with_progress = (
            [self.cmd[0]] + ["-progress", "-", "-nostats"] + self.cmd[1:]
        )

        stderr = []
        base_popen_kwargs = self.base_popen_kwargs.copy()
        if popen_kwargs is not None:
            base_popen_kwargs.update(popen_kwargs)

        if sys.platform == "wind32":
            self.process = subprocess.Popen(
                cmd_with_progress,
                **base_popen_kwargs,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )  # type: ignore
        else:
            self.process = subprocess.Popen(
                cmd_with_progress,
                **base_popen_kwargs,
            )  # type: ignore

        yield 0

        while True:
            if self.process.stdout is None:
                continue

            stderr_line = (
                self.process.stdout.readline().decode("utf-8", errors="replace").strip()
            )

            if self.stderr_callback:
                self.stderr_callback(stderr_line)

            if stderr_line == "" and self.process.poll() is not None:
                break

            stderr.append(stderr_line.strip())

            self.stderr = "\n".join(stderr)

            if total_dur is None:
                total_dur_match = self.DUR_REGEX.search(stderr_line)
                if total_dur_match:
                    total_dur = to_ms(**total_dur_match.groupdict())
                    continue
                elif duration_override is not None:
                    # use the override (should apply in the first loop)
                    total_dur = int(duration_override * 1000)
                    continue

            if total_dur:
                progress_time = FfmpegProgress.TIME_REGEX.search(stderr_line)
                if progress_time:
                    elapsed_time = to_ms(**progress_time.groupdict())
                    yield int(elapsed_time * 100/ total_dur)

        if self.process is None or self.process.returncode != 0:
            #print(self.process)
            #print(self.process.returncode)
            _pretty_stderr = "\n".join(stderr)
            raise RuntimeError(f"Error running command {self.cmd}: {_pretty_stderr}")

        yield 100
        self.process = None

    def quit_gracefully(self) -> None:
        """
        Quit the ffmpeg process by sending 'q'

        Raises:
            RuntimeError: If no process is found.
        """
        if self.process is None:
            raise RuntimeError("No process found. Did you run the command?")

        self.process.communicate(input=b"q")
        self.process.kill()
        self.process = None

    def quit(self) -> None:
        """
        Quit the ffmpeg process by sending SIGKILL.

        Raises:
            RuntimeError: If no process is found.
        """
        if self.process is None:
            raise RuntimeError("No process found. Did you run the command?")

        self.process.kill()
        self.process = None

#=======================================================================================================================================#

#============================================================== VOSK PART ==============================================================#

import requests
from zipfile import ZipFile
from re import match
from pathlib import Path
from tqdm import tqdm
import _cffi_backend

_ffi = _cffi_backend.FFI('vosk.vosk_cffi',
    _version = 0x2601,
    _types = b'\x00\x00\x03\x0D\x00\x00\x00\x0F\x00\x00\x1B\x0D\x00\x00\x5B\x03\x00\x00\x0D\x01\x00\x00\x00\x0F\x00\x00\x0A\x0D\x00\x00\x60\x03\x00\x00\x00\x0F\x00\x00\x1E\x0D\x00\x00\x5D\x03\x00\x00\x0D\x01\x00\x00\x00\x0F\x00\x00\x1E\x0D\x00\x00\x0A\x11\x00\x00\x0D\x01\x00\x00\x5F\x03\x00\x00\x00\x0F\x00\x00\x1E\x0D\x00\x00\x0A\x11\x00\x00\x0D\x01\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x10\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x07\x0D\x00\x00\x5C\x03\x00\x00\x00\x0F\x00\x00\x07\x0D\x00\x00\x5E\x03\x00\x00\x00\x0F\x00\x00\x2A\x0D\x00\x00\x1B\x11\x00\x00\x00\x0F\x00\x00\x2A\x0D\x00\x00\x0A\x11\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x2A\x0D\x00\x00\x1E\x11\x00\x00\x07\x11\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x2A\x0D\x00\x00\x1E\x11\x00\x00\x04\x03\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x2A\x0D\x00\x00\x1E\x11\x00\x00\x61\x03\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x03\x11\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x1B\x11\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x1B\x11\x00\x00\x07\x11\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x1B\x11\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x0A\x11\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x1E\x11\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x1E\x11\x00\x00\x10\x11\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x1E\x11\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x10\x11\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x62\x0D\x00\x00\x00\x0F\x00\x00\x00\x09\x00\x00\x01\x09\x00\x00\x02\x09\x00\x00\x03\x09\x00\x00\x04\x09\x00\x00\x02\x01\x00\x00\x05\x01\x00\x00\x00\x01',
    _globals = (b'\x00\x00\x36\x23vosk_batch_model_free',0,b'\x00\x00\x00\x23vosk_batch_model_new',0,b'\x00\x00\x36\x23vosk_batch_model_wait',0,b'\x00\x00\x3C\x23vosk_batch_recognizer_accept_waveform',0,b'\x00\x00\x39\x23vosk_batch_recognizer_finish_stream',0,b'\x00\x00\x39\x23vosk_batch_recognizer_free',0,b'\x00\x00\x1A\x23vosk_batch_recognizer_front_result',0,b'\x00\x00\x20\x23vosk_batch_recognizer_get_pending_chunks',0,b'\x00\x00\x02\x23vosk_batch_recognizer_new',0,b'\x00\x00\x39\x23vosk_batch_recognizer_pop',0,b'\x00\x00\x41\x23vosk_batch_recognizer_set_nlsml',0,b'\x00\x00\x59\x23vosk_gpu_init',0,b'\x00\x00\x59\x23vosk_gpu_thread_init',0,b'\x00\x00\x23\x23vosk_model_find_word',0,b'\x00\x00\x45\x23vosk_model_free',0,b'\x00\x00\x06\x23vosk_model_new',0,b'\x00\x00\x27\x23vosk_recognizer_accept_waveform',0,b'\x00\x00\x2C\x23vosk_recognizer_accept_waveform_f',0,b'\x00\x00\x31\x23vosk_recognizer_accept_waveform_s',0,b'\x00\x00\x1D\x23vosk_recognizer_final_result',0,b'\x00\x00\x48\x23vosk_recognizer_free',0,b'\x00\x00\x09\x23vosk_recognizer_new',0,b'\x00\x00\x12\x23vosk_recognizer_new_grm',0,b'\x00\x00\x0D\x23vosk_recognizer_new_spk',0,b'\x00\x00\x1D\x23vosk_recognizer_partial_result',0,b'\x00\x00\x48\x23vosk_recognizer_reset',0,b'\x00\x00\x1D\x23vosk_recognizer_result',0,b'\x00\x00\x4F\x23vosk_recognizer_set_max_alternatives',0,b'\x00\x00\x4F\x23vosk_recognizer_set_nlsml',0,b'\x00\x00\x4F\x23vosk_recognizer_set_partial_words',0,b'\x00\x00\x4B\x23vosk_recognizer_set_spk_model',0,b'\x00\x00\x4F\x23vosk_recognizer_set_words',0,b'\x00\x00\x56\x23vosk_set_log_level',0,b'\x00\x00\x53\x23vosk_spk_model_free',0,b'\x00\x00\x17\x23vosk_spk_model_new',0),
    _struct_unions = ((b'\x00\x00\x00\x5B\x00\x00\x00\x10VoskBatchModel',),(b'\x00\x00\x00\x5C\x00\x00\x00\x10VoskBatchRecognizer',),(b'\x00\x00\x00\x5D\x00\x00\x00\x10VoskModel',),(b'\x00\x00\x00\x5E\x00\x00\x00\x10VoskRecognizer',),(b'\x00\x00\x00\x5F\x00\x00\x00\x10VoskSpkModel',)),
    _typenames = (b'\x00\x00\x00\x5BVoskBatchModel',b'\x00\x00\x00\x5CVoskBatchRecognizer',b'\x00\x00\x00\x5DVoskModel',b'\x00\x00\x00\x5EVoskRecognizer',b'\x00\x00\x00\x5FVoskSpkModel'),
)

# Remote location of the models and local folders
MODEL_PRE_URL = 'https://alphacephei.com/vosk/models/'
MODEL_LIST_URL = MODEL_PRE_URL + 'model-list.json'
MODEL_DIRS = [os.getenv('VOSK_MODEL_PATH'), Path('/usr/share/vosk'), Path.home() / 'AppData/Local/vosk', Path.home() / '.cache/vosk']

def libvoskdir():
    if sys.platform == 'win32':
        libvosk = "libvosk.dll"
    elif sys.platform == 'linux':
        libvosk = "libvosk.so"
    elif sys.platform == 'linux':
        libvosk = "libvosk.dyld"
    dlldir = os.path.abspath(os.path.dirname(__file__))
    os.environ["PATH"] = dlldir + os.pathsep + os.environ['PATH']
    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        if os.path.isfile(os.path.join(path, libvosk)):
            return path
    raise TypeError('libvosk not found')
    

def open_dll():
    dlldir = libvoskdir()
    if sys.platform == 'win32':
        # We want to load dependencies too
        os.environ["PATH"] = dlldir + os.pathsep + os.environ['PATH']
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dlldir)
        return _ffi.dlopen(os.path.join(dlldir, "libvosk.dll"))
    elif sys.platform == 'linux':
        return _ffi.dlopen(os.path.join(dlldir, "libvosk.so"))
    elif sys.platform == 'darwin':
        return _ffi.dlopen(os.path.join(dlldir, "libvosk.dyld"))
    else:
        raise TypeError("Unsupported platform")


_c = open_dll()


def list_models():
    response = requests.get(MODEL_LIST_URL)
    for model in response.json():
        print(model['name']) 


def list_languages():
    response = requests.get(MODEL_LIST_URL)
    languages = set([m['lang'] for m in response.json()])
    for lang in languages:
        print (lang)


class Model(object):
    def __init__(self, model_path=None, model_name=None, lang=None):
        if model_path != None:
            self._handle = _c.vosk_model_new(model_path.encode('utf-8'))
        else:
            model_path = self.get_model_path(model_name, lang)
            self._handle = _c.vosk_model_new(model_path.encode('utf-8'))
        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a model")

    def __del__(self):
        _c.vosk_model_free(self._handle)

    def vosk_model_find_word(self, word):
        return _c.vosk_model_find_word(self._handle, word.encode('utf-8'))

    def get_model_path(self, model_name, lang):
        if model_name is None:
            model_path = self.get_model_by_lang(lang)
        else:
            model_path = self.get_model_by_name(model_name)
        return str(model_path)

    def get_model_by_name(self, model_name):
        for directory in MODEL_DIRS:
            if directory is None or not Path(directory).exists():
                continue
            model_file_list = os.listdir(directory)
            model_file = [model for model in model_file_list if model == model_name]
            if model_file != []:
                return Path(directory, model_file[0])
        response = requests.get(MODEL_LIST_URL)
        result_model = [model['name'] for model in response.json() if model['name'] == model_name]
        if result_model == []:
            raise Exception("model name %s does not exist" % (model_name))
        else:
            self.download_model(Path(directory, result_model[0]))
            return Path(directory, result_model[0])

    def get_model_by_lang(self, lang):
        for directory in MODEL_DIRS:
            if directory is None or not Path(directory).exists():
                continue
            model_file_list = os.listdir(directory)
            model_file = [model for model in model_file_list if match(f"vosk-model(-small)?-{lang}", model)]
            if model_file != []:
                return Path(directory, model_file[0])
        response = requests.get(MODEL_LIST_URL)
        result_model = [model['name'] for model in response.json() if model['lang'] == lang and model['type'] == 'small' and model['obsolete'] == 'false']
        if result_model == []:
            raise Exception("lang %s does not exist" % (lang))
        else:
            self.download_model(Path(directory, result_model[0]))
            return Path(directory, result_model[0])

    def download_model(self, model_name):
        if not MODEL_DIRS[3].exists():
            MODEL_DIRS[3].mkdir()
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                desc=(MODEL_PRE_URL + str(model_name.name) + '.zip').split('/')[-1]) as t:
            reporthook = self.download_progress_hook(t)
            urlretrieve(MODEL_PRE_URL + str(model_name.name) + '.zip', str(model_name) + '.zip', 
                reporthook=reporthook, data=None)
            t.total = t.n
            with ZipFile(str(model_name) + '.zip', 'r') as model_ref:
                model_ref.extractall(model_name.parent)
            Path(str(model_name) + '.zip').unlink()

    def download_progress_hook(self, t):
        last_b = [0]
        def update_to(b=1, bsize=1, tsize=None):
            if tsize not in (None, -1):
                t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed
        return update_to


class SpkModel(object):
    def __init__(self, model_path):
        self._handle = _c.vosk_spk_model_new(model_path.encode('utf-8'))
        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a speaker model")

    def __del__(self):
        _c.vosk_spk_model_free(self._handle)


class KaldiRecognizer(object):
    def __init__(self, *args):
        if len(args) == 2:
            self._handle = _c.vosk_recognizer_new(args[0]._handle, args[1])
        elif len(args) == 3 and type(args[2]) is SpkModel:
            self._handle = _c.vosk_recognizer_new_spk(args[0]._handle, args[1], args[2]._handle)
        elif len(args) == 3 and type(args[2]) is str:
            self._handle = _c.vosk_recognizer_new_grm(args[0]._handle, args[1], args[2].encode('utf-8'))
        else:
            raise TypeError("Unknown arguments")

        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a recognizer")

    def __del__(self):
        _c.vosk_recognizer_free(self._handle)

    def SetMaxAlternatives(self, max_alternatives):
        _c.vosk_recognizer_set_max_alternatives(self._handle, max_alternatives)

    def SetWords(self, enable_words):
        _c.vosk_recognizer_set_words(self._handle, 1 if enable_words else 0)

    def SetPartialWords(self, enable_partial_words):
        _c.vosk_recognizer_set_partial_words(self._handle, 1 if enable_partial_words else 0)

    def SetNLSML(self, enable_nlsml):
        _c.vosk_recognizer_set_nlsml(self._handle, 1 if enable_nlsml else 0)

    def SetSpkModel(self, spk_model):
        _c.vosk_recognizer_set_spk_model(self._handle, spk_model._handle)

    def AcceptWaveform(self, data):
        res = _c.vosk_recognizer_accept_waveform(self._handle, data, len(data))
        if res < 0:
            raise Exception("Failed to process waveform")
        return res

    def Result(self):
        return _ffi.string(_c.vosk_recognizer_result(self._handle)).decode('utf-8')

    def PartialResult(self):
        return _ffi.string(_c.vosk_recognizer_partial_result(self._handle)).decode('utf-8')

    def FinalResult(self):
        return _ffi.string(_c.vosk_recognizer_final_result(self._handle)).decode('utf-8')

    def Reset(self):
        return _c.vosk_recognizer_reset(self._handle)


def SetLogLevel(level):
    return _c.vosk_set_log_level(level)


def GpuInit():
    _c.vosk_gpu_init()


def GpuThreadInit():
    _c.vosk_gpu_thread_init()


class BatchModel(object):
    def __init__(self, *args):
        self._handle = _c.vosk_batch_model_new()

        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a model")

    def __del__(self):
        _c.vosk_batch_model_free(self._handle)

    def Wait(self):
        _c.vosk_batch_model_wait(self._handle)


class BatchRecognizer(object):
    def __init__(self, *args):
        self._handle = _c.vosk_batch_recognizer_new(args[0]._handle, args[1])

        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a recognizer")

    def __del__(self):
        _c.vosk_batch_recognizer_free(self._handle)

    def AcceptWaveform(self, data):
        res = _c.vosk_batch_recognizer_accept_waveform(self._handle, data, len(data))

    def Result(self):
        ptr = _c.vosk_batch_recognizer_front_result(self._handle)
        res = _ffi.string(ptr).decode('utf-8')
        _c.vosk_batch_recognizer_pop(self._handle)
        return res

    def FinishStream(self):
        _c.vosk_batch_recognizer_finish_stream(self._handle)

    def GetPendingChunks(self):
        return _c.vosk_batch_recognizer_get_pending_chunks(self._handle)

#=======================================================================================================================================#

#============================================================== APP PARTS ==============================================================#

arraylist_models = []
arraylist_models.append("ca-es");
arraylist_models.append("zh-cn")
arraylist_models.append("cs-cz");
arraylist_models.append("nl-nl");
arraylist_models.append("en-us")
arraylist_models.append("eo-eo");
arraylist_models.append("fr-fr");
arraylist_models.append("de-de");
arraylist_models.append("hi-in");
arraylist_models.append("it-it");
arraylist_models.append("ja-jp");
arraylist_models.append("kk-kz");
arraylist_models.append("fa-ir");
arraylist_models.append("pl-pl");
arraylist_models.append("pt-pt");
arraylist_models.append("ru-ru");
arraylist_models.append("es-es");
arraylist_models.append("sv-se");
arraylist_models.append("tr-tr");
arraylist_models.append("uk-ua");
arraylist_models.append("vi-vn");

arraylist_src = []
arraylist_src.append("ca")
arraylist_src.append("zh")
arraylist_src.append("cs")
arraylist_src.append("nl")
arraylist_src.append("en")
arraylist_src.append("eo")
arraylist_src.append("fr")
arraylist_src.append("de")
arraylist_src.append("hi")
arraylist_src.append("it")
arraylist_src.append("ja")
arraylist_src.append("kk")
arraylist_src.append("fa")
arraylist_src.append("pl")
arraylist_src.append("pt")
arraylist_src.append("ru")
arraylist_src.append("es")
arraylist_src.append("sv")
arraylist_src.append("tr")
arraylist_src.append("uk")
arraylist_src.append("vi")

arraylist_src_languages = []
arraylist_src_languages.append("Catalan")
arraylist_src_languages.append("Chinese")
arraylist_src_languages.append("Czech")
arraylist_src_languages.append("Dutch")
arraylist_src_languages.append("English")
arraylist_src_languages.append("Esperanto")
arraylist_src_languages.append("French")
arraylist_src_languages.append("German")
arraylist_src_languages.append("Hindi")
arraylist_src_languages.append("Italian")
arraylist_src_languages.append("Japanese")
arraylist_src_languages.append("Kazakh")
arraylist_src_languages.append("Persian")
arraylist_src_languages.append("Polish")
arraylist_src_languages.append("Portuguese")
arraylist_src_languages.append("Russian")
arraylist_src_languages.append("Spanish")
arraylist_src_languages.append("Swedish")
arraylist_src_languages.append("Turkish")
arraylist_src_languages.append("Ukrainian")
arraylist_src_languages.append("Vietnamese")

map_model_language = dict(zip(arraylist_src_languages, arraylist_models))
map_src_of_language = dict(zip(arraylist_src_languages, arraylist_src))
map_language_of_src = dict(zip(arraylist_src, arraylist_src_languages))

arraylist_dst = []
arraylist_dst.append("af")
arraylist_dst.append("sq")
arraylist_dst.append("am")
arraylist_dst.append("ar")
arraylist_dst.append("hy")
arraylist_dst.append("as")
arraylist_dst.append("ay")
arraylist_dst.append("az")
arraylist_dst.append("bm")
arraylist_dst.append("eu")
arraylist_dst.append("be")
arraylist_dst.append("bn")
arraylist_dst.append("bho")
arraylist_dst.append("bs")
arraylist_dst.append("bg")
arraylist_dst.append("ca")
arraylist_dst.append("ceb")
arraylist_dst.append("ny")
arraylist_dst.append("zh-CN")
arraylist_dst.append("zh-TW")
arraylist_dst.append("co")
arraylist_dst.append("hr")
arraylist_dst.append("cs")
arraylist_dst.append("da")
arraylist_dst.append("dv")
arraylist_dst.append("doi")
arraylist_dst.append("nl")
arraylist_dst.append("en")
arraylist_dst.append("eo")
arraylist_dst.append("et")
arraylist_dst.append("ee")
arraylist_dst.append("fil")
arraylist_dst.append("fi")
arraylist_dst.append("fr")
arraylist_dst.append("fy")
arraylist_dst.append("gl")
arraylist_dst.append("ka")
arraylist_dst.append("de")
arraylist_dst.append("el")
arraylist_dst.append("gn")
arraylist_dst.append("gu")
arraylist_dst.append("ht")
arraylist_dst.append("ha")
arraylist_dst.append("haw")
arraylist_dst.append("he")
arraylist_dst.append("hi")
arraylist_dst.append("hmn")
arraylist_dst.append("hu")
arraylist_dst.append("is")
arraylist_dst.append("ig")
arraylist_dst.append("ilo")
arraylist_dst.append("id")
arraylist_dst.append("ga")
arraylist_dst.append("it")
arraylist_dst.append("ja")
arraylist_dst.append("jv")
arraylist_dst.append("kn")
arraylist_dst.append("kk")
arraylist_dst.append("km")
arraylist_dst.append("rw")
arraylist_dst.append("gom")
arraylist_dst.append("ko")
arraylist_dst.append("kri")
arraylist_dst.append("kmr")
arraylist_dst.append("ckb")
arraylist_dst.append("ky")
arraylist_dst.append("lo")
arraylist_dst.append("la")
arraylist_dst.append("lv")
arraylist_dst.append("ln")
arraylist_dst.append("lt")
arraylist_dst.append("lg")
arraylist_dst.append("lb")
arraylist_dst.append("mk")
arraylist_dst.append("mg")
arraylist_dst.append("ms")
arraylist_dst.append("ml")
arraylist_dst.append("mt")
arraylist_dst.append("mi")
arraylist_dst.append("mr")
arraylist_dst.append("mni-Mtei")
arraylist_dst.append("lus")
arraylist_dst.append("mn")
arraylist_dst.append("my")
arraylist_dst.append("ne")
arraylist_dst.append("no")
arraylist_dst.append("or")
arraylist_dst.append("om")
arraylist_dst.append("ps")
arraylist_dst.append("fa")
arraylist_dst.append("pl")
arraylist_dst.append("pt")
arraylist_dst.append("pa")
arraylist_dst.append("qu")
arraylist_dst.append("ro")
arraylist_dst.append("ru")
arraylist_dst.append("sm")
arraylist_dst.append("sa")
arraylist_dst.append("gd")
arraylist_dst.append("nso")
arraylist_dst.append("sr")
arraylist_dst.append("st")
arraylist_dst.append("sn")
arraylist_dst.append("sd")
arraylist_dst.append("si")
arraylist_dst.append("sk")
arraylist_dst.append("sl")
arraylist_dst.append("so")
arraylist_dst.append("es")
arraylist_dst.append("su")
arraylist_dst.append("sw")
arraylist_dst.append("sv")
arraylist_dst.append("tg")
arraylist_dst.append("ta")
arraylist_dst.append("tt")
arraylist_dst.append("te")
arraylist_dst.append("th")
arraylist_dst.append("ti")
arraylist_dst.append("ts")
arraylist_dst.append("tr")
arraylist_dst.append("tk")
arraylist_dst.append("tw")
arraylist_dst.append("uk")
arraylist_dst.append("ur")
arraylist_dst.append("ug")
arraylist_dst.append("uz")
arraylist_dst.append("vi")
arraylist_dst.append("cy")
arraylist_dst.append("xh")
arraylist_dst.append("yi")
arraylist_dst.append("yo")
arraylist_dst.append("zu")

arraylist_dst_languages = []
arraylist_dst_languages.append("Afrikaans");
arraylist_dst_languages.append("Albanian");
arraylist_dst_languages.append("Amharic");
arraylist_dst_languages.append("Arabic");
arraylist_dst_languages.append("Armenian");
arraylist_dst_languages.append("Assamese");
arraylist_dst_languages.append("Aymara");
arraylist_dst_languages.append("Azerbaijani");
arraylist_dst_languages.append("Bambara");
arraylist_dst_languages.append("Basque");
arraylist_dst_languages.append("Belarusian");
arraylist_dst_languages.append("Bengali");
arraylist_dst_languages.append("Bhojpuri");
arraylist_dst_languages.append("Bosnian");
arraylist_dst_languages.append("Bulgarian");
arraylist_dst_languages.append("Catalan");
arraylist_dst_languages.append("Cebuano");
arraylist_dst_languages.append("Chichewa");
arraylist_dst_languages.append("Chinese (Simplified)");
arraylist_dst_languages.append("Chinese (Traditional)");
arraylist_dst_languages.append("Corsican");
arraylist_dst_languages.append("Croatian");
arraylist_dst_languages.append("Czech");
arraylist_dst_languages.append("Danish");
arraylist_dst_languages.append("Dhivehi");
arraylist_dst_languages.append("Dogri");
arraylist_dst_languages.append("Dutch");
arraylist_dst_languages.append("English");
arraylist_dst_languages.append("Esperanto");
arraylist_dst_languages.append("Estonian");
arraylist_dst_languages.append("Ewe");
arraylist_dst_languages.append("Filipino");
arraylist_dst_languages.append("Finnish");
arraylist_dst_languages.append("French");
arraylist_dst_languages.append("Frisian");
arraylist_dst_languages.append("Galician");
arraylist_dst_languages.append("Georgian");
arraylist_dst_languages.append("German");
arraylist_dst_languages.append("Greek");
arraylist_dst_languages.append("Guarani");
arraylist_dst_languages.append("Gujarati");
arraylist_dst_languages.append("Haitian Creole");
arraylist_dst_languages.append("Hausa");
arraylist_dst_languages.append("Hawaiian");
arraylist_dst_languages.append("Hebrew");
arraylist_dst_languages.append("Hindi");
arraylist_dst_languages.append("Hmong");
arraylist_dst_languages.append("Hungarian");
arraylist_dst_languages.append("Icelandic");
arraylist_dst_languages.append("Igbo");
arraylist_dst_languages.append("Ilocano");
arraylist_dst_languages.append("Indonesian");
arraylist_dst_languages.append("Irish");
arraylist_dst_languages.append("Italian");
arraylist_dst_languages.append("Japanese");
arraylist_dst_languages.append("Javanese");
arraylist_dst_languages.append("Kannada");
arraylist_dst_languages.append("Kazakh");
arraylist_dst_languages.append("Khmer");
arraylist_dst_languages.append("Kinyarwanda");
arraylist_dst_languages.append("Konkani");
arraylist_dst_languages.append("Korean");
arraylist_dst_languages.append("Krio");
arraylist_dst_languages.append("Kurdish (Kurmanji)");
arraylist_dst_languages.append("Kurdish (Sorani)");
arraylist_dst_languages.append("Kyrgyz");
arraylist_dst_languages.append("Lao");
arraylist_dst_languages.append("Latin");
arraylist_dst_languages.append("Latvian");
arraylist_dst_languages.append("Lingala");
arraylist_dst_languages.append("Lithuanian");
arraylist_dst_languages.append("Luganda");
arraylist_dst_languages.append("Luxembourgish");
arraylist_dst_languages.append("Macedonian");
arraylist_dst_languages.append("Malagasy");
arraylist_dst_languages.append("Malay");
arraylist_dst_languages.append("Malayalam");
arraylist_dst_languages.append("Maltese");
arraylist_dst_languages.append("Maori");
arraylist_dst_languages.append("Marathi");
arraylist_dst_languages.append("Meiteilon (Manipuri)");
arraylist_dst_languages.append("Mizo");
arraylist_dst_languages.append("Mongolian");
arraylist_dst_languages.append("Myanmar (Burmese)");
arraylist_dst_languages.append("Nepali");
arraylist_dst_languages.append("Norwegian");
arraylist_dst_languages.append("Odiya (Oriya)");
arraylist_dst_languages.append("Oromo");
arraylist_dst_languages.append("Pashto");
arraylist_dst_languages.append("Persian");
arraylist_dst_languages.append("Polish");
arraylist_dst_languages.append("Portuguese");
arraylist_dst_languages.append("Punjabi");
arraylist_dst_languages.append("Quechua");
arraylist_dst_languages.append("Romanian");
arraylist_dst_languages.append("Russian");
arraylist_dst_languages.append("Samoan");
arraylist_dst_languages.append("Sanskrit");
arraylist_dst_languages.append("Scots Gaelic");
arraylist_dst_languages.append("Sepedi");
arraylist_dst_languages.append("Serbian");
arraylist_dst_languages.append("Sesotho");
arraylist_dst_languages.append("Shona");
arraylist_dst_languages.append("Sindhi");
arraylist_dst_languages.append("Sinhala");
arraylist_dst_languages.append("Slovak");
arraylist_dst_languages.append("Slovenian");
arraylist_dst_languages.append("Somali");
arraylist_dst_languages.append("Spanish");
arraylist_dst_languages.append("Sundanese");
arraylist_dst_languages.append("Swahili");
arraylist_dst_languages.append("Swedish");
arraylist_dst_languages.append("Tajik");
arraylist_dst_languages.append("Tamil");
arraylist_dst_languages.append("Tatar");
arraylist_dst_languages.append("Telugu");
arraylist_dst_languages.append("Thai");
arraylist_dst_languages.append("Tigrinya");
arraylist_dst_languages.append("Tsonga");
arraylist_dst_languages.append("Turkish");
arraylist_dst_languages.append("Turkmen");
arraylist_dst_languages.append("Twi (Akan)");
arraylist_dst_languages.append("Ukrainian");
arraylist_dst_languages.append("Urdu");
arraylist_dst_languages.append("Uyghur");
arraylist_dst_languages.append("Uzbek");
arraylist_dst_languages.append("Vietnamese");
arraylist_dst_languages.append("Welsh");
arraylist_dst_languages.append("Xhosa");
arraylist_dst_languages.append("Yiddish");
arraylist_dst_languages.append("Yoruba");
arraylist_dst_languages.append("Zulu");

map_dst_of_language = dict(zip(arraylist_dst_languages, arraylist_dst))
map_language_of_dst = dict(zip(arraylist_dst, arraylist_dst_languages))

q = queue.Queue()
SampleRate = None
Device = None
wav_filepath = None
recognizing = False

thread_record_streaming = None
thread_recognize = None
thread_timed_translate = None
thread_vosk_transcribe = None
all_transcribe_threads = None
thread_save_temporary_transcriptions = None
thread_save_declared_subtitle_filename = None
regions = None
transcriptions = None
translated_transcriptions = None
text = None
translated_text = None

#----------------------------------------------------------- MISC FUNCTIONS ------------------------------------------------------------#

def srt_formatter(subtitles, padding_before=0, padding_after=0):
    """
    Serialize a list of subtitles according to the SRT format, with optional time padding.
    """
    sub_rip_file = pysrt.SubRipFile()
    for i, ((start, end), text) in enumerate(subtitles, start=1):
        item = pysrt.SubRipItem()
        item.index = i
        item.text = six.text_type(text)
        item.start.seconds = max(0, start - padding_before)
        item.end.seconds = end + padding_after
        sub_rip_file.append(item)
    return '\n'.join(six.text_type(item) for item in sub_rip_file)


def vtt_formatter(subtitles, padding_before=0, padding_after=0):
    """
    Serialize a list of subtitles according to the VTT format, with optional time padding.
    """
    text = srt_formatter(subtitles, padding_before, padding_after)
    text = 'WEBVTT\n\n' + text.replace(',', '.')
    return text


def json_formatter(subtitles):
    """
    Serialize a list of subtitles as a JSON blob.
    """
    subtitle_dicts = [
        {
            'start': start,
            'end': end,
            'content': text,
        }
        for ((start, end), text)
        in subtitles
    ]
    return json.dumps(subtitle_dicts)


def raw_formatter(subtitles):
    """
    Serialize a list of subtitles as a newline-delimited string.
    """
    return ' '.join(text for (_rng, text) in subtitles)


FORMATTERS = {
    'srt': srt_formatter,
    'vtt': vtt_formatter,
    'json': json_formatter,
    'raw': raw_formatter,
}


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def worker_recognize(src, dst):
    global main_window, recognizing, text, Device, SampleRate, wav_filepath, regions, transcriptions, \
        start_button_click_time, ffmpeg_start_run_time, tmp_recorded_streaming_filepath, get_tmp_recorded_streaming_filepath_time, \
        loading_time, first_streaming_duration_recorded

    p = 0
    f = 0
    l = 0

    start_time = None
    end_time = None
    first_streaming_duration_recorded = None
    absolute_start_time = None
    audio_filename, SampleRate = None, None
    first_regions = None

    time_value_filename = "time_value"
    time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
    #print("time_value_filepath = {}".format(time_value_filepath))
    #print("os.path.isfile(time_value_filepath) = {}".format(os.path.isfile(time_value_filepath)))
    if os.path.isfile(time_value_filepath):
        time_value_file = open(time_value_filepath, "r")
        time_value_string = time_value_file.read()
        if time_value_string:
            first_streaming_duration_recorded = datetime.strptime(time_value_string, "%H:%M:%S.%f") - datetime(1900, 1, 1)
        else:
            first_streaming_duration_recorded = None
        time_value_file.close()

    try:
        if SampleRate is None:
            #soundfile expects an int, sounddevice provides a float:
            device_info = sd.query_devices(Device, "input")
            SampleRate = int(device_info["default_samplerate"])

        if wav_filepath:
            dump_fn = open(wav_filepath, "wb")
        else:
            dump_fn = None

        model = Model(lang = src)

        with sd.RawInputStream(samplerate=SampleRate, blocksize=8000, device=Device, dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, SampleRate)

            while recognizing:
                if not recognizing:
                    rec = None
                    data = None
                    result = None
                    results_text = None
                    partial_results_text = None
                    #print("BREAK")
                    break

                data = q.get()

                results_text = ''
                partial_results_text = ''

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())

                    if result["text"] and absolute_start_time:
                        text = result["text"]

                        if len(text) > 0:
                            main_window.write_event_value('-EVENT-RESULTS-', text)
                            transcriptions.append(text)

                            f += 1
                            p = 0

                            absolute_end_time = datetime.now()
                            end_time = start_time + (absolute_end_time - absolute_start_time)
                            regions.append((start_time.total_seconds(), end_time.total_seconds()))

                            #start_time_str = "{:02d}:{:02d}:{:06.3f}".format(
                            start_time_str = "{:02d}:{:02d}:{:02.0f}.{:03.0f}".format(
                                start_time.seconds // 3600,  # hours
                                (start_time.seconds // 60) % 60,  # minutes
                                start_time.seconds % 60, # seconds
                                start_time.microseconds / 1000000  # microseconds
                            )

                            #end_time_str = "{:02d}:{:02d}:{:06.3f}".format(
                            end_time_str = "{:02d}:{:02d}:{:02.0f}.{:03.0f}".format(
                                end_time.seconds // 3600,  # hours
                                (end_time.seconds // 60) % 60,  # minutes
                                end_time.seconds % 60, # seconds
                                end_time.microseconds / 1000000  # microseconds
                            )

                            time_stamp_string = start_time_str + "-" + end_time_str

                            tmp_src_txt_transcription_filename = "record.txt"
                            tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
                            tmp_src_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "a")
                            tmp_src_txt_transcription_file.write(time_stamp_string + " " + text + "\n")
                            tmp_src_txt_transcription_file.close()

                else:
                    result = json.loads(rec.PartialResult())
                    if result["partial"]:
                        text = result["partial"]

                        if len(text) == 0:
                            main_window.write_event_value('-EVENT-NULL-RESULTS-', text)

                        if len(text)>0:
                            main_window.write_event_value('-EVENT-PARTIAL-RESULTS-', text)

                            # IF RECORD STREAMING
                            if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:

                                if ffmpeg_start_run_time and l == 0:

                                    loading_time = datetime.now() - ffmpeg_start_run_time - (ffmpeg_start_run_time - get_tmp_recorded_streaming_filepath_time)
                                    if first_streaming_duration_recorded:

                                        # A ROUGH GUESS OF THE FIRST PARTIAL RESULT START TIME
                                        # ACTUAL START TIME WILL BE PROVIED BY vosk_transcribe() FUNCTION
                                        #start_time = first_streaming_duration_recorded
                                        #start_time = first_streaming_duration_recorded + loading_time

                                        start_time = first_streaming_duration_recorded + timedelta(seconds=0.5)

                                        # MAKE SURE THAT IT'S EXECUTED ONLY ONCE
                                        l += 1

                                    else:
                                        time_value_filename = "time_value"
                                        time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
                                        #print("time_value_filepath = {}".format(time_value_filepath))
                                        #print("os.path.isfile(time_value_filepath) = {}".format(os.path.isfile(time_value_filepath)))
                                        if os.path.isfile(time_value_filepath):
                                            time_value_file = open(time_value_filepath, "r")
                                            time_value_string = time_value_file.read()
                                            if time_value_string:
                                                first_streaming_duration_recorded = datetime.strptime(time_value_string, "%H:%M:%S.%f") - datetime(1900, 1, 1)
                                            else:
                                                first_streaming_duration_recorded = None
                                            time_value_file.close()

                                else:
                                    if p == 0:
                                        absolute_start_time = datetime.now()

                                        if end_time:
                                            start_time = end_time
                                        p += 1

                            # NOT RECORD STREAMING
                            else:
                                if l == 0:
                                    now_datetime = datetime.now()
                                    now_time = now_datetime.time()
                                    now_microseconds = (now_time.hour * 3600 + now_time.minute * 60 + now_time.second) * 1000000 + now_time.microsecond

                                    start_time = timedelta(microseconds=now_microseconds)

                                    l += 1

                                else:
                                    if p == 0:
                                        absolute_start_time = datetime.now()

                                        if end_time:
                                            start_time = end_time
                                        p += 1

                if dump_fn is not None:
                    dump_fn.write(data)

    except KeyboardInterrupt:
        recognizing==False
        rec = None
        data = None
        parser.exit(0)

    except Exception as e:
        parser.exit(type(e).__name__ + ": " + str(e))


class SentenceTranslator(object):
    def __init__(self, src, dst, patience=-1):
        self.src = src
        self.dst = dst
        self.patience = patience

    def __call__(self, sentence):
        translated_sentence = []
        # handle the special case: empty string.
        if not sentence:
            return None

        translated_sentence = GoogleTranslate(sentence, src=self.src, dst=self.dst)

        fail_to_translate = translated_sentence[-1] == '\n'
        while fail_to_translate and patience:
            translated_sentence = GoogleTranslate(translated_sentence, src=self.src, dst=self.dst).text
            if translated_sentence[-1] == '\n':
                if patience == -1:
                    continue
                patience -= 1
            else:
                fail_to_translate = False
        return translated_sentence

'''
def GoogleTranslate(text, src, dst):
    url = 'https://translate.googleapis.com/translate_a/'
    params = 'single?client=gtx&sl='+src+'&tl='+dst+'&dt=t&q='+text;
    with httpx.Client(http2=True) as client:
        client.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://translate.google.com',})
        response = client.get(url+params)
        #print("response.status_code = {}".format(response.status_code))
        if response.status_code == 200:
            response_json = response.json()[0]
            #print("response_json = {}".format(response_json))
            if response_json!= None:
                length = len(response_json)
                #print("length = {}".format(length))
                translation = ""
                for i in range(length):
                    #print("{} {}".format(i, response_json[i][0]))
                    translation = translation + response_json[i][0]
                return translation
        return
'''

def GoogleTranslate(text, src, dst):
    url = 'https://translate.googleapis.com/translate_a/'
    params = 'single?client=gtx&sl='+src+'&tl='+dst+'&dt=t&q='+text;
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://translate.google.com',}
    response = requests.get(url+params, headers=headers)
    if response.status_code == 200:
        response_json = response.json()[0]
        length = len(response_json)
        translation = ""
        for i in range(length):
            translation = translation + response_json[i][0]
        return translation
    return


def worker_timed_translate(src, dst):
    global main_window, recognizing

    while (recognizing==True):
        if not recognizing:
            break
        partial_result_filename = "partial_result"
        partial_result_filepath = os.path.join(tempfile.gettempdir(), partial_result_filename)
        if os.path.isfile(partial_result_filepath):
            partial_result_file = open(partial_result_filepath, "r")
            partial_result = partial_result_file.read()

            n_words = len(partial_result.split())
            if partial_result and n_words % 3 == 0:
                translated_text = GoogleTranslate(partial_result, src, dst)
                main_window.write_event_value('-EVENT-VOICE-TRANSLATED-', translated_text)


def stop_thread(thread):
    global main_window

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        main_window.write_event_value("-EXCEPTION-", "nonexistent thread id")
        #raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        main_window.write_event_value("-EXCEPTION-", "PyThreadState_SetAsyncExc failed")
        #raise SystemError("PyThreadState_SetAsyncExc failed")


def time_string_to_timedelta(time_string):
    # Split the time string into its components
    hours, minutes, seconds = map(float, time_string.split(':'))
    seconds, microseconds = divmod(seconds, 1)
    microseconds *= 1000000

    # Create a timedelta object with the given components
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)


def is_streaming_url(url):

    streamlink = Streamlink()

    if is_valid_url(url):
        #print("is_valid_url(url) = {}".format(is_valid_url(url)))
        try:
            os.environ['STREAMLINK_DIR'] = './streamlink/'
            os.environ['STREAMLINK_PLUGINS'] = './streamlink/plugins/'
            os.environ['STREAMLINK_PLUGIN_DIR'] = './streamlink/plugins/'

            streams = streamlink.streams(url)
            if streams:
                #print("is_streams = {}".format(True))
                return True
            else:
                #print("is_streams = {}".format(False))
                return False

        except OSError:
            #print("is_streams = OSError")
            return False
        except ValueError:
            #print("is_streams = ValueError")
            return False
        except KeyError:
            #print("is_streams = KeyError")
            return False
        except RuntimeError:
            #print("is_streams = RuntimeError")
            return False
        except NoPluginError:
            #print("is_streams = NoPluginError")
            return False
        except StreamlinkError:
            return False
            #print("is_streams = StreamlinkError")
        except StreamError:
            return False
            #print("is_streams = StreamlinkError")
        except NotImplementedError:
            #print("is_streams = NotImplementedError")
            return False
        except Exception as e:
            #print("is_streams = {}".format(e))
            return False
    else:
        #print("is_valid_url(url) = {}".format(is_valid_url(url)))
        return False


def is_valid_url(url):
    try:
        response = httpx.head(url)
        response.raise_for_status()
        return True
    except (httpx.HTTPError, ValueError):
        return False


def record_streaming_windows(hls_url, filename):
    global ffmpeg_start_run_time, first_streaming_duration_recorded, recognizing, main_window

    #ffmpeg_cmd = ['ffmpeg', '-y', '-i', hls_url,  '-movflags', '+frag_keyframe+separate_moof+omit_tfhd_offset+empty_moov', '-fflags', 'nobuffer', '-loglevel', 'error',  f'{filename}']
    ffmpeg_cmd = ['ffmpeg', '-y', '-i', f'{hls_url}',  '-movflags', '+frag_keyframe+separate_moof+omit_tfhd_offset+empty_moov', '-fflags', 'nobuffer', f'{filename}']

    ffmpeg_start_run_time = datetime.now()
    #print("ffmpeg_start_run_time = {}".format(ffmpeg_start_run_time))

    first_streaming_duration_recorded = None

    i = 0
    j = 0

    process = subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    #process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)


    msg = "RECORDING"
    main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

    while recognizing:
        if not recognizing:
            break

        #for line in iter(process.stderr.readline, b''):
        for line in iter(process.stderr.readline, b''):
            line = line.decode('utf-8').rstrip()

            if 'time=' in line:

                # Extract the time value from the progress information
                time_str = line.split('time=')[1].split()[0]
                #print("time_str = {}".format(time_str))

                if i == 0:
                    ffmpeg_start_write_time = datetime.now()
                    #print("ffmpeg_start_write_time = {}".format(ffmpeg_start_write_time))

                    first_streaming_duration_recorded = datetime.strptime(time_str, "%H:%M:%S.%f") - datetime(1900, 1, 1)
                    #print("first_streaming_duration_recorded = {}".format(first_streaming_duration_recorded))

                    # MAKE SURE THAT first_streaming_duration_recorded EXECUTED ONLY ONCE
                    i += 1

                    time_value_filename = "time_value"
                    time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
                    time_value_file = open(time_value_filepath, "w")
                    time_value_file.write(time_str)
                    time_value_file.close()

                streaming_duration_recorded = datetime.strptime(time_str, "%H:%M:%S.%f") - datetime(1900, 1, 1)
                #print("streaming_duration_recorded = {}".format(streaming_duration_recorded))
                main_window.write_event_value('-EVENT-STREAMING-DURATION-RECORDED-', streaming_duration_recorded)

    process.wait()


# subprocess.Popen(ffmpeg_cmd) THREAD BEHAVIOR IS DIFFERENT IN LINUX
def record_streaming_linux(url, output_file):
    global recognizing, ffmpeg_start_run_time, first_streaming_duration_recorded, main_window

    #ffmpeg_cmd = ['ffmpeg', '-y', '-i', url, '-c', 'copy', '-bsf:a', 'aac_adtstoasc', '-f', 'mp4', output_file]
    ffmpeg_cmd = ['ffmpeg', '-y', '-i', f'{url}',  '-movflags', '+frag_keyframe+separate_moof+omit_tfhd_offset+empty_moov', '-fflags', 'nobuffer', f'{output_file}']

    ffmpeg_start_run_time = datetime.now()

    # Define a function to run the ffmpeg process in a separate thread
    def run_ffmpeg():
        i = 0
        line = None

        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        msg = "RECORDING"
        main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

        # Set up a timer to periodically check for new output
        timeout = 1.0
        timer = Timer(timeout, lambda: None)
        timer.start()

        # Read output from ffmpeg process
        while recognizing:
            # Check if the process has finished
            if process.poll() is not None:
                break

            # Check if there is new output to read, special for linux
            rlist, _, _ = select.select([process.stderr], [], [], timeout)
            if not rlist:
                continue

            # Read the new output and print it
            line = rlist[0].readline().strip()
            #print(line)

            # Search for the time information in the output and print it
            time_str = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)

            if time_str:
                time_value = time_str.group(1)
                #print(f"Time: {time_value}")

                if i == 0:
                    ffmpeg_start_write_time = datetime.now()
                    #print("ffmpeg_start_write_time = {}".format(ffmpeg_start_write_time))

                    first_streaming_duration_recorded = datetime.strptime(str(time_value), "%H:%M:%S.%f") - datetime(1900, 1, 1)
                    #print("first_streaming_duration_recorded = {}".format(first_streaming_duration_recorded))

                    # MAKE SURE THAT first_streaming_duration_recorded EXECUTED ONLY ONCE
                    i += 1

                    #print("writing time_value")
                    time_value_filename = "time_value"
                    time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
                    time_value_file = open(time_value_filepath, "w")
                    time_value_file.write(str(time_value))
                    time_value_file.close()
                    #print("time_value = {}".format(time_value))

                streaming_duration_recorded = datetime.strptime(str(time_value), "%H:%M:%S.%f") - datetime(1900, 1, 1)
                #print("streaming_duration_recorded = {}".format(streaming_duration_recorded))
                main_window.write_event_value('-EVENT-STREAMING-DURATION-RECORDED-', streaming_duration_recorded)

            # Restart the timer to check for new output
            timer.cancel()
            timer = Timer(timeout, lambda: None)
            if recognizing: timer.start()

        # Close the ffmpeg process and print the return code
        process.stdout.close()
        process.stderr.close()

    # Start the thread to run ffmpeg
    thread = Thread(target=run_ffmpeg)
    if recognizing: thread.start()

    # Return the thread object so that the caller can join it if needed
    return thread


def stop_ffmpeg_windows():

    tasklist_output = subprocess.check_output(['tasklist'], creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8')
    ffmpeg_pid = None
    for line in tasklist_output.split('\n'):
        if "ffmpeg" in line:
            ffmpeg_pid = line.split()[1]
            break
    if ffmpeg_pid:
        devnull = open(os.devnull, 'w')
        #subprocess.Popen(['taskkill', '/F', '/T', '/PID', ffmpeg_pid], stdout=devnull, stderr=devnull, creationflags=subprocess.CREATE_NO_WINDOW)
        subprocess.Popen(['taskkill', '/F', '/T', '/PID', ffmpeg_pid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)


def stop_ffmpeg_linux():

    process_name = 'ffmpeg'
    try:
        output = subprocess.check_output(['ps', '-ef'])
        pid = [line.split()[1] for line in output.decode('utf-8').split('\n') if process_name in line][0]
        subprocess.call(['kill', '-9', str(pid)])
        #print(f"{process_name} has been killed")
    except IndexError:
        #print(f"{process_name} is not running")
        pass

def stop_record_streaming_windows():
    global main_window, thread_record_streaming

    if thread_record_streaming and thread_record_streaming.is_alive():
        # Use ctypes to call the TerminateThread() function from the Windows API
        # This forcibly terminates the thread, which can be unsafe in some cases
        kernel32 = ctypes.windll.kernel32
        thread_handle = kernel32.OpenThread(1, False, thread_record_streaming.ident)
        ret = kernel32.TerminateThread(thread_handle, 0)
        if ret == 0:
            msg = "TERMINATION ERROR!"
        else:
            msg = "TERMINATED"
            thread_record_streaming = None
        if main_window: main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

        tasklist_output = subprocess.check_output(['tasklist'], creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8')
        ffmpeg_pid = None
        for line in tasklist_output.split('\n'):
            if "ffmpeg" in line:
                ffmpeg_pid = line.split()[1]
                break
        if ffmpeg_pid:
            devnull = open(os.devnull, 'w')
            #subprocess.Popen(['taskkill', '/F', '/T', '/PID', ffmpeg_pid], stdout=devnull, stderr=devnull, creationflags=subprocess.CREATE_NO_WINDOW)
            subprocess.Popen(['taskkill', '/F', '/T', '/PID', ffmpeg_pid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
        else:
            msg = 'FFMPEG IS NOT RUNNING'
            if main_window: main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

    else:
        msg = "NOT RECORDING"
        main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)


def stop_record_streaming_linux():
    global main_window, thread_record_streaming

    if thread_record_streaming and thread_record_streaming.is_alive():
        #print("thread_record_streaming.is_alive()")
        exc = ctypes.py_object(SystemExit)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_record_streaming.ident), exc)
        if res == 0:
            raise ValueError("nonexistent thread id")
        elif res > 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_record_streaming.ident, None)
            msg = "PyThreadState_SetAsyncExc failed"
            main_window.write_event_value('-EXCEPTION-', msg)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    msg = "TERMINATED"
    if main_window: main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

    ffmpeg_pid = subprocess.check_output(['pgrep', '-f', 'ffmpeg']).strip()
    if ffmpeg_pid:
        subprocess.Popen(['kill', ffmpeg_pid])
    else:
        msg = 'FFMPEG IS NOT RUNNING'
        if main_window: main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)


def is_same_language(lang1, lang2):
    return lang1.split("-")[0] == lang2.split("-")[0]


def which(program):
    """
    Return the path for a given executable.
    """
    def is_exe(file_path):
        """
        Checks whether a file is executable.
        """
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def ffmpeg_check():
    """
    Return the ffmpeg executable name. "None" returned when no executable exists.
    """
    if which("ffmpeg"):
        return "ffmpeg"
    if which("ffmpeg.exe"):
        return "ffmpeg.exe"
    return None


def extract_audio(filename, n_channels=1, rate=16000):
    global not_transcribing, main_window

    if not_transcribing: return

    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

    if not os.path.isfile(filename):
        #print("The given file does not exist: {0}".format(filename))
        #raise Exception("Invalid filepath: {0}".format(filename))
        not_transcribing=True
        window_key = '-OUTPUT-MESSAGES-'
        msg = "The given file does not exist: {0}".format(filename)
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    if not ffmpeg_check():
        #print("ffmpeg: Executable not found on machine.")
        #raise Exception("Dependency not found: ffmpeg")
        not_transcribing=True
        window_key = '-OUTPUT-MESSAGES-'
        msg = 'ffmpeg: Executable not found on machine'
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    #command = ["ffmpeg", "-y", "-i", filename, "-ac", f'{str(n_channels)}', "-ar", f"{str(rate)}", "-loglevel", "error", temp.name]
    command = ["ffmpeg", "-y", "-i", filename, "-ac", str(n_channels), "-ar", str(rate), "-loglevel", "-1", temp.name]
    ff = FfmpegProgress(command)
    file_display_name = os.path.basename(filename).split('/')[-1]

    try:
        info = 'Converting {} to a temporary WAV file'.format(file_display_name)
        total = 100
        for progress in ff.run_command_with_progress():
            percentage = f'{progress}%'
            main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, percentage, progress))
        main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, "100%", 100))
        time.sleep(1)

    except Exception as e:
        not_transcribing = True
        msg = [e, n_channels, str(n_channels)]
        sg.Popup(msg, title="Info", line_width=50)
        main_window.write_event_value('-EXCEPTION-', e)

    if not_transcribing: return
    if sys.platform == "win32":
        subprocess.check_output(command, stdin=open(os.devnull), creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.check_output(command, stdin=open(os.devnull))

    return temp.name, rate


class NoConsoleProcess(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        '''
        if hasattr(multiprocessing, 'get_all_start_methods') and 'spawn' in multiprocessing.get_all_start_methods():
            # If running on Windows
            kwargs['start_method'] = 'spawn'
        '''
        super().__init__(*args, **kwargs)
        self.queue = multiprocessing.Queue()

    def run(self):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            super().run()
        except Exception as e:
            self.queue.put(('error', str(e)))
        else:
            self.queue.put(('done', None))


def vosk_transcribe(src, dst, video_filepath, subtitle_format):
    global all_transcribe_threads, thread_vosk_transcribe, not_transcribing, main_window, regions, transcriptions, \
        created_regions, created_subtitles, translated_transcriptions, saved_src_subtitle_filepath, saved_dst_subtitle_filepath, \
        subtitle_filepath, subtitle_file_extension

    if not os.path.isfile(video_filepath):
        FONT_TYPE = "Helvetica"
        FONT_SIZE = 9
        sg.set_options(font=(FONT_TYPE, FONT_SIZE))
        sg.Popup("{} is not exist!".format(video_filepath), title="Info", line_width=50)
        return

    if not_transcribing: return

    pool = multiprocessing.Pool(10, initializer=NoConsoleProcess)
    wav_filepath = None
    SampleRate = None
    subtitle_file = None
    translated_subtitle_file = None
    file_display_name = os.path.basename(video_filepath).split('/')[-1]
    file_display_name = os.path.basename(video_filepath).split('/')[-1]

    if not_transcribing: return

    main_window.write_event_value('-EVENT-TRANSCRIBE-PREPARE-WINDOW-', True)
    main_window.write_event_value('-EVENT-TRANSCRIBE-SHOW-PROGRESS-BAR-', False)

    window_key = '-OUTPUT-MESSAGES-'
    msg = "Processing {} :\n".format(file_display_name)
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    window_key = '-OUTPUT-MESSAGES-'
    msg = "Converting {} to a temporary WAV file...\n".format(file_display_name)
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))
    main_window.write_event_value('-EVENT-TRANSCRIBE-SHOW-PROGRESS-BAR-', True)

    info = 'Converting {} to a temporary WAV file'.format(file_display_name)
    total = 100
    main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, "0%", 0))

    wav_filepath, SampleRate = extract_audio(video_filepath)

    if not_transcribing: return

    regions = []
    transcriptions = []
    translated_transcriptions = []

    if not_transcribing: return

    if os.path.isfile(wav_filepath):

        window_key = '-OUTPUT-MESSAGES-'
        msg = 'Creating {} transcriptions...\n'.format(file_display_name)
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

        SetLogLevel(-1)
        SampleRate = 16000
        model = Model(lang = src)
        rec = KaldiRecognizer(model, SampleRate)
        rec.SetWords(True)
        block_size = 4096

        try:
            reader = wave.open(wav_filepath)
            rate = reader.getframerate()

        except Exception as e:
            not_transcribing = True
            sg.Popup(e, title="Info", line_width=50)

        total_duration = reader.getnframes() / rate

        info = 'Creating {} transcriptions'.format(file_display_name)
        total = 100
        main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, "0%", 0))

        while True:

            block = reader.readframes(block_size)

            if not block:
                break

            if rec.AcceptWaveform(block):

                recResult_json = json.loads(rec.Result())

                if 'result' in recResult_json:

                    result = recResult_json["result"]

                    text = recResult_json["text"]

                    transcriptions.append(text)

                    start_time = result[0]["start"]

                    progress = int(start_time*100/total_duration)
                    percentage = f'{progress}%'
                    main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, percentage, progress))

                    end_time = result[len(result)-1]["end"]

                    progress = int(end_time*100/total_duration)
                    percentage = f'{progress}%'
                    main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, percentage, progress))

                    regions.append((start_time, end_time))

        main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, "100%", total))

        regions_filename = "regions"
        regions_filepath = os.path.join(tempfile.gettempdir(), regions_filename)
        regions_file = open(regions_filepath, "w")
        json.dump(regions, regions_file)
        regions_file.close()
        
        transcriptions_filename = "transcriptions"
        transcriptions_filepath = os.path.join(tempfile.gettempdir(), transcriptions_filename)
        transcriptions_file = open(transcriptions_filepath, "w")
        json.dump(transcriptions, transcriptions_file)
        transcriptions_file.close()

        if not_transcribing: return

        timed_subtitles = [(r, t) for r, t in zip(regions, transcriptions) if t]
        formatter = FORMATTERS.get(subtitle_format)
        formatted_subtitles = formatter(timed_subtitles)

        base, ext = os.path.splitext(video_filepath)
        subtitle_file = "{base}.{format}".format(base=base, format=subtitle_format)

        if not_transcribing: return

        with open(subtitle_file, 'wb') as f:
            f.write(formatted_subtitles.encode("utf-8"))
            f.close()

        with open(subtitle_file, 'a') as f:
            f.write("\n")
            f.close()

        if not_transcribing: return

        if (not is_same_language(src, dst)):
            translated_subtitle_file = subtitle_file[ :-4] + '.translated.' + subtitle_format

            if not_transcribing: return

            created_regions = []
            created_subtitles = []
            for entry in timed_subtitles:
                created_regions.append(entry[0])
                created_subtitles.append(entry[1])

            if not_transcribing: return

            window_key = '-OUTPUT-MESSAGES-'
            msg = 'Translating {} subtitles from {} to {} ...\n'.format(file_display_name, src, dst)
            append_flag = True
            main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

            info = 'Translating {} subtitles from {} to {}'.format(file_display_name, src, dst)
            total = 100
            main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, "0%", 0))

            transcript_translator = SentenceTranslator(src=src, dst=dst)
            translated_transcriptions = []

            for i, translated_transcription in enumerate(pool.imap(transcript_translator, created_subtitles)):

                if not_transcribing:
                    pool.close()
                    pool.join()
                    pool = None
                    return

                translated_transcriptions.append(translated_transcription)

                progress = int(i*100/len(timed_subtitles))
                percentage = f'{progress}%'
                main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, percentage, progress))
            main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (info, "100%", total))

            if not_transcribing: return

            timed_translated_subtitles = [(r, t) for r, t in zip(created_regions, translated_transcriptions) if t]
            formatter = FORMATTERS.get(subtitle_format)
            formatted_translated_subtitles = formatter(timed_translated_subtitles)

            if not_transcribing: return

            with open(translated_subtitle_file, 'wb') as f:
                f.write(formatted_translated_subtitles.encode("utf-8"))
            with open(translated_subtitle_file, 'a') as f:
                f.write("\n")

            if not_transcribing: return

        window_key = '-OUTPUT-MESSAGES-'
        msg = "{} temporary subtitles file created at : {}\n".format(file_display_name, subtitle_file)
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

        if (not is_same_language(src, dst)):
            window_key = '-OUTPUT-MESSAGES-'
            msg = "{} temporary translated subtitles file created at : {}\n" .format(file_display_name, translated_subtitle_file)
            append_flag = True
            main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

        if not_transcribing: return

    window_key = '-OUTPUT-MESSAGES-'
    msg = 'Saving {} temporary transcriptions text files...\n'.format(file_display_name)
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    # SAVING TEMPORARY TRANSCRIPTIONS record.txt AND TRANSLATED TRANSCRIPTIONS record.traslated.txt
    tmp_src_txt_transcription_filename = "record.txt"
    tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
    tmp_dst_txt_transcription_filename = "record.translated.txt"
    tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)

    for i in range(len(created_regions)):
        start_time, end_time = regions[i]
        start_time = timedelta(seconds=start_time)
        end_time = timedelta(seconds=end_time)
        start_time_str = "{:02d}:{:02d}:{:02.0f}.{:03.0f}".format(
            start_time.seconds // 3600,  # hours
            (start_time.seconds // 60) % 60,  # minutes
            start_time.seconds % 60, # seconds
            start_time.microseconds / 1000000  # microseconds
        )

        end_time_str = "{:02d}:{:02d}:{:02.0f}.{:03.0f}".format(
            end_time.seconds // 3600,  # hours
            (end_time.seconds // 60) % 60,  # minutes
            end_time.seconds % 60, # seconds
            end_time.microseconds / 1000000  # microseconds
        )

        time_stamp_string = start_time_str + "-" + end_time_str

        tmp_src_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "a")
        tmp_src_txt_transcription_file.write(time_stamp_string + " " + created_subtitles[i] + "\n")
        tmp_src_txt_transcription_file.close()

        if (not is_same_language(src, dst)):
            tmp_dst_txt_transcription_file = open(tmp_dst_txt_transcription_filepath, "a")
            tmp_dst_txt_transcription_file.write(time_stamp_string + " " + translated_transcriptions[i] + "\n")
            tmp_dst_txt_transcription_file.close()

    window_key = '-OUTPUT-MESSAGES-'
    msg = "{} temporary transcriptions text file created at : {}\n".format(file_display_name, tmp_src_txt_transcription_filepath)
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    if (not is_same_language(src, dst)):
        window_key = '-OUTPUT-MESSAGES-'
        msg = "{} temporary translated transcriptions text file created at : {}\n".format(file_display_name, tmp_dst_txt_transcription_filepath)
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    window_key = '-OUTPUT-MESSAGES-'
    msg = "Done.\n".format(file_display_name, tmp_src_txt_transcription_filepath)
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    time.sleep(2)
    main_window.write_event_value('-EVENT-TRANSCRIBE-CLOSE-', True)

    transcribe_is_done_filename = "transcribe_is_done"
    transcribe_is_done_filepath = os.path.join(tempfile.gettempdir(), transcribe_is_done_filename)
    transcribe_is_done_file = open(transcribe_is_done_filepath, "w")
    transcribe_is_done_file.write("True")
    transcribe_is_done_file.close()

    if len(all_transcribe_threads) > 0:
        for t in all_transcribe_threads:
            if t.is_alive():
                stop_thread(t)

    pool.close()
    pool.join()
    pool = None
    remove_temp_files("flac")
    remove_temp_files("wav")
    not_transcribing = True


def save_temporary_transcriptions(src, dst, subtitle_format):
    global thread_save_temporary_transcriptions, main_window, regions, transcriptions, \
        created_regions, created_subtitles, translated_transcriptions

    tmp_src_subtitle_filename = "record" + "." + subtitle_format
    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)

    # SAVING TRANSCRIPTIONS IN A TEMPORARY SUBTITLE FILE
    timed_subtitles = [(r, t) for r, t in zip(regions, transcriptions) if t]
    formatter = FORMATTERS.get(subtitle_format)
    formatted_subtitles = formatter(timed_subtitles)

    tmp_src_subtitle_file = open(tmp_src_subtitle_filepath, 'wb')
    tmp_src_subtitle_file.write(formatted_subtitles.encode("utf-8"))
    tmp_src_subtitle_file.close()

    # SAVING TEMPORARY TRANSLATED TRANSCRIPTIONS AS FORMATTED SUBTITLE FILE
    created_regions = []
    created_subtitles = []
    for entry in timed_subtitles:
        created_regions.append(entry[0])
        created_subtitles.append(entry[1])

    translated_transcriptions = []
    i = 0
    if len(created_subtitles)>0:
        info = 'Saving temporary translated transcriptions'
        total = 100
        main_window.write_event_value('-EVENT-PROGRESS-BAR-', (info, total, "0%", 0))
        transcript_translator = SentenceTranslator(src=src, dst=dst)
        translated_transcriptions = []
        pool = multiprocessing.Pool(10, initializer=NoConsoleProcess)
        for i, translated_transcription in enumerate(pool.imap(transcript_translator, created_subtitles)):
            translated_transcriptions.append(translated_transcription)
            progress = int(i*100/len(created_subtitles))
            percentage = f'{progress}%'
            main_window.write_event_value('-EVENT-PROGRESS-BAR-', (info, total, percentage, progress))
        main_window.write_event_value('-EVENT-PROGRESS-BAR-', (info, total, "100%", 100))
        pool.close()
        pool.join()
        pool = None

        timed_translated_subtitles = [(r, t) for r, t in zip(created_regions, translated_transcriptions) if t]
        formatter = FORMATTERS.get(subtitle_format)
        formatted_translated_subtitles = formatter(timed_translated_subtitles)

        tmp_dst_subtitle_filepath = tmp_src_subtitle_filepath[ :-4] + ".translated." + subtitle_format

        with open(tmp_dst_subtitle_filepath, 'wb') as f:
            f.write(formatted_translated_subtitles.encode("utf-8"))
            f.close()

        with open(tmp_dst_subtitle_filepath, 'a') as f:
            f.write("\n")
            f.close()

   # SAVING TEMPORARY TRANSLATED TRANSCRIPTION AS TEXT FILE
    tmp_dst_txt_transcription_filename = "record.translated.txt"
    tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)

    with open(tmp_dst_txt_transcription_filepath, 'w') as f:
        for i in range(len(translated_transcriptions)):

            start_time_delta = timedelta(seconds=created_regions[i][0])
            start_time_str = "{:02d}:{:02d}:{:02.0f}.{:03.0f}".format(
                start_time_delta.seconds // 3600,  # hours
                (start_time_delta.seconds // 60) % 60,  # minutes
                start_time_delta.seconds % 60, # seconds
                start_time_delta.microseconds / 1000000  # microseconds
            )

            end_time_delta = timedelta(seconds=created_regions[i][1])
            end_time_str = "{:02d}:{:02d}:{:02.0f}.{:03.0f}".format(
                end_time_delta.seconds // 3600,  # hours
                (end_time_delta.seconds // 60) % 60,  # minutes
                end_time_delta.seconds % 60, # seconds
                end_time_delta.microseconds / 1000000  # microseconds
            )

            time_stamp_string = start_time_str + "-" + end_time_str
            f.write(time_stamp_string + " " + translated_transcriptions[i] + "\n")

        f.close()


def save_as_declared_subtitle_filename(declared_subtitle_filename, src, dst):

    subtitle_file_base, subtitle_file_extension = os.path.splitext(declared_subtitle_filename)
    subtitle_file_extension = subtitle_file_extension[1:]

    regions_filename = "regions"
    regions_filepath = os.path.join(tempfile.gettempdir(), regions_filename)
    transcriptions_filename = "transcriptions"
    transcriptions_filepath = os.path.join(tempfile.gettempdir(), transcriptions_filename)
    regions = None
    transcriptions = None

    transcribe_is_done_filename = "transcribe_is_done"
    transcribe_is_done_filepath = os.path.join(tempfile.gettempdir(), transcribe_is_done_filename)
    #transcribe_is_done_file = open(transcribe_is_done_filepath, "r")
    #transcribe_is_done = transcribe_is_done_file.read()
    #transcribe_is_done_file.close()

    while True:
        if os.path.isfile(transcribe_is_done_filepath):
            if os.path.isfile(regions_filepath):
                regions_file = open(regions_filepath, "r")
                regions = tuple(json.load(regions_file))
                regions_file.close()

            if os.path.isfile(transcriptions_filepath):
                transcriptions_file = open(transcriptions_filepath, "r")
                transcriptions = tuple(json.load(transcriptions_file))
                transcriptions_file.close()

            if subtitle_file_extension and subtitle_file_extension in FORMATTERS.keys()and len(regions)>0 and len(transcriptions)>0:
                subtitle_filepath = "{base}.{format}".format(base=subtitle_file_base, format=subtitle_file_extension)
                timed_subtitles = [(r, t) for r, t in zip(regions, transcriptions) if t]
                subtitle_format = subtitle_file_extension
                formatter = FORMATTERS.get(subtitle_format)
                formatted_subtitles = formatter(timed_subtitles)
                subtitle_file = open(subtitle_filepath, 'wb')
                subtitle_file.write(formatted_subtitles.encode("utf-8"))
                subtitle_file.close()

                created_regions = []
                created_subtitles = []
                for entry in timed_subtitles:
                    created_regions.append(entry[0])
                    created_subtitles.append(entry[1])

            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("Subtitle format you typed is not supported, subtitle file will be saved in TXT format.", title="Info", line_width=50)
                subtitle_filepath = "{base}.{format}".format(base=subtitle_file_base, format="txt")
                shutil.copy(tmp_src_txt_transcription_filepath, subtitle_filepath)

            # SAVING TRANSLATED SUBTITLE FILE WITH FILENAME BASED ON SUBTITLE FILENAME DECLARED IN COMMAND LINE ARGUMENTS
            if not is_same_language(src, dst):

                saved_dst_subtitle_filepath = subtitle_filepath[ :-4] + ".translated." + subtitle_format

                if subtitle_file_extension and subtitle_file_extension in FORMATTERS.keys() and len(created_regions)>0 \
                        and len(translated_transcriptions)>0:

                    timed_translated_subtitles = [(r, t) for r, t in zip(created_regions, translated_transcriptions) if t]
                    subtitle_format = subtitle_file_extension
                    formatter = FORMATTERS.get(subtitle_format)
                    formatted_translated_subtitles = formatter(timed_translated_subtitles)
                    saved_dst_subtitle_file = open(saved_dst_subtitle_filepath, 'wb')
                    saved_dst_subtitle_file.write(formatted_translated_subtitles.encode("utf-8"))
                    saved_dst_subtitle_file.close()

                else:
                    shutil.copy(tmp_dst_txt_transcription_filepath, saved_dst_subtitle_filepath)

            break


def scroll_to_last_line(window, element):
    if isinstance(element.Widget, tk.Text):
        # Get the number of lines in the Text element
        num_lines = element.Widget.index('end-1c').split('.')[0]

        # Scroll to the last line
        element.Widget.see(f"{num_lines}.0")

    elif isinstance(element.Widget, tk.Label):
        # Scroll the Label text
        element.Widget.configure(text=element.get())
    
    # Update the window to show the scroll position
    window.refresh()


def remove_temp_files(extension):
    temp_dir = tempfile.gettempdir()
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith("." + extension):
                os.remove(os.path.join(root, file))


#----------------------------------------------------------- GUI FUNCTIONS ------------------------------------------------------------#

def handle_focus(event):
    if event.widget == window:
        window.focus_set()


def steal_focus():
    global main_window, overlay_translation_window

    #print("debug")
    if overlay_translation_window:
        overlay_translation_window.close()
        overlay_translation_window=None
    if(sys.platform == "win32"):
        main_window.TKroot.attributes('-topmost', True)
        main_window.TKroot.attributes('-topmost', False)
        main_window.TKroot.deiconify()
    if(sys.platform == "linux"):
        #main_window.TKroot.attributes('-topmost', 1)
        #main_window.TKroot.attributes('-topmost', 0)
        #main_window.TKroot.deiconify()
        #main_window.bring_to_front()
        main_window.BringToFront()


def font_length(Text, Font, Size) :
    f = tf.Font(family=Font , size = Size)
    length = f.measure(Text)
    #print(length)
    return length


def popup_yes_no(text, title=None):
    layout = [
        [sg.Text(text)],
        [sg.Push(), sg.Button('Yes'), sg.Button('No')],
    ]
    return sg.Window(title if title else text, layout, resizable=True).read(close=True)


def make_progress_bar_window(info, total):

    FONT_TYPE = "Helvetica"
    FONT_SIZE = 9
    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    layout = [
                [sg.Text(info, key='-INFO-')],
                [sg.ProgressBar(total, orientation='h', size=(30, 10), key='-PROGRESS-'), sg.Text(size=(5,1), key='-PERCENTAGE-')]
             ]

    progress_bar_window = sg.Window("Progress", layout, no_titlebar=False, finalize=True)
    progress_bar_window['-PROGRESS-'].update(max=total)
    move_center(progress_bar_window)

    return progress_bar_window


def make_transcribe_window(info, total):
    global not_transcribing

    FONT_TYPE = "Helvetica"
    FONT_SIZE = 9
    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    layout = [
                [sg.Text("Progress info", key='-INFO-')],
                [
                    sg.ProgressBar(100, orientation='h', size=(50, 10), key='-PROGRESS-'),
                    sg.Text("0%", size=(5,1), key='-PERCENTAGE-')
                ],
                [sg.Multiline(size=(60, 10), expand_x=True, expand_y=True, key='-OUTPUT-MESSAGES-')],
                [sg.Button('Cancel', font=("Helvetica", 9), expand_x=True, expand_y=True, key='-CANCEL-')]
             ]

    transcribe_window = sg.Window('Transcribe', layout, font=FONT_TYPE, resizable=True, keep_on_top=True, finalize=True)
    transcribe_window['-PROGRESS-'].update(max=total)
    move_center(transcribe_window)

    return transcribe_window


def make_overlay_translation_window(translated_text):
    xmax, ymax = sg.Window.get_screen_size()

    '''
    max_columns = None
    max_lines = None

    if sys.platform == "win32":
        SM_CXSCREEN = 0
        SM_CYSCREEN = 1
        width = ctypes.windll.user32.GetSystemMetrics(SM_CXSCREEN)
        height = ctypes.windll.user32.GetSystemMetrics(SM_CYSCREEN)

        max_columns = int(width/8.5)
        max_lines = int(height/18)

    else:
        try:
            output = subprocess.check_output(['xrandr'], check=True)
            primary_line = [line for line in output.decode().splitlines() if ' connected primary' in line][0]
            primary_size = primary_line.split()[3].replace("+"," ").split()[0]
            max_columns, max_lines = int(primary_size.split('x')[0]) // 8, int(primary_size.split('x')[1]) // 18

        except subprocess.CalledProcessError:
            max_columns = int(os.environ.get('COLUMNS', 80))
            max_lines = int(os.environ.get('LINES', 24))
    '''

    max_columns = int(os.environ.get('COLUMNS', 80))
    max_lines = int(os.environ.get('LINES', 24))

    columns = max_columns
    rows = max_lines

    FONT_TYPE = 'Helvetica'
    FONT_SIZE = 16

    if len(translated_text)<=96:
        window_width = int(9.9*(xmax/1280)*len(translated_text) + 60)
        #window_width = font_length(translated_text, FONT_TYPE, FONT_SIZE)
    else:
        window_width=int(960*xmax/1280)

    nl = int((len(translated_text)/96) + 1)

    if nl == 1:
        LINE_SIZE = 32
    else:
        LINE_SIZE = 28*ymax/720

    window_height = int(nl*LINE_SIZE)
    window_x=int((xmax-window_width)/2)

    if nl == 1:
        window_y = int(528*ymax/720)
    else:
        window_y = int((528-(nl-1)*28)*ymax/720)

    if xmax > 1280: FONT_SIZE = 14
    if xmax <= 1280: FONT_SIZE = 16

    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    multiline_width = len(translated_text)
    multiline_height = nl

    TRANSPARENT_BLACK='#add123'

    if not (sys.platform == "win32"):

        layout = [
                    [
                        sg.Multiline(default_text=translated_text, size=(multiline_width, multiline_height), text_color='yellow1',
                            border_width=0, background_color='black', no_scrollbar=True, justification='l',
                            expand_x=True, expand_y=True, key='-ML-OVERLAY-DST-PARTIAL-RESULTS-')
                    ]
                 ]

        overlay_translation_window = sg.Window('Translation', layout, no_titlebar=True, keep_on_top=True,
            size=(window_width,window_height), location=(window_x,window_y), margins=(0, 0), background_color='black',
            alpha_channel=0.7, return_keyboard_events=True, finalize=True)

        overlay_translation_window.TKroot.attributes('-type', 'splash')
        overlay_translation_window.TKroot.attributes('-topmost', 1)
        #overlay_translation_window.TKroot.attributes('-topmost', 0)

    else:

        layout = [
                    [
                        sg.Multiline(default_text=translated_text, size=(multiline_width, multiline_height), text_color='yellow1',
                            border_width=0, background_color='white', no_scrollbar=True, justification='l', 
                            expand_x=True, expand_y=True, key='-ML-OVERLAY-DST-PARTIAL-RESULTS-')
                    ]
                 ]

        overlay_translation_window = sg.Window('Translation', layout, no_titlebar=True, keep_on_top=True, 
            size=(window_width,window_height), location=(window_x,window_y), margins=(0, 0), background_color='white',
            transparent_color='white', grab_anywhere=True, return_keyboard_events=True, finalize=True)

    event, values = overlay_translation_window.read(500)

    return overlay_translation_window


def move_center(window):
    screen_width, screen_height = window.get_screen_dimensions()
    win_width, win_height = window.size
    x, y = (screen_width-win_width)//2, (screen_height-win_height)//2 - 30
    window.move(x, y)


def get_clipboard_text():
    return subprocess.check_output(['xclip', '-selection', 'clipboard', '-out']).decode()


def set_clipboard_text(text):
    subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode())


#------------------------------------------------------------ MAIN PROGRAM ------------------------------------------------------------#


def main():
    global main_window, thread_recognize, thread_timed_translate, thread_record_streaming, text, recognizing, \
        Device, SampleRate, wav_filepath, tmp_recorded_streaming_filepath, regions, transcriptions, \
        created_regions, created_subtitles, translated_transcriptions, start_button_click_time, ffmpeg_start_run_time, \
        get_tmp_recorded_streaming_filepath_time, first_streaming_duration_recorded, is_valid_url_streaming, \
        transcribe_window, thread_vosk_transcribe, all_transcribe_threads, not_transcribing, \
        saved_src_subtitle_filepath, saved_dst_subtitle_filepath, subtitle_filepath, subtitle_file_extension

    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalLock.argtypes = [ctypes.wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.wintypes.LPVOID
        kernel32.GlobalSize.argtypes = [ctypes.wintypes.HGLOBAL]
        kernel32.GlobalSize.restype = ctypes.c_size_t
        user32.OpenClipboard.argtypes = [ctypes.wintypes.HWND]
        user32.OpenClipboard.restype = ctypes.wintypes.BOOL
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = ctypes.wintypes.BOOL
        user32.GetClipboardData.argtypes = [ctypes.wintypes.UINT]
        user32.GetClipboardData.restype = ctypes.wintypes.HANDLE


#------------------------------------------------------------- GUI PARTS -------------------------------------------------------------#

    xmax, ymax = sg.Window.get_screen_size()
    main_window_width = int(0.5*xmax)
    main_window_height = int(0.5*ymax)
    multiline_width = int(0.15*main_window_width)
    multiline_height = int(0.0125*main_window_height)
    FONT_TYPE = "Helvetica"
    FONT_SIZE = 9
    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    layout =  [
                [sg.Frame('Hints',[
                                    [sg.Text('Click \"Start\" button to start listening and printing subtitles on screen.', expand_x=True, expand_y=True)],
                                    [sg.Text('Paste the streaming URL into URL inputbox then check the \"Record Streaming\" checkbox to record the streaming.', expand_x=True, expand_y=True)],
                                    #[sg.Text('If you record the streaming, when you save transcriptions, all timestamps will be relative to the \"Start\" button clicked time.', expand_x=True, expand_y=True)],
                                    #[sg.Text('If you don\'t record the streaming, all timestamps will be based on your system clock.', expand_x=True, expand_y=True)],
                                ], border_width=2, expand_x=True, expand_y=True)],
                [
                    sg.Text('URL', size=(12, 1)),
                    sg.Input(size=(20, 1), expand_x=True, expand_y=True, key='-URL-', enable_events=True, right_click_menu=['&Edit', ['&Paste',]]),
                    sg.Checkbox("Record Streaming", key='-RECORD-STREAMING-', enable_events=True)
                ],
                [
                    #sg.Text('', size=(3, 1)),
                    sg.Text('Thread status', size=(12, 1)),
                    sg.Text('NOT RECORDING', size=(24, 1), background_color='green1', text_color='black', expand_x=True, expand_y=True, key='-RECORD-STREAMING-STATUS-'),
                    sg.Text('Duration recorded', size=(16, 1)),
                    sg.Text('0:00:00.000000', size=(9, 1), background_color='green1', text_color='black', expand_x=True, expand_y=True, key='-STREAMING-DURATION-RECORDED-'),
                    sg.Text('', size=(12, 1), expand_x=True, expand_y=True)
                ],
                [sg.Text('', expand_x=True, expand_y=True),
                 sg.Button('SAVE RECORDED STREAMING', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-RECORDED-STREAMING-'),
                 sg.Text('', expand_x=True, expand_y=True)],
                [sg.Text('Audio language', size=(12, 1), expand_x=True, expand_y=True),
                 sg.Combo(list(map_src_of_language), size=(int(0.2*multiline_width), 1), default_value="English", expand_x=True, expand_y=True, key='-SRC-')],
                [sg.Multiline(size=(multiline_width, multiline_height), expand_x=True, expand_y=True, key='-ML-SRC-RESULTS-')],
                [sg.Multiline(size=(multiline_width, 0.5*multiline_height), expand_x=True, expand_y=True, key='-ML-SRC-PARTIAL-RESULTS-')],
                [sg.Text('', expand_x=True, expand_y=True),
                 sg.Button('SAVE TRANSCRIPTION', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-SRC-TRANSCRIPTION-'),
                 sg.Text('', expand_x=True, expand_y=True)],
                [sg.Text('Translation language', size=(12, 1),expand_x=True, expand_y=True),
                 sg.Combo(list(map_dst_of_language), size=(int(0.2*multiline_width), 1), default_value="Indonesian", expand_x=True, expand_y=True, key='-DST-')],
                [sg.Multiline(size=(multiline_width, multiline_height), expand_x=True, expand_y=True, key='-ML-DST-RESULTS-')],
                [sg.Multiline(size=(multiline_width, 0.5*multiline_height), expand_x=True, expand_y=True, key='-ML-DST-PARTIAL-RESULTS-')],
                [sg.Text('', expand_x=True, expand_y=True, key='-SPACES3-'),
                 sg.Button('SAVE TRANSLATED TRANSCRIPTION', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-DST-TRANSCRIPTION-'),
                 sg.Text('', expand_x=True, expand_y=True, key='-SPACES4-')],
                [sg.Button('Start', expand_x=True, expand_y=True, button_color=('white', '#283b5b'), key='-START-BUTTON-'),
                 sg.Button('Exit', expand_x=True, expand_y=True)],
            ]


#--------------------------------------------------------- NON GUI PARTS -------------------------------------------------------------#

    # VOSK LogLevel
    SetLogLevel(-1)

    if sys.platform == "win32":
        stop_ffmpeg_windows()
    else:
        stop_ffmpeg_linux()

    last_selected_src = None
    last_selected_dst = None

    src_filename = "src"
    src_filepath = os.path.join(tempfile.gettempdir(), src_filename)
    if os.path.isfile(src_filepath):
        src_file = open(src_filepath, "r")
        last_selected_src = src_file.read()

    dst_filename = "dst"
    dst_filepath = os.path.join(tempfile.gettempdir(), dst_filename)
    if os.path.isfile(dst_filepath):
        dst_file = open(dst_filepath, "r")
        last_selected_dst = dst_file.read()

    recognizing = False
    not_transcribing = True

    tmp_recorded_streaming_filename = "record.mp4"
    tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)
    if os.path.isfile(tmp_recorded_streaming_filepath):
        os.remove(tmp_recorded_streaming_filepath)

    saved_recorded_streaming_filename = None

    partial_result_filename = "partial_result"
    partial_result_filepath = os.path.join(tempfile.gettempdir(), partial_result_filename)
    if os.path.isfile(partial_result_filepath): os.remove(partial_result_filepath)

    time_value_filename = "time_value"
    time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
    if os.path.isfile(time_value_filepath): os.remove(time_value_filepath)

    subtitle_file_types = [
        ('SRT Files', '*.srt'),
        ('VTT Files', '*.vtt'),
        ('JSON Files', '*.json'),
        ('RAW Files', '*.raw'),
        ('Text Files', '*.txt'),
        ('All Files', '*.*'),
    ]

    FORMATTERS = {
        'srt': srt_formatter,
        'vtt': vtt_formatter,
        'json': json_formatter,
        'raw': raw_formatter,
    }

    video_file_types = [
        ('MP4 Files', '*.mp4'),
    ]

    tmp_src_txt_transcription_filename = "record.txt"
    tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
    if os.path.isfile(tmp_src_txt_transcription_filepath): os.remove(tmp_src_txt_transcription_filepath)

    tmp_dst_txt_transcription_filename = "record.translated.txt"
    tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)
    if os.path.isfile(tmp_dst_txt_transcription_filepath): os.remove(tmp_dst_txt_transcription_filepath)

    tmp_src_subtitle_filename = "record.srt"
    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)
    if os.path.isfile(tmp_src_subtitle_filepath): os.remove(tmp_src_subtitle_filepath)

    tmp_dst_subtitle_filename = "record.translated.srt"
    tmp_dst_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_subtitle_filename)
    if os.path.isfile(tmp_dst_subtitle_filepath): os.remove(tmp_dst_subtitle_filepath)

    regions_filename = "regions"
    regions_filepath = os.path.join(tempfile.gettempdir(), regions_filename)
    if os.path.isfile(regions_filepath): os.remove(regions_filepath)

    transcriptions_filename = "transcriptions"
    transcriptions_filepath = os.path.join(tempfile.gettempdir(), transcriptions_filename)
    if os.path.isfile(transcriptions_filepath): os.remove(transcriptions_filepath)

    transcribe_is_done_filename = "transcribe_is_done"
    transcribe_is_done_filepath = os.path.join(tempfile.gettempdir(), transcribe_is_done_filename)
    if os.path.isfile(transcribe_is_done_filepath): os.remove(transcribe_is_done_filepath)

    saved_src_subtitle_filename = None
    saved_dst_subtitle_filename = None

    ffmpeg_start_run_time = None

    parser = argparse.ArgumentParser(add_help=False)

    if last_selected_src:
        parser.add_argument('-S', '--src-language', help="Spoken language", default=last_selected_src)
    else:
        parser.add_argument('-S', '--src-language', help="Spoken language", default="en")

    if last_selected_dst:
        parser.add_argument('-D', '--dst-language', help="Desired language for translation", default=last_selected_dst)
    else:
        parser.add_argument('-D', '--dst-language', help="Desired language for translation", default="id")

    parser.add_argument('-lls', '--list-languages-src', help="List all available source languages", action='store_true')
    parser.add_argument('-lld', '--list-languages-dst', help="List all available destination languages", action='store_true')

    parser.add_argument('-sf', '--subtitle-filename', help="Subtitle file name for saved transcriptions")
    parser.add_argument('-F', '--subtitle-format', help="Desired subtitle format for saved transcriptions (default is \"srt\")", default="srt")
    parser.add_argument('-lsf', '--list-subtitle-formats', help="List all available subtitle formats", action='store_true')

    parser.add_argument("-ld", "--list-devices", action="store_true", help="show list of audio devices and exit")

    args, remaining = parser.parse_known_args()

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])

    parser.add_argument("-af", "--audio-filename", type=str, metavar="AUDIO_FILENAME", help="audio file to store recording to", default=None)
    parser.add_argument("-d", "--device", type=int_or_str, help="input device (numeric ID or substring)")
    parser.add_argument("-r", "--samplerate", type=int, help="sampling rate in Hertz for example 8000, 16000, 44100, or 48000", default=16000)

    parser.add_argument("-u", "--url", type=str, metavar="URL", help="URL of live streaming if you want to record the streaming")
    parser.add_argument("-vf", "--video-filename", type=str, metavar="VIDEO_FILENAME", help="video file to store recording to", default=None)

    parser.add_argument('-v', '--version', action='version', version='0.1.5')

    args = parser.parse_args(remaining)
    args = parser.parse_args()

    if args.src_language:
        src = args.src_language
        last_selected_src = src
        src_file = open(src_filepath, "w")
        src_file.write(src)
        src_file.close()

    if args.dst_language:
        dst = args.dst_language
        last_selected_dst = dst
        dst_file = open(dst_filepath, "w")
        dst_file.write(dst)
        dst_file.close()

    if args.src_language not in map_language_of_src.keys():
        print("Audio language not supported. Run with --list-languages-src to see all supported source languages.")
        parser.exit(0)

    if args.dst_language not in map_language_of_dst.keys():
        print("Destination language not supported. Run with --list-languages-dst to see all supported destination languages.")
        parser.exit(0)

    if args.list_languages_src:
        print("List of all source languages:")
        for code, language in sorted(map_language_of_src.items()):
            print("{code}\t{language}".format(code=code, language=language))
        parser.exit(0)

    if args.list_languages_dst:
        print("List of all destination languages:")
        for code, language in sorted(map_language_of_dst.items()):
            print("{code}\t{language}".format(code=code, language=language))
        parser.exit(0)

    if args.device:
        Device = args.device

    if args.samplerate:
        SampleRate = args.samplerate

    if args.audio_filename:
        wav_filepath = args.audio_filename
        dump_fn = open(args.audio_filename, "wb")
    else:
        dump_fn = None

    if args.subtitle_format not in FORMATTERS.keys():
        FONT_TYPE = "Helvetica"
        FONT_SIZE = 9
        sg.set_options(font=(FONT_TYPE, FONT_SIZE))
        sg.Popup("Subtitle format is not supported, you can choose it later when you save transcriptions.", title="Info", line_width=50)

    subtitle_format = args.subtitle_format

    if args.video_filename:
        saved_recorded_streaming_basename, saved_recorded_streaming_extension = os.path.splitext(os.path.basename(args.video_filename))
        saved_recorded_streaming_extension = saved_recorded_streaming_extension[1:]
        if saved_recorded_streaming_extension != "mp4":
            saved_recorded_streaming_filename = "{base}.{format}".format(base=saved_recorded_streaming_basename, format="mp4")
        else:
            saved_recorded_streaming_filename = args.video_filename


#------------------------------------ REMAINING GUI PARTS NEEDED TO LOAD AT LAST BEFORE MAIN LOOP ------------------------------------#


    main_window = sg.Window('VOSK Live Subtitles', layout, resizable=True, keep_on_top=True, finalize=True)
    main_window['-SRC-'].block_focus()

    if (sys.platform == "win32"):
        if main_window.TKroot is not None:
            main_window.TKroot.attributes('-topmost', True)
            main_window.TKroot.attributes('-topmost', False)

    if not (sys.platform == "win32"):
        if main_window.TKroot is not None:
            main_window.TKroot.attributes('-topmost', 1)
            main_window.TKroot.attributes('-topmost', 0)

    overlay_translation_window = None
    move_center(main_window)

    FONT_TYPE = "Helvetica"
    FONT_SIZE = 9
    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    #transcribe_window = make_transcribe_window("", 100)
    #move_center(transcribe_window)
    #transcribe_window.Hide()
    transcribe_window = None

    progressbar = make_progress_bar_window("", 100)
    progressbar.Hide()

    main_window['-URL-'].update("")
    main_window['-RECORD-STREAMING-'].update(False)

    if args.src_language:
        combo_src = map_language_of_src[args.src_language]
        main_window['-SRC-'].update(combo_src)

    if args.dst_language:
        combo_dst = map_language_of_dst[args.dst_language]
        main_window['-DST-'].update(combo_dst)

    if args.url:
        is_valid_url_streaming = is_streaming_url(args.url)
        #print("is_valid_url_streaming = {}".format(is_valid_url_streaming))
        if is_valid_url_streaming:
            url = args.url
            main_window['-URL-'].update(url)
            main_window['-RECORD-STREAMING-'].update(True)

    main_window['-ML-SRC-PARTIAL-RESULTS-'].update("")
    main_window['-ML-SRC-RESULTS-'].update("")
    main_window['-ML-DST-PARTIAL-RESULTS-'].update("")
    main_window['-ML-DST-RESULTS-'].update("")

    thread_record_streaming = None
    thread_recognize = None
    thread_timed_translate = None
    thread_vosk_transcribe = None
    all_transcribe_threads = []
    thread_save_temporary_transcriptions = None
    thread_save_declared_subtitle_filename = None

    regions = None
    transcriptions = None
    created_regions = None
    created_subtitles = None
    translated_transcriptions = None


#-------------------------------------------------------------- MAIN LOOP -------------------------------------------------------------#


    while True:
        window, event, values = sg.read_all_windows()

        src = map_src_of_language[str(main_window['-SRC-'].get())]
        last_selected_src = src
        src_file = open(src_filepath, "w")
        src_file.write(src)
        src_file.close()

        dst = map_dst_of_language[str(main_window['-DST-'].get())]
        last_selected_dst = dst
        dst_file = open(dst_filepath, "w")
        dst_file.write(dst)
        dst_file.close()


        if event == 'Exit' or event == sg.WIN_CLOSED:

            if transcribe_window is not None and window == transcribe_window:

                if not not_transcribing:
                    FONT_TYPE = "Helvetica"
                    FONT_SIZE = 9
                    sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                    answer = popup_yes_no('             Are you sure?              ', title='Confirm')
                    if 'Yes' in answer:
                        not_transcribing = True
                        if thread_vosk_transcribe and thread_vosk_transcribe.is_alive(): stop_thread(thread_vosk_transcribe)
                        transcribe_window.close()
                        main_window.Normal()
                else:
                    not_transcribing = True
                    if thread_vosk_transcribe and thread_vosk_transcribe.is_alive(): stop_thread(thread_vosk_transcribe)
                    transcribe_window.close()
                    main_window.Normal()

            elif window == main_window:

                if recognizing:
                    recognizing = False
                    not_transcribing = True

                    if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:
                        if sys.platform == "win32":
                            stop_record_streaming_windows()
                        elif sys.platform == "linux":
                            stop_record_streaming_linux()

                    text = ""
                    translated_text = ""

                    if overlay_translation_window:
                        overlay_translation_window.close()
                        overlay_translation_window = None

                break


        elif event.endswith('+R') and values['-URL-']:

            if sys.platform == "win32":
                user32.OpenClipboard(None)
                clipboard_data = user32.GetClipboardData(1)  # 1 is CF_TEXT
                if clipboard_data:
                    data = ctypes.c_char_p(clipboard_data)
                    main_window['-URL-'].update(data.value.decode('utf-8'))
                user32.CloseClipboard()
            elif sys.platform == "linux":
                text = get_clipboard_text()
                if text:
                    main_window['-URL-'].update(text.strip())


        elif event == 'Paste':

            if sys.platform == "win32":
                user32.OpenClipboard(None)
                clipboard_data = user32.GetClipboardData(1)  # 1 is CF_TEXT
                if clipboard_data:
                    data = ctypes.c_char_p(clipboard_data)
                    main_window['-URL-'].update(data.value.decode('utf-8'))
                user32.CloseClipboard()
            elif sys.platform == "linux":
                text = get_clipboard_text()
                if text:
                    main_window['-URL-'].update(text.strip())


        elif event == '-RECORD-STREAMING-':

            if values['-RECORD-STREAMING-']:
                is_valid_url_streaming = is_streaming_url(str(values['-URL-']).strip())
                if not is_valid_url_streaming:
                    FONT_TYPE = "Helvetica"
                    FONT_SIZE = 9
                    sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                    sg.Popup('Invalid URL, please enter a valid URL.                   ', title="Info", line_width=50)
                    main_window['-RECORD-STREAMING-'].update(False)


        elif event == '-START-BUTTON-':
            recognizing = not recognizing
            #print("Start button clicked, changing recognizing status")
            #print("recognizing = {}".format(recognizing))

            main_window['-START-BUTTON-'].update(('Stop','Start')[not recognizing], button_color=(('white', ('red', '#283b5b')[not recognizing])))

            if recognizing:
                #print("VOSK Live Subtitle is START LISTENING now")
                #print("recognizing = {}".format(recognizing))

                start_button_click_time = datetime.now()
                #print("start_button_click_time = {}".format(start_button_click_time))

                #if tmp_recorded_streaming_filepath and os.path.isfile(tmp_recorded_streaming_filepath): os.remove(tmp_recorded_streaming_filepath)
                if partial_result_filepath and os.path.isfile(partial_result_filepath): os.remove(partial_result_filepath)
                if time_value_filepath and os.path.isfile(time_value_filepath): os.remove(time_value_filepath)
                if tmp_src_subtitle_filepath and os.path.isfile(tmp_src_subtitle_filepath): os.remove(tmp_src_subtitle_filepath)
                if tmp_dst_subtitle_filepath and os.path.isfile(tmp_dst_subtitle_filepath): os.remove(tmp_dst_subtitle_filepath)
                if tmp_src_txt_transcription_filepath and os.path.isfile(tmp_src_txt_transcription_filepath): os.remove(tmp_src_txt_transcription_filepath)
                if tmp_dst_txt_transcription_filepath and os.path.isfile(tmp_dst_txt_transcription_filepath): os.remove(tmp_dst_txt_transcription_filepath)

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:

                    if is_valid_url_streaming:

                        url = values['-URL-']

                        tmp_recorded_streaming_filename = "record.mp4"
                        tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)

                        get_tmp_recorded_streaming_filepath_time = datetime.now()
                        #print("get_tmp_recorded_streaming_filepath_time = {}".format(get_tmp_recorded_streaming_filepath_time))

                        #NEEDED FOR streamlink MODULE WHEN RUN AS PYINSTALLER COMPILED BINARY
                        os.environ['STREAMLINK_DIR'] = './streamlink/'
                        os.environ['STREAMLINK_PLUGINS'] = './streamlink/plugins/'
                        os.environ['STREAMLINK_PLUGIN_DIR'] = './streamlink/plugins/'

                        streamlink = Streamlink()
                        streams = streamlink.streams(url)
                        stream_url = streams['360p']

                        # WINDOWS AND LINUX HAS DIFFERENT BEHAVIOR WHEN RECORDING FFMPEG AS THREAD
                        # EVEN thread_record_streaming WAS DECLARED FIRST, IT ALWAYS GET LOADED AT LAST
                        if sys.platform == "win32":
                            thread_record_streaming = Thread(target=record_streaming_windows, args=(stream_url.url, tmp_recorded_streaming_filepath), daemon=True)
                            thread_record_streaming.start()

                        elif sys.platform == "linux":
                            thread_record_streaming = Thread(target=record_streaming_linux, args=(stream_url.url, tmp_recorded_streaming_filepath))
                            thread_record_streaming.start()

                    else:
                        FONT_TYPE = "Helvetica"
                        FONT_SIZE = 9
                        sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                        sg.Popup('Invalid URL, please enter a valid URL.                   ', title="Info", line_width=50)
                        recognizing = False
                        main_window['-START-BUTTON-'].update(('Stop','Start')[not recognizing], button_color=(('white', ('red', '#283b5b')[not recognizing])))
                        main_window['-RECORD-STREAMING-'].update(False)

                regions = []
                transcriptions = []

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:
                    time.sleep(1)

                tmp_src_txt_transcription_filename = "record.txt"
                tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
                tmp_src_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "w")
                tmp_src_txt_transcription_file.write("")
                tmp_src_txt_transcription_file.close()

                tmp_dst_txt_transcription_filename = "record.translated.txt"
                tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)
                tmp_dst_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "w")
                tmp_dst_txt_transcription_file.write("")
                tmp_dst_txt_transcription_file.close()

                main_window['-ML-SRC-PARTIAL-RESULTS-'].update("")
                main_window['-ML-SRC-RESULTS-'].update("")
                main_window['-ML-DST-PARTIAL-RESULTS-'].update("")
                main_window['-ML-DST-RESULTS-'].update("")

                thread_recognize = Thread(target=worker_recognize, args=(src, dst), daemon=True)
                thread_recognize.start()

                thread_timed_translate = Thread(target=worker_timed_translate, args=(src, dst), daemon=True)
                thread_timed_translate.start()

            else:
                #print("VOSK Live Subtitle is STOP LISTENING now")
                #print("recognizing = {}".format(recognizing))

                main_window['-ML-SRC-PARTIAL-RESULTS-'].update("")
                main_window['-ML-DST-PARTIAL-RESULTS-'].update("")

                if overlay_translation_window:
                    overlay_translation_window.Hide()

                if thread_recognize and thread_recognize.is_alive(): stop_thread(thread_recognize)
                if thread_timed_translate and thread_timed_translate.is_alive(): stop_thread(thread_timed_translate)

                #if thread_recognize and thread_recognize.is_alive(): thread_recognize.join()
                #if thread_timed_translate and thread_timed_translate.is_alive(): thread_timed_translate.join()

                # IF RECORD STREAMING
                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:

                    if sys.platform == "win32":
                        #print("thread_record_streaming.is_alive() = {}".format(thread_record_streaming.is_alive()))
                        stop_record_streaming_windows()

                    elif sys.platform == "linux":
                        #print("thread_record_streaming.is_alive() = {}".format(thread_record_streaming.is_alive()))
                        stop_record_streaming_linux()

                    tmp_src_subtitle_basename, extension = os.path.splitext(os.path.basename(tmp_recorded_streaming_filepath))
                    tmp_src_subtitle_filename = tmp_src_subtitle_basename + "." + subtitle_format
                    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)

                    # PERFORM TRANSCRIBE
                    if os.path.isfile(tmp_recorded_streaming_filepath):

                        if os.path.isfile(regions_filepath): os.remove(regions_filepath)
                        if os.path.isfile(transcriptions_filepath): os.remove(transcriptions_filepath)

                        not_transcribing = False
                        thread_vosk_transcribe = Thread(target=vosk_transcribe, args=(src, dst, tmp_recorded_streaming_filepath, subtitle_format), daemon=True)
                        thread_vosk_transcribe.start()
                        all_transcribe_threads.append(thread_vosk_transcribe)

                # IF NOT RECORD STREAMING
                else:
                    thread_save_temporary_transcriptions = Thread(target=save_temporary_transcriptions, args=(src, dst, subtitle_format), daemon=True)
                    thread_save_temporary_transcriptions.start()


                if thread_record_streaming and thread_record_streaming.is_alive():
                    if sys.platform == "win32":
                        stop_record_streaming_windows()

                    elif sys.platform == "linux":
                        stop_record_streaming_linux()

                # SAVING TRANSCRIPTIONS AS FORMATTED SUBTITLE FILE WITH FILENAME DECLARED IN COMMAND LINE ARGUMENTS
                #if args.subtitle_filename and os.path.isfile(regions_filepath) and os.path.isfile(transcriptions_filepath):
                if args.subtitle_filename:
                    thread_save_declared_subtitle_filename = Thread(target=save_as_declared_subtitle_filename, args=(args.subtitle_filename, src, dst), daemon=True)
                    thread_save_declared_subtitle_filename.start()

                if args.video_filename:
                    shutil.copy(tmp_recorded_streaming_filepath, saved_recorded_streaming_filename)


            if not (sys.platform == "win32"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event == '-EVENT-NULL-RESULTS-' and recognizing==True:

            overlay_translation_window.Hide()


        elif event == '-EVENT-PARTIAL-RESULTS-' and recognizing==True:

            text = str(values[event]).strip().lower()
            main_window['-ML-SRC-PARTIAL-RESULTS-'].update(text,background_color_for_value='yellow1',autoscroll=True)

            partial_result_filename = "partial_result"
            partial_result_filepath = os.path.join(tempfile.gettempdir(), partial_result_filename)
            partial_result_file = open(partial_result_filepath, "w")
            partial_result_file.write(text)
            partial_result_file.close()

            if not (sys.platform == "win32"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event == '-EVENT-RESULTS-' and recognizing==True:

            line = str(values[event]).strip().lower()
            main_window['-ML-SRC-RESULTS-'].update(line + "\n", background_color_for_value='yellow1', append=True, autoscroll=True)
            translated_line = GoogleTranslate(line, src, dst)
            main_window['-ML-DST-RESULTS-'].update(translated_line + "\n", background_color_for_value='yellow1', append=True, autoscroll=True)
            if overlay_translation_window is not None:
                overlay_translation_window.UnHide
                overlay_translation_window['-ML-OVERLAY-DST-PARTIAL-RESULTS-'].update(translated_line, background_color_for_value='black', visible=True, autoscroll=True)
            else:
                overlay_translation_window = make_overlay_translation_window(100*' ')
                overlay_translation_window['-ML-OVERLAY-DST-PARTIAL-RESULTS-'].update(translated_line, background_color_for_value='black', visible=True, autoscroll=True)


        elif event == '-EVENT-VOICE-TRANSLATED-' and recognizing==True:

            translated_text = str(values[event]).strip().lower()
            if translated_text:
                main_window['-ML-DST-PARTIAL-RESULTS-'].update(translated_text,background_color_for_value='yellow1',autoscroll=True)

                # SHOWING OVERLAY TRANSLATION WINDOW
                if not overlay_translation_window:
                    overlay_translation_window = make_overlay_translation_window(100*' ')
                overlay_translation_window['-ML-OVERLAY-DST-PARTIAL-RESULTS-'].update(translated_text,background_color_for_value='black', visible=True, autoscroll=True)
            else:
                overlay_translation_window.Hide()

            if not (sys.platform == "win32"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event == '-EVENT-THREAD-RECORD-STREAMING-STATUS-':

            msg = str(values[event]).strip()
            main_window['-RECORD-STREAMING-STATUS-'].update(msg)

            if "RECORDING" in msg:
                main_window['-RECORD-STREAMING-STATUS-'].update(text_color='white', background_color='red')
                main_window['-STREAMING-DURATION-RECORDED-'].update(text_color='white', background_color='red')
            else:
                main_window['-RECORD-STREAMING-STATUS-'].update(text_color='black', background_color='green1')
                main_window['-STREAMING-DURATION-RECORDED-'].update(text_color='black', background_color='green1')


        elif event == '-EVENT-STREAMING-DURATION-RECORDED-':

            record_duration = str(values[event]).strip()
            main_window['-STREAMING-DURATION-RECORDED-'].update(record_duration)


        elif event == '-EVENT-PROGRESS-BAR-':

            pb = values[event]
            info = pb[0]
            total = pb[1]
            percentage = pb[2]
            progress = pb[3]

            FONT_TYPE = "Helvetica"
            FONT_SIZE = 9
            sg.set_options(font=(FONT_TYPE, FONT_SIZE))

            progressbar.UnHide()

            progressbar['-INFO-'].update(info)
            progressbar['-PERCENTAGE-'].update(percentage)
            progressbar['-PROGRESS-'].update(progress)
            if progress == total:
                time.sleep(1)
                progressbar.close()


        elif event == '-EVENT-TRANSCRIBE-PREPARE-WINDOW-':

            prepare = values[event]

            if prepare:

                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                #transcribe_window.UnHide()
                transcribe_window = make_transcribe_window("", 100)

                for element in transcribe_window.element_list():
                    if isinstance(element, sg.Text) or isinstance(element, sg.Multiline):
                        element.update(font=(FONT_TYPE, FONT_SIZE))
                    if isinstance(element, sg.Button):
                        transcribe_window['-CANCEL-'].Widget.config(font=(FONT_TYPE, FONT_SIZE))

                transcribe_window['-OUTPUT-MESSAGES-'].update("")
                transcribe_window['-INFO-'].update("Progress info")
                transcribe_window['-PROGRESS-'].update(0)
                transcribe_window['-PERCENTAGE-'].update("0%")
                move_center(transcribe_window)
                transcribe_window.refresh()


        elif event == '-EVENT-TRANSCRIBE-SEND-MESSAGES-':

            m = values[event]
            window_key = m[0]
            msg = m[1]
            append_flag = m[2]

            transcribe_window[window_key].update(msg, append=append_flag)
            scroll_to_last_line(transcribe_window, transcribe_window[window_key])


        elif event == '-EVENT-TRANSCRIBE-SHOW-PROGRESS-BAR-':

            show = values[event]

            if show:
                transcribe_window['-INFO-'].update(visible=True)
                transcribe_window['-PROGRESS-'].update(visible=True)
                transcribe_window['-PERCENTAGE-'].update(visible=True)
            elif not show:
                transcribe_window['-INFO-'].update(visible=False)
                transcribe_window['-PROGRESS-'].update(visible=False)
                transcribe_window['-PERCENTAGE-'].update(visible=False)


        elif event == '-EVENT-TRANSCRIBE-HIDE-PROGRESS-BAR-':

            hide = values[event]

            transcribe_window['-INFO-'].update(visible=False)
            transcribe_window['-PROGRESS-'].update(visible=False)
            transcribe_window['-PERCENTAGE-'].update(visible=False)


        elif event == '-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-':

            pb = values[event]
            info = pb[0]
            percentage = pb[1]
            progress = pb[2]

            transcribe_window['-INFO-'].update(info)
            transcribe_window['-PERCENTAGE-'].update(percentage)
            transcribe_window['-PROGRESS-'].update(progress)


        if event == '-CANCEL-' or event == sg.WIN_CLOSED:
            if window == transcribe_window and transcribe_window is not None:
                if not not_transcribing:
                    main_window.minimize()
                    transcribe_window.minimize()
                    answer = popup_yes_no('             Are you sure?              ', title='Confirm')
                    if 'Yes' in answer:
                        not_transcribing = True
                        if thread_vosk_transcribe and thread_vosk_transcribe.is_alive(): stop_thread(thread_vosk_transcribe)
                        transcribe_window.close()
                        main_window.Normal()
                    else:
                        main_window.Normal()
                        transcribe_window.Normal()
                else:
                    if thread_vosk_transcribe and thread_vosk_transcribe.is_alive(): stop_thread(thread_vosk_transcribe)
                    transcribe_window.close()
                    main_window.Normal()


        elif event == '-EVENT-TRANSCRIBE-CLOSE-':

            close = values[event]
            if close: transcribe_window.Hide()


        elif event == '-EVENT-SAVE-RECORDED-STREAMING-':

            if not args.video_filename:
                tmp_recorded_streaming_filename = "record.mp4"
            else:
                tmp_recorded_streaming_filename = args.video_filename

            tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)

            if os.path.isfile(tmp_recorded_streaming_filepath):

                saved_recorded_streaming_filename = sg.popup_get_file('', no_window=True, save_as=True, font=(FONT_TYPE, FONT_SIZE), default_path=saved_recorded_streaming_filename, file_types=video_file_types)

                if saved_recorded_streaming_filename:
                    shutil.copy(tmp_recorded_streaming_filepath, saved_recorded_streaming_filename)

            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("No streaming was recorded.                             ", title="Info", line_width=50)


        elif event == '-EVENT-SAVE-SRC-TRANSCRIPTION-':
            if os.path.isfile(tmp_src_txt_transcription_filepath):

                saved_src_subtitle_filename = sg.popup_get_file('', no_window=True, save_as=True, font=(FONT_TYPE, FONT_SIZE), file_types=subtitle_file_types)

                if saved_src_subtitle_filename:

                    selected_subtitle_format = saved_src_subtitle_filename.split('.')[-1]

                    if selected_subtitle_format in FORMATTERS.keys():
                        timed_subtitles = [(r, t) for r, t in zip(regions, transcriptions) if t]
                        subtitle_format = selected_subtitle_format
                        formatter = FORMATTERS.get(subtitle_format)
                        formatted_subtitles = formatter(timed_subtitles)
                        tmp_src_subtitle_filename = saved_src_subtitle_filename
                        subtitle_file = open(tmp_src_subtitle_filename, 'wb')
                        subtitle_file.write(formatted_subtitles.encode("utf-8"))
                        subtitle_file.close()
                    else:
                        saved_src_subtitle_file = open(saved_src_subtitle_filename, "w")
                        tmp_src_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "r")
                        for line in tmp_src_txt_transcription_file:
                            if line:
                                saved_src_subtitle_file.write(line)
                        saved_src_subtitle_file.close()
                        tmp_src_txt_transcription_file.close()

            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("No transcriptions was recorded                             ", title="Info", line_width=50)


        elif event == '-EVENT-SAVE-DST-TRANSCRIPTION-':

            if os.path.isfile(tmp_dst_txt_transcription_filepath):

                saved_dst_subtitle_filename = sg.popup_get_file('', no_window=True, save_as=True, font=(FONT_TYPE, FONT_SIZE), file_types=subtitle_file_types)

                if saved_dst_subtitle_filename:

                    selected_subtitle_format = saved_dst_subtitle_filename.split('.')[-1]

                    if selected_subtitle_format in FORMATTERS.keys():
                        timed_translated_subtitles = [(r, t) for r, t in zip(created_regions, translated_transcriptions) if t]
                        subtitle_format = selected_subtitle_format
                        formatter = FORMATTERS.get(subtitle_format)
                        formatted_translated_subtitles = formatter(timed_translated_subtitles)
                        saved_dst_subtitle_file = open(saved_dst_subtitle_filename, 'wb')
                        saved_dst_subtitle_file.write(formatted_translated_subtitles.encode("utf-8"))
                        saved_dst_subtitle_file.close()
                    else:
                        saved_dst_subtitle_file = open(saved_dst_subtitle_filename, "w")
                        tmp_dst_txt_transcription_file = open(tmp_dst_txt_transcription_filepath, "r")
                        for line in tmp_dst_txt_transcription_file:
                            if line:
                                saved_dst_subtitle_file.write(line)
                        saved_dst_subtitle_file.close()
                        tmp_dst_txt_transcription_file.close()
            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("No translated transcriptions was recorded                 ", title="Info", line_width=50)


        elif event == '-EXCEPTION-':
            e = str(values[event]).strip()
            #sg.Popup("File format is not supported", title="Info", line_width=50)
            sg.Popup(e, title="Info", line_width=50)
            if transcribe_window: transcribe_window['-OUTPUT-MESSAGES-'].update("", append=False)


        elif event == '-LIVE-THREAD-':
            t = values[event]
            main_window['-DEBUG-'].update(t)


    if overlay_translation_window:
        overlay_translation_window.close()
        overlay_translation_window = None

    if transcribe_window:
        transcribe_window.close()
        transcribe_window = None

    if progressbar:
        progressbar.close()
        progressbar = None

    if thread_recognize and thread_recognize.is_alive():
        stop_thread(thread_recognize)

    if thread_timed_translate and thread_timed_translate.is_alive():
        stop_thread(thread_timed_translate)

    if thread_vosk_transcribe and thread_vosk_transcribe.is_alive():
        stop_thread(thread_vosk_transcribe)

    if thread_save_temporary_transcriptions and thread_save_temporary_transcriptions.is_alive():
        stop_thread(thread_save_temporary_transcriptions)

    if thread_save_declared_subtitle_filename and thread_save_declared_subtitle_filename.is_alive():
        stop_thread(thread_save_declared_subtitle_filename)


    if sys.platform == "win32":
        stop_ffmpeg_windows()
    else:
        stop_ffmpeg_linux()

    time.sleep(2)
    if tmp_recorded_streaming_filepath and os.path.isfile(tmp_recorded_streaming_filepath): os.remove(tmp_recorded_streaming_filepath)
    if partial_result_filepath and os.path.isfile(partial_result_filepath): os.remove(partial_result_filepath)
    if time_value_filepath and os.path.isfile(time_value_filepath): os.remove(time_value_filepath)
    if tmp_src_subtitle_filepath and os.path.isfile(tmp_src_subtitle_filepath): os.remove(tmp_src_subtitle_filepath)
    if tmp_dst_subtitle_filepath and os.path.isfile(tmp_dst_subtitle_filepath): os.remove(tmp_dst_subtitle_filepath)
    if tmp_src_txt_transcription_filepath and os.path.isfile(tmp_src_txt_transcription_filepath): os.remove(tmp_src_txt_transcription_filepath)
    if tmp_dst_txt_transcription_filepath and os.path.isfile(tmp_dst_txt_transcription_filepath): os.remove(tmp_dst_txt_transcription_filepath)
    if os.path.isfile(regions_filepath): os.remove(regions_filepath)
    if os.path.isfile(transcriptions_filepath): os.remove(transcriptions_filepath)
    if os.path.isfile(transcribe_is_done_filepath): os.remove(transcribe_is_done_filepath)

    remove_temp_files("wav")

    main_window.close()
    sys.exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
