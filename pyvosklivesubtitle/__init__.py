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
if sys.platform == "win32":
    import win32clipboard
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

VERSION = '0.2.1'


#============================================================== VOSK PART ==============================================================#


import requests
from urllib.request import urlretrieve
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
    elif sys.platform == 'darwin':
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


class VoskLanguage:
    def __init__(self):
        self.list_langs = []
        self.list_langs.append("ca")
        self.list_langs.append("cn")
        self.list_langs.append("cs")
        self.list_langs.append("nl")
        self.list_langs.append("en-us")
        self.list_langs.append("eo")
        self.list_langs.append("fr")
        self.list_langs.append("de")
        self.list_langs.append("hi")
        self.list_langs.append("it")
        self.list_langs.append("ja")
        self.list_langs.append("kz")
        self.list_langs.append("ko")
        self.list_langs.append("fa")
        self.list_langs.append("pl")
        self.list_langs.append("pt")
        self.list_langs.append("ru")
        self.list_langs.append("es")
        self.list_langs.append("sv")
        self.list_langs.append("tr")
        self.list_langs.append("ua")
        self.list_langs.append("uz")
        self.list_langs.append("vn")

        self.list_models = []
        self.list_models.append("vosk-model-small-ca-0.4")
        self.list_models.append("vosk-model-small-cn-0.22")
        self.list_models.append("vosk-model-small-cs-0.4-rhasspy")
        self.list_models.append("vosk-model-small-nl-0.22")
        self.list_models.append("vosk-model-small-en-us-0.15")
        self.list_models.append("vosk-model-small-eo-0.42")
        self.list_models.append("vosk-model-small-fr-0.22")
        self.list_models.append("vosk-model-small-de-0.15")
        self.list_models.append("vosk-model-small-hi-0.22")
        self.list_models.append("vosk-model-small-it-0.22")
        self.list_models.append("vosk-model-small-ja-0.22")
        self.list_models.append("vosk-model-small-kz-0.15")
        self.list_models.append("vosk-model-small-ko-0.22")
        self.list_models.append("vosk-model-small-fa-0.5")
        self.list_models.append("vosk-model-small-pl-0.22")
        self.list_models.append("vosk-model-small-pt-0.3")
        self.list_models.append("vosk-model-small-ru-0.22")
        self.list_models.append("vosk-model-small-es-0.42")
        self.list_models.append("vosk-model-small-sv-rhasspy-0.15")
        self.list_models.append("vosk-model-small-tr-0.3")
        self.list_models.append("vosk-model-small-uk-v3-small")
        self.list_models.append("vosk-model-small-uz-0.22")
        self.list_models.append("vosk-model-small-vn-0.3")

        self.list_codes = []
        self.list_codes.append("ca")
        self.list_codes.append("zh")
        self.list_codes.append("cs")
        self.list_codes.append("nl")
        self.list_codes.append("en")
        self.list_codes.append("eo")
        self.list_codes.append("fr")
        self.list_codes.append("de")
        self.list_codes.append("hi")
        self.list_codes.append("it")
        self.list_codes.append("ja")
        self.list_codes.append("kk")
        self.list_codes.append("ko")
        self.list_codes.append("fa")
        self.list_codes.append("pl")
        self.list_codes.append("pt")
        self.list_codes.append("ru")
        self.list_codes.append("es")
        self.list_codes.append("sv")
        self.list_codes.append("tr")
        self.list_codes.append("uk")
        self.list_codes.append("vi")

        self.list_names = []
        self.list_names.append("Catalan")
        self.list_names.append("Chinese")
        self.list_names.append("Czech")
        self.list_names.append("Dutch")
        self.list_names.append("English")
        self.list_names.append("Esperanto")
        self.list_names.append("French")
        self.list_names.append("German")
        self.list_names.append("Hindi")
        self.list_names.append("Italian")
        self.list_names.append("Japanese")
        self.list_names.append("Kazakh")
        self.list_names.append("Korean")
        self.list_names.append("Persian")
        self.list_names.append("Polish")
        self.list_names.append("Portuguese")
        self.list_names.append("Russian")
        self.list_names.append("Spanish")
        self.list_names.append("Swedish")
        self.list_names.append("Turkish")
        self.list_names.append("Ukrainian")
        self.list_names.append("Vietnamese")

        self.lang_of_name = dict(zip(self.list_names, self.list_langs))
        self.lang_of_code = dict(zip(self.list_codes, self.list_langs))
        self.lang_of_model = dict(zip(self.list_models, self.list_langs))

        self.model_of_name = dict(zip(self.list_names, self.list_models))
        self.model_of_code = dict(zip(self.list_codes, self.list_models))
        self.model_of_lang = dict(zip(self.list_langs, self.list_models))

        self.name_of_model = dict(zip(self.list_models, self.list_names))
        self.name_of_code = dict(zip(self.list_codes, self.list_names))
        self.name_of_lang = dict(zip(self.list_langs, self.list_names))

        self.code_of_name = dict(zip(self.list_names, self.list_codes))
        self.code_of_lang = dict(zip(self.list_langs, self.list_codes))
        self.code_of_model = dict(zip(self.list_models, self.list_codes))

        self.dict = {
                        'ca': 'Catalan',
                        'zh': 'Chinese',
                        'cs': 'Czech',
                        'nl': 'Dutch',
                        'en': 'English',
                        'eo': 'Esperanto',
                        'fr': 'French',
                        'de': 'German',
                        'hi': 'Hindi',
                        'it': 'Italian',
                        'ja': 'Japanese',
                        'kk': 'Kazakh',
                        'ko': 'Korean',
                        'fa': 'Persian',
                        'pl': 'Polish',
                        'pt': 'Portuguese',
                        'ru': 'Russian',
                        'es': 'Spanish',
                        'sv': 'Swedish',
                        'tr': 'Turkish',
                        'uk': 'Ukrainian',
                        'vi': 'Vietnamese',
					}

    def get_lang_of_name(self, name):
        return self.lang_of_name[name]

    def get_lang_of_code(self, code):
        return self.lang_of_code[code]

    def get_lang_of_model(self, model):
        return self.lang_of_model[model]

    def get_model_of_name(self, name):
        return self.model_of_name[name]

    def get_model_of_code(self, code):
        return self.model_of_code[code]

    def get_model_of_lang(self, lang):
        return self.model_of_lang[lang]

    def get_name_of_model(self, model):
        return self.name_of_model[model]

    def get_name_of_code(self, code):
        return self.name_of_code[code]

    def get_name_of_lang(self, lang):
        return self.name_of_lang[lang]

    def get_code_of_name(self, name):
        return self.code_of_name[name]

    def get_code_of_lang(self, lang):
        return self.code_of_lang[lang]

    def get_code_of_model(self, model):
        return self.code_of_model[model]


class GoogleLanguage:
    def __init__(self):
        self.list_codes = []
        self.list_codes.append("af")
        self.list_codes.append("sq")
        self.list_codes.append("am")
        self.list_codes.append("ar")
        self.list_codes.append("hy")
        self.list_codes.append("as")
        self.list_codes.append("ay")
        self.list_codes.append("az")
        self.list_codes.append("bm")
        self.list_codes.append("eu")
        self.list_codes.append("be")
        self.list_codes.append("bn")
        self.list_codes.append("bho")
        self.list_codes.append("bs")
        self.list_codes.append("bg")
        self.list_codes.append("ca")
        self.list_codes.append("ceb")
        self.list_codes.append("ny")
        self.list_codes.append("zh")
        self.list_codes.append("zh-CN")
        self.list_codes.append("zh-TW")
        self.list_codes.append("co")
        self.list_codes.append("hr")
        self.list_codes.append("cs")
        self.list_codes.append("da")
        self.list_codes.append("dv")
        self.list_codes.append("doi")
        self.list_codes.append("nl")
        self.list_codes.append("en")
        self.list_codes.append("eo")
        self.list_codes.append("et")
        self.list_codes.append("ee")
        self.list_codes.append("fil")
        self.list_codes.append("fi")
        self.list_codes.append("fr")
        self.list_codes.append("fy")
        self.list_codes.append("gl")
        self.list_codes.append("ka")
        self.list_codes.append("de")
        self.list_codes.append("el")
        self.list_codes.append("gn")
        self.list_codes.append("gu")
        self.list_codes.append("ht")
        self.list_codes.append("ha")
        self.list_codes.append("haw")
        self.list_codes.append("he")
        self.list_codes.append("hi")
        self.list_codes.append("hmn")
        self.list_codes.append("hu")
        self.list_codes.append("is")
        self.list_codes.append("ig")
        self.list_codes.append("ilo")
        self.list_codes.append("id")
        self.list_codes.append("ga")
        self.list_codes.append("it")
        self.list_codes.append("ja")
        self.list_codes.append("jv")
        self.list_codes.append("kn")
        self.list_codes.append("kk")
        self.list_codes.append("km")
        self.list_codes.append("rw")
        self.list_codes.append("gom")
        self.list_codes.append("ko")
        self.list_codes.append("kri")
        self.list_codes.append("kmr")
        self.list_codes.append("ckb")
        self.list_codes.append("ky")
        self.list_codes.append("lo")
        self.list_codes.append("la")
        self.list_codes.append("lv")
        self.list_codes.append("ln")
        self.list_codes.append("lt")
        self.list_codes.append("lg")
        self.list_codes.append("lb")
        self.list_codes.append("mk")
        self.list_codes.append("mg")
        self.list_codes.append("ms")
        self.list_codes.append("ml")
        self.list_codes.append("mt")
        self.list_codes.append("mi")
        self.list_codes.append("mr")
        self.list_codes.append("mni-Mtei")
        self.list_codes.append("lus")
        self.list_codes.append("mn")
        self.list_codes.append("my")
        self.list_codes.append("ne")
        self.list_codes.append("no")
        self.list_codes.append("or")
        self.list_codes.append("om")
        self.list_codes.append("ps")
        self.list_codes.append("fa")
        self.list_codes.append("pl")
        self.list_codes.append("pt")
        self.list_codes.append("pa")
        self.list_codes.append("qu")
        self.list_codes.append("ro")
        self.list_codes.append("ru")
        self.list_codes.append("sm")
        self.list_codes.append("sa")
        self.list_codes.append("gd")
        self.list_codes.append("nso")
        self.list_codes.append("sr")
        self.list_codes.append("st")
        self.list_codes.append("sn")
        self.list_codes.append("sd")
        self.list_codes.append("si")
        self.list_codes.append("sk")
        self.list_codes.append("sl")
        self.list_codes.append("so")
        self.list_codes.append("es")
        self.list_codes.append("su")
        self.list_codes.append("sw")
        self.list_codes.append("sv")
        self.list_codes.append("tg")
        self.list_codes.append("ta")
        self.list_codes.append("tt")
        self.list_codes.append("te")
        self.list_codes.append("th")
        self.list_codes.append("ti")
        self.list_codes.append("ts")
        self.list_codes.append("tr")
        self.list_codes.append("tk")
        self.list_codes.append("tw")
        self.list_codes.append("uk")
        self.list_codes.append("ur")
        self.list_codes.append("ug")
        self.list_codes.append("uz")
        self.list_codes.append("vi")
        self.list_codes.append("cy")
        self.list_codes.append("xh")
        self.list_codes.append("yi")
        self.list_codes.append("yo")
        self.list_codes.append("zu")

        self.list_names = []
        self.list_names.append("Afrikaans")
        self.list_names.append("Albanian")
        self.list_names.append("Amharic")
        self.list_names.append("Arabic")
        self.list_names.append("Armenian")
        self.list_names.append("Assamese")
        self.list_names.append("Aymara")
        self.list_names.append("Azerbaijani")
        self.list_names.append("Bambara")
        self.list_names.append("Basque")
        self.list_names.append("Belarusian")
        self.list_names.append("Bengali")
        self.list_names.append("Bhojpuri")
        self.list_names.append("Bosnian")
        self.list_names.append("Bulgarian")
        self.list_names.append("Catalan")
        self.list_names.append("Cebuano")
        self.list_names.append("Chichewa")
        self.list_names.append("Chinese")
        self.list_names.append("Chinese (Simplified)")
        self.list_names.append("Chinese (Traditional)")
        self.list_names.append("Corsican")
        self.list_names.append("Croatian")
        self.list_names.append("Czech")
        self.list_names.append("Danish")
        self.list_names.append("Dhivehi")
        self.list_names.append("Dogri")
        self.list_names.append("Dutch")
        self.list_names.append("English")
        self.list_names.append("Esperanto")
        self.list_names.append("Estonian")
        self.list_names.append("Ewe")
        self.list_names.append("Filipino")
        self.list_names.append("Finnish")
        self.list_names.append("French")
        self.list_names.append("Frisian")
        self.list_names.append("Galician")
        self.list_names.append("Georgian")
        self.list_names.append("German")
        self.list_names.append("Greek")
        self.list_names.append("Guarani")
        self.list_names.append("Gujarati")
        self.list_names.append("Haitian Creole")
        self.list_names.append("Hausa")
        self.list_names.append("Hawaiian")
        self.list_names.append("Hebrew")
        self.list_names.append("Hindi")
        self.list_names.append("Hmong")
        self.list_names.append("Hungarian")
        self.list_names.append("Icelandic")
        self.list_names.append("Igbo")
        self.list_names.append("Ilocano")
        self.list_names.append("Indonesian")
        self.list_names.append("Irish")
        self.list_names.append("Italian")
        self.list_names.append("Japanese")
        self.list_names.append("Javanese")
        self.list_names.append("Kannada")
        self.list_names.append("Kazakh")
        self.list_names.append("Khmer")
        self.list_names.append("Kinyarwanda")
        self.list_names.append("Konkani")
        self.list_names.append("Korean")
        self.list_names.append("Krio")
        self.list_names.append("Kurdish (Kurmanji)")
        self.list_names.append("Kurdish (Sorani)")
        self.list_names.append("Kyrgyz")
        self.list_names.append("Lao")
        self.list_names.append("Latin")
        self.list_names.append("Latvian")
        self.list_names.append("Lingala")
        self.list_names.append("Lithuanian")
        self.list_names.append("Luganda")
        self.list_names.append("Luxembourgish")
        self.list_names.append("Macedonian")
        self.list_names.append("Malagasy")
        self.list_names.append("Malay")
        self.list_names.append("Malayalam")
        self.list_names.append("Maltese")
        self.list_names.append("Maori")
        self.list_names.append("Marathi")
        self.list_names.append("Meiteilon (Manipuri)")
        self.list_names.append("Mizo")
        self.list_names.append("Mongolian")
        self.list_names.append("Myanmar (Burmese)")
        self.list_names.append("Nepali")
        self.list_names.append("Norwegian")
        self.list_names.append("Odiya (Oriya)")
        self.list_names.append("Oromo")
        self.list_names.append("Pashto")
        self.list_names.append("Persian")
        self.list_names.append("Polish")
        self.list_names.append("Portuguese")
        self.list_names.append("Punjabi")
        self.list_names.append("Quechua")
        self.list_names.append("Romanian")
        self.list_names.append("Russian")
        self.list_names.append("Samoan")
        self.list_names.append("Sanskrit")
        self.list_names.append("Scots Gaelic")
        self.list_names.append("Sepedi")
        self.list_names.append("Serbian")
        self.list_names.append("Sesotho")
        self.list_names.append("Shona")
        self.list_names.append("Sindhi")
        self.list_names.append("Sinhala")
        self.list_names.append("Slovak")
        self.list_names.append("Slovenian")
        self.list_names.append("Somali")
        self.list_names.append("Spanish")
        self.list_names.append("Sundanese")
        self.list_names.append("Swahili")
        self.list_names.append("Swedish")
        self.list_names.append("Tajik")
        self.list_names.append("Tamil")
        self.list_names.append("Tatar")
        self.list_names.append("Telugu")
        self.list_names.append("Thai")
        self.list_names.append("Tigrinya")
        self.list_names.append("Tsonga")
        self.list_names.append("Turkish")
        self.list_names.append("Turkmen")
        self.list_names.append("Twi (Akan)")
        self.list_names.append("Ukrainian")
        self.list_names.append("Urdu")
        self.list_names.append("Uyghur")
        self.list_names.append("Uzbek")
        self.list_names.append("Vietnamese")
        self.list_names.append("Welsh")
        self.list_names.append("Xhosa")
        self.list_names.append("Yiddish")
        self.list_names.append("Yoruba")
        self.list_names.append("Zulu")

        # NOTE THAT Google Translate AND Vosk Speech Recognition API USE ISO-639-1 STANDARD CODE ('al', 'af', 'as', ETC)
        # WHEN ffmpeg SUBTITLES STREAMS USE ISO 639-2 STANDARD CODE ('afr', 'alb', 'amh', ETC)

        self.list_ffmpeg_codes = []
        self.list_ffmpeg_codes.append("afr")  # Afrikaans
        self.list_ffmpeg_codes.append("alb")  # Albanian
        self.list_ffmpeg_codes.append("amh")  # Amharic
        self.list_ffmpeg_codes.append("ara")  # Arabic
        self.list_ffmpeg_codes.append("hye")  # Armenian
        self.list_ffmpeg_codes.append("asm")  # Assamese
        self.list_ffmpeg_codes.append("aym")  # Aymara
        self.list_ffmpeg_codes.append("aze")  # Azerbaijani
        self.list_ffmpeg_codes.append("bam")  # Bambara
        self.list_ffmpeg_codes.append("eus")  # Basque
        self.list_ffmpeg_codes.append("bel")  # Belarusian
        self.list_ffmpeg_codes.append("ben")  # Bengali
        self.list_ffmpeg_codes.append("bho")  # Bhojpuri
        self.list_ffmpeg_codes.append("bos")  # Bosnian
        self.list_ffmpeg_codes.append("bul")  # Bulgarian
        self.list_ffmpeg_codes.append("cat")  # Catalan
        self.list_ffmpeg_codes.append("ceb")  # Cebuano
        self.list_ffmpeg_codes.append("nya")  # Chichewa
        self.list_ffmpeg_codes.append("zho")  # Chinese
        self.list_ffmpeg_codes.append("zho-CN")  # Chinese (Simplified)
        self.list_ffmpeg_codes.append("zho-TW")  # Chinese (Traditional)
        self.list_ffmpeg_codes.append("cos")  # Corsican
        self.list_ffmpeg_codes.append("hrv")  # Croatian
        self.list_ffmpeg_codes.append("ces")  # Czech
        self.list_ffmpeg_codes.append("dan")  # Danish
        self.list_ffmpeg_codes.append("div")  # Dhivehi
        self.list_ffmpeg_codes.append("doi")  # Dogri
        self.list_ffmpeg_codes.append("nld")  # Dutch
        self.list_ffmpeg_codes.append("eng")  # English
        self.list_ffmpeg_codes.append("epo")  # Esperanto
        self.list_ffmpeg_codes.append("est")  # Estonian
        self.list_ffmpeg_codes.append("ewe")  # Ewe
        self.list_ffmpeg_codes.append("fil")  # Filipino
        self.list_ffmpeg_codes.append("fin")  # Finnish
        self.list_ffmpeg_codes.append("fra")  # French
        self.list_ffmpeg_codes.append("fry")  # Frisian
        self.list_ffmpeg_codes.append("glg")  # Galician
        self.list_ffmpeg_codes.append("kat")  # Georgian
        self.list_ffmpeg_codes.append("deu")  # German
        self.list_ffmpeg_codes.append("ell")  # Greek
        self.list_ffmpeg_codes.append("grn")  # Guarani
        self.list_ffmpeg_codes.append("guj")  # Gujarati
        self.list_ffmpeg_codes.append("hat")  # Haitian Creole
        self.list_ffmpeg_codes.append("hau")  # Hausa
        self.list_ffmpeg_codes.append("haw")  # Hawaiian
        self.list_ffmpeg_codes.append("heb")  # Hebrew
        self.list_ffmpeg_codes.append("hin")  # Hindi
        self.list_ffmpeg_codes.append("hmn")  # Hmong
        self.list_ffmpeg_codes.append("hun")  # Hungarian
        self.list_ffmpeg_codes.append("isl")  # Icelandic
        self.list_ffmpeg_codes.append("ibo")  # Igbo
        self.list_ffmpeg_codes.append("ilo")  # Ilocano
        self.list_ffmpeg_codes.append("ind")  # Indonesian
        self.list_ffmpeg_codes.append("gle")  # Irish
        self.list_ffmpeg_codes.append("ita")  # Italian
        self.list_ffmpeg_codes.append("jpn")  # Japanese
        self.list_ffmpeg_codes.append("jav")  # Javanese
        self.list_ffmpeg_codes.append("kan")  # Kannada
        self.list_ffmpeg_codes.append("kaz")  # Kazakh
        self.list_ffmpeg_codes.append("khm")  # Khmer
        self.list_ffmpeg_codes.append("kin")  # Kinyarwanda
        self.list_ffmpeg_codes.append("kok")  # Konkani
        self.list_ffmpeg_codes.append("kor")  # Korean
        self.list_ffmpeg_codes.append("kri")  # Krio
        self.list_ffmpeg_codes.append("kmr")  # Kurdish (Kurmanji)
        self.list_ffmpeg_codes.append("ckb")  # Kurdish (Sorani)
        self.list_ffmpeg_codes.append("kir")  # Kyrgyz
        self.list_ffmpeg_codes.append("lao")  # Lao
        self.list_ffmpeg_codes.append("lat")  # Latin
        self.list_ffmpeg_codes.append("lav")  # Latvian
        self.list_ffmpeg_codes.append("lin")  # Lingala
        self.list_ffmpeg_codes.append("lit")  # Lithuanian
        self.list_ffmpeg_codes.append("lug")  # Luganda
        self.list_ffmpeg_codes.append("ltz")  # Luxembourgish
        self.list_ffmpeg_codes.append("mkd")  # Macedonian
        self.list_ffmpeg_codes.append("mlg")  # Malagasy
        self.list_ffmpeg_codes.append("msa")  # Malay
        self.list_ffmpeg_codes.append("mal")  # Malayalam
        self.list_ffmpeg_codes.append("mlt")  # Maltese
        self.list_ffmpeg_codes.append("mri")  # Maori
        self.list_ffmpeg_codes.append("mar")  # Marathi
        self.list_ffmpeg_codes.append("mni-Mtei")  # Meiteilon (Manipuri)
        self.list_ffmpeg_codes.append("lus")  # Mizo
        self.list_ffmpeg_codes.append("mon")  # Mongolian
        self.list_ffmpeg_codes.append("mya")  # Myanmar (Burmese)
        self.list_ffmpeg_codes.append("nep")  # Nepali
        self.list_ffmpeg_codes.append("nor")  # Norwegian
        self.list_ffmpeg_codes.append("ori")  # Odiya (Oriya)
        self.list_ffmpeg_codes.append("orm")  # Oromo
        self.list_ffmpeg_codes.append("pus")  # Pashto
        self.list_ffmpeg_codes.append("fas")  # Persian
        self.list_ffmpeg_codes.append("pol")  # Polish
        self.list_ffmpeg_codes.append("por")  # Portuguese
        self.list_ffmpeg_codes.append("pan")  # Punjabi
        self.list_ffmpeg_codes.append("que")  # Quechua
        self.list_ffmpeg_codes.append("ron")  # Romanian
        self.list_ffmpeg_codes.append("rus")  # Russian
        self.list_ffmpeg_codes.append("smo")  # Samoan
        self.list_ffmpeg_codes.append("san")  # Sanskrit
        self.list_ffmpeg_codes.append("gla")  # Scots Gaelic
        self.list_ffmpeg_codes.append("nso")  # Sepedi
        self.list_ffmpeg_codes.append("srp")  # Serbian
        self.list_ffmpeg_codes.append("sot")  # Sesotho
        self.list_ffmpeg_codes.append("sna")  # Shona
        self.list_ffmpeg_codes.append("snd")  # Sindhi
        self.list_ffmpeg_codes.append("sin")  # Sinhala
        self.list_ffmpeg_codes.append("slk")  # Slovak
        self.list_ffmpeg_codes.append("slv")  # Slovenian
        self.list_ffmpeg_codes.append("som")  # Somali
        self.list_ffmpeg_codes.append("spa")  # Spanish
        self.list_ffmpeg_codes.append("sun")  # Sundanese
        self.list_ffmpeg_codes.append("swa")  # Swahili
        self.list_ffmpeg_codes.append("swe")  # Swedish
        self.list_ffmpeg_codes.append("tgk")  # Tajik
        self.list_ffmpeg_codes.append("tam")  # Tamil
        self.list_ffmpeg_codes.append("tat")  # Tatar
        self.list_ffmpeg_codes.append("tel")  # Telugu
        self.list_ffmpeg_codes.append("tha")  # Thai
        self.list_ffmpeg_codes.append("tir")  # Tigrinya
        self.list_ffmpeg_codes.append("tso")  # Tsonga
        self.list_ffmpeg_codes.append("tur")  # Turkish
        self.list_ffmpeg_codes.append("tuk")  # Turkmen
        self.list_ffmpeg_codes.append("twi")  # Twi (Akan)
        self.list_ffmpeg_codes.append("ukr")  # Ukrainian
        self.list_ffmpeg_codes.append("urd")  # Urdu
        self.list_ffmpeg_codes.append("uig")  # Uyghur
        self.list_ffmpeg_codes.append("uzb")  # Uzbek
        self.list_ffmpeg_codes.append("vie")  # Vietnamese
        self.list_ffmpeg_codes.append("wel")  # Welsh
        self.list_ffmpeg_codes.append("xho")  # Xhosa
        self.list_ffmpeg_codes.append("yid")  # Yiddish
        self.list_ffmpeg_codes.append("yor")  # Yoruba
        self.list_ffmpeg_codes.append("zul")  # Zulu

        self.code_of_name = dict(zip(self.list_names, self.list_codes))
        self.code_of_ffmpeg_code = dict(zip(self.list_ffmpeg_codes, self.list_codes))

        self.name_of_code = dict(zip(self.list_codes, self.list_names))
        self.name_of_ffmpeg_code = dict(zip(self.list_ffmpeg_codes, self.list_names))

        self.ffmpeg_code_of_name = dict(zip(self.list_names, self.list_ffmpeg_codes))
        self.ffmpeg_code_of_code = dict(zip(self.list_codes, self.list_ffmpeg_codes))

        self.dict = {
                        'af': 'Afrikaans',
                        'sq': 'Albanian',
                        'am': 'Amharic',
                        'ar': 'Arabic',
                        'hy': 'Armenian',
                        'as': 'Assamese',
                        'ay': 'Aymara',
                        'az': 'Azerbaijani',
                        'bm': 'Bambara',
                        'eu': 'Basque',
                        'be': 'Belarusian',
                        'bn': 'Bengali',
                        'bho': 'Bhojpuri',
                        'bs': 'Bosnian',
                        'bg': 'Bulgarian',
                        'ca': 'Catalan',
                        'ceb': 'Cebuano',
                        'ny': 'Chichewa',
                        'zh': 'Chinese',
                        'zh-CN': 'Chinese (Simplified)',
                        'zh-TW': 'Chinese (Traditional)',
                        'co': 'Corsican',
                        'hr': 'Croatian',
                        'cs': 'Czech',
                        'da': 'Danish',
                        'dv': 'Dhivehi',
                        'doi': 'Dogri',
                        'nl': 'Dutch',
                        'en': 'English',
                        'eo': 'Esperanto',
                        'et': 'Estonian',
                        'ee': 'Ewe',
                        'fil': 'Filipino',
                        'fi': 'Finnish',
                        'fr': 'French',
                        'fy': 'Frisian',
                        'gl': 'Galician',
                        'ka': 'Georgian',
                        'de': 'German',
                        'el': 'Greek',
                        'gn': 'Guarani',
                        'gu': 'Gujarati',
                        'ht': 'Haitian Creole',
                        'ha': 'Hausa',
                        'haw': 'Hawaiian',
                        'he': 'Hebrew',
                        'hi': 'Hindi',
                        'hmn': 'Hmong',
                        'hu': 'Hungarian',
                        'is': 'Icelandic',
                        'ig': 'Igbo',
                        'ilo': 'Ilocano',
                        'id': 'Indonesian',
                        'ga': 'Irish',
                        'it': 'Italian',
                        'ja': 'Japanese',
                        'jv': 'Javanese',
                        'kn': 'Kannada',
                        'kk': 'Kazakh',
                        'km': 'Khmer',
                        'rw': 'Kinyarwanda',
                        'gom': 'Konkani',
                        'ko': 'Korean',
                        'kri': 'Krio',
                        'kmr': 'Kurdish (Kurmanji)',
                        'ckb': 'Kurdish (Sorani)',
                        'ky': 'Kyrgyz',
                        'lo': 'Lao',
                        'la': 'Latin',
                        'lv': 'Latvian',
                        'ln': 'Lingala',
                        'lt': 'Lithuanian',
                        'lg': 'Luganda',
                        'lb': 'Luxembourgish',
                        'mk': 'Macedonian',
                        'mg': 'Malagasy',
                        'ms': 'Malay',
                        'ml': 'Malayalam',
                        'mt': 'Maltese',
                        'mi': 'Maori',
                        'mr': 'Marathi',
                        'mni-Mtei': 'Meiteilon (Manipuri)',
                        'lus': 'Mizo',
                        'mn': 'Mongolian',
                        'my': 'Myanmar (Burmese)',
                        'ne': 'Nepali',
                        'no': 'Norwegian',
                        'or': 'Odiya (Oriya)',
                        'om': 'Oromo',
                        'ps': 'Pashto',
                        'fa': 'Persian',
                        'pl': 'Polish',
                        'pt': 'Portuguese',
                        'pa': 'Punjabi',
                        'qu': 'Quechua',
                        'ro': 'Romanian',
                        'ru': 'Russian',
                        'sm': 'Samoan',
                        'sa': 'Sanskrit',
                        'gd': 'Scots Gaelic',
                        'nso': 'Sepedi',
                        'sr': 'Serbian',
                        'st': 'Sesotho',
                        'sn': 'Shona',
                        'sd': 'Sindhi',
                        'si': 'Sinhala',
                        'sk': 'Slovak',
                        'sl': 'Slovenian',
                        'so': 'Somali',
                        'es': 'Spanish',
                        'su': 'Sundanese',
                        'sw': 'Swahili',
                        'sv': 'Swedish',
                        'tg': 'Tajik',
                        'ta': 'Tamil',
                        'tt': 'Tatar',
                        'te': 'Telugu',
                        'th': 'Thai',
                        'ti': 'Tigrinya',
                        'ts': 'Tsonga',
                        'tr': 'Turkish',
                        'tk': 'Turkmen',
                        'tw': 'Twi (Akan)',
                        'uk': 'Ukrainian',
                        'ur': 'Urdu',
                        'ug': 'Uyghur',
                        'uz': 'Uzbek',
                        'vi': 'Vietnamese',
                        'cy': 'Welsh',
                        'xh': 'Xhosa',
                        'yi': 'Yiddish',
                        'yo': 'Yoruba',
                        'zu': 'Zulu',
                    }

        self.ffmpeg_dict = {
                                'af': 'afr', # Afrikaans
                                'sq': 'alb', # Albanian
                                'am': 'amh', # Amharic
                                'ar': 'ara', # Arabic
                                'hy': 'arm', # Armenian
                                'as': 'asm', # Assamese
                                'ay': 'aym', # Aymara
                                'az': 'aze', # Azerbaijani
                                'bm': 'bam', # Bambara
                                'eu': 'baq', # Basque
                                'be': 'bel', # Belarusian
                                'bn': 'ben', # Bengali
                                'bho': 'bho', # Bhojpuri
                                'bs': 'bos', # Bosnian
                                'bg': 'bul', # Bulgarian
                                'ca': 'cat', # Catalan
                                'ceb': 'ceb', # Cebuano
                                'ny': 'nya', # Chichewa
                                'zh': 'chi', # Chinese
                                'zh-CN': 'chi', # Chinese (Simplified)
                                'zh-TW': 'chi', # Chinese (Traditional)
                                'co': 'cos', # Corsican
                                'hr': 'hrv', # Croatian
                                'cs': 'cze', # Czech
                                'da': 'dan', # Danish
                                'dv': 'div', # Dhivehi
                                'doi': 'doi', # Dogri
                                'nl': 'dut', # Dutch
                                'en': 'eng', # English
                                'eo': 'epo', # Esperanto
                                'et': 'est', # Estonian
                                'ee': 'ewe', # Ewe
                                'fil': 'fil', # Filipino
                                'fi': 'fin', # Finnish
                                'fr': 'fre', # French
                                'fy': 'fry', # Frisian
                                'gl': 'glg', # Galician
                                'ka': 'geo', # Georgian
                                'de': 'ger', # German
                                'el': 'gre', # Greek
                                'gn': 'grn', # Guarani
                                'gu': 'guj', # Gujarati
                                'ht': 'hat', # Haitian Creole
                                'ha': 'hau', # Hausa
                                'haw': 'haw', # Hawaiian
                                'he': 'heb', # Hebrew
                                'hi': 'hin', # Hindi
                                'hmn': 'hmn', # Hmong
                                'hu': 'hun', # Hungarian
                                'is': 'ice', # Icelandic
                                'ig': 'ibo', # Igbo
                                'ilo': 'ilo', # Ilocano
                                'id': 'ind', # Indonesian
                                'ga': 'gle', # Irish
                                'it': 'ita', # Italian
                                'ja': 'jpn', # Japanese
                                'jv': 'jav', # Javanese
                                'kn': 'kan', # Kannada
                                'kk': 'kaz', # Kazakh
                                'km': 'khm', # Khmer
                                'rw': 'kin', # Kinyarwanda
                                'gom': 'kok', # Konkani
                                'ko': 'kor', # Korean
                                'kri': 'kri', # Krio
                                'kmr': 'kur', # Kurdish (Kurmanji)
                                'ckb': 'kur', # Kurdish (Sorani)
                                'ky': 'kir', # Kyrgyz
                                'lo': 'lao', # Lao
                                'la': 'lat', # Latin
                                'lv': 'lav', # Latvian
                                'ln': 'lin', # Lingala
                                'lt': 'lit', # Lithuanian
                                'lg': 'lug', # Luganda
                                'lb': 'ltz', # Luxembourgish
                                'mk': 'mac', # Macedonian
                                'mg': 'mlg', # Malagasy
                                'ms': 'may', # Malay
                                'ml': 'mal', # Malayalam
                                'mt': 'mlt', # Maltese
                                'mi': 'mao', # Maori
                                'mr': 'mar', # Marathi
                                'mni-Mtei': 'mni', # Meiteilon (Manipuri)
                                'lus': 'lus', # Mizo
                                'mn': 'mon', # Mongolian
                                'my': 'bur', # Myanmar (Burmese)
                                'ne': 'nep', # Nepali
                                'no': 'nor', # Norwegian
                                'or': 'ori', # Odiya (Oriya)
                                'om': 'orm', # Oromo
                                'ps': 'pus', # Pashto
                                'fa': 'per', # Persian
                                'pl': 'pol', # Polish
                                'pt': 'por', # Portuguese
                                'pa': 'pan', # Punjabi
                                'qu': 'que', # Quechua
                                'ro': 'rum', # Romanian
                                'ru': 'rus', # Russian
                                'sm': 'smo', # Samoan
                                'sa': 'san', # Sanskrit
                                'gd': 'gla', # Scots Gaelic
                                'nso': 'nso', # Sepedi
                                'sr': 'srp', # Serbian
                                'st': 'sot', # Sesotho
                                'sn': 'sna', # Shona
                                'sd': 'snd', # Sindhi
                                'si': 'sin', # Sinhala
                                'sk': 'slo', # Slovak
                                'sl': 'slv', # Slovenian
                                'so': 'som', # Somali
                                'es': 'spa', # Spanish
                                'su': 'sun', # Sundanese
                                'sw': 'swa', # Swahili
                                'sv': 'swe', # Swedish
                                'tg': 'tgk', # Tajik
                                'ta': 'tam', # Tamil
                                'tt': 'tat', # Tatar
                                'te': 'tel', # Telugu
                                'th': 'tha', # Thai
                                'ti': 'tir', # Tigrinya
                                'ts': 'tso', # Tsonga
                                'tr': 'tur', # Turkish
                                'tk': 'tuk', # Turkmen
                                'tw': 'twi', # Twi (Akan)
                                'uk': 'ukr', # Ukrainian
                                'ur': 'urd', # Urdu
                                'ug': 'uig', # Uyghur
                                'uz': 'uzb', # Uzbek
                                'vi': 'vie', # Vietnamese
                                'cy': 'wel', # Welsh
                                'xh': 'xho', # Xhosa
                                'yi': 'yid', # Yiddish
                                'yo': 'yor', # Yoruba
                                'zu': 'zul', # Zulu
                           }

    def get_code_of_name(self, name):
        return self.code_of_name[name]

    def get_code_of_ffmpeg_code(self, ffmpeg_code):
        return self.code_of_ffmpeg_code[ffmpeg_code]

    def get_name_of_code(self, code):
        return self.name_of_code[code]

    def get_name_of_ffmpeg_code(self, ffmpeg_code):
        return self.name_of_ffmpeg_code[ffmpeg_code]

    def get_ffmpeg_code_of_name(self, name):
        return self.ffmpeg_code_of_name[name]

    def get_ffmpeg_code_of_code(self, code):
        return self.ffmpeg_code_of_code[code]


class WavConverter:
    @staticmethod
    def which(program):
        def is_exe(file_path):
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

    @staticmethod
    def ffprobe_check():
        if WavConverter.which("ffprobe"):
            return "ffprobe"
        if WavConverter.which("ffprobe.exe"):
            return "ffprobe.exe"
        return None

    @staticmethod
    def ffmpeg_check():
        if WavConverter.which("ffmpeg"):
            return "ffmpeg"
        if WavConverter.which("ffmpeg.exe"):
            return "ffmpeg.exe"
        return None

    def __init__(self, channels=1, rate=48000, progress_callback=None, error_messages_callback=None):
        self.channels = channels
        self.rate = rate
        self.progress_callback = progress_callback
        self.error_messages_callback = error_messages_callback

    def __call__(self, media_filepath):
        temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

        if "\\" in media_filepath:
            media_filepath = media_filepath.replace("\\", "/")

        if not os.path.isfile(media_filepath):
            if self.error_messages_callback:
                self.error_messages_callback(f"The given file does not exist: {media_filepath}")
            else:
                print(f"The given file does not exist: {media_filepath}")
                raise Exception(f"Invalid file: {media_filepath}")

        if not self.ffprobe_check():
            if self.error_messages_callback:
                self.error_messages_callback("Cannot find ffprobe executable")
            else:
                print("Cannot find ffprobe executable")
                raise Exception("Dependency not found: ffprobe")

        if not self.ffmpeg_check():
            if self.error_messages_callback:
                self.error_messages_callback("Cannot find ffmpeg executable")
            else:
                print("Cannot find ffmpeg executable")
                raise Exception("Dependency not found: ffmpeg")

        ffmpeg_command = [
                            'ffmpeg',
                            '-hide_banner',
                            '-loglevel', 'error',
                            '-v', 'error',
                            '-y',
                            '-i', media_filepath,
                            '-ac', str(self.channels),
                            '-ar', str(self.rate),
                            '-progress', '-', '-nostats',
                            temp.name
                         ]

        try:
            media_file_display_name = os.path.basename(media_filepath).split('/')[-1]
            info = f"Converting '{media_file_display_name}' to a temporary WAV file"
            start_time = time.time()

            ffprobe_command = [
                                'ffprobe',
                                '-hide_banner',
                                '-v', 'error',
                                '-loglevel', 'error',
                                '-show_entries',
                                'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1',
                                media_filepath
                              ]

            ffprobe_process = None
            if sys.platform == "win32":
                ffprobe_process = subprocess.check_output(ffprobe_command, stdin=open(os.devnull), universal_newlines=True, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                ffprobe_process = subprocess.check_output(ffprobe_command, stdin=open(os.devnull), universal_newlines=True)

            total_duration = float(ffprobe_process.strip())

            process = None
            if sys.platform == "win32":
                process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            while True:
                if process.stdout is None:
                    continue

                stderr_line = (process.stdout.readline().decode("utf-8", errors="replace").strip())
 
                if stderr_line == '' and process.poll() is not None:
                    break

                if "out_time=" in stderr_line:
                    time_str = stderr_line.split('time=')[1].split()[0]
                    current_duration = sum(float(x) * 1000 * 60 ** i for i, x in enumerate(reversed(time_str.split(":"))))

                    if current_duration>0 and current_duration<=total_duration*1000:
                        percentage = int(current_duration*100/(int(float(total_duration))*1000))
                        if self.progress_callback and percentage <= 100:
                            self.progress_callback(info, media_file_display_name, percentage, start_time)

            if self.progress_callback:
                self.progress_callback(info, media_file_display_name, 100, start_time)

            temp.close()

            return temp.name, self.rate

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return


class VoskRecognizer:
    def __init__(self, loglevel=-1, language_code="en", block_size=1024, progress_callback=None,  error_messages_callback=None):
        self.loglevel = loglevel
        self.language_code = language_code
        self.block_size = block_size
        self.progress_callback = progress_callback
        self.error_messages_callback = error_messages_callback

    def __call__(self, wav_filepath):
        try:
            SetLogLevel(self.loglevel)
            reader = wave.open(wav_filepath)
            rate = reader.getframerate()
            total_duration = reader.getnframes() / rate
            vosk_language = VoskLanguage()
            model = Model(lang=vosk_language.lang_of_code[self.language_code])
            rec = KaldiRecognizer(model, rate)
            rec.SetWords(True)
            regions = []
            transcripts = []
            media_file_display_name = os.path.basename(wav_filepath).split('/')[-1]
            info = f"Creating '{media_file_display_name}' transcriptions"
            start_time = time.time()
        
            while True:
                block = reader.readframes(self.block_size)
                if not block:
                    break
                if rec.AcceptWaveform(block):
                    recResult_json = json.loads(rec.Result())
                    if 'result' in recResult_json:
                        result = recResult_json["result"]
                        text = recResult_json["text"]
                        region_start_time = result[0]["start"]
                        region_end_time = result[len(result)-1]["end"]
                        progress = int(int(region_end_time)*100/total_duration)
                        regions.append((region_start_time, region_end_time))
                        transcripts.append(text)
                        if self.progress_callback:
                            self.progress_callback(info, media_file_display_name, progress, start_time)

            if self.progress_callback:
                self.progress_callback(info, media_file_display_name, 100, start_time)

            return regions, transcripts

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)


class SentenceTranslator(object):
    def __init__(self, src, dst, patience=-1, timeout=30, error_messages_callback=None):
        self.src = src
        self.dst = dst
        self.patience = patience
        self.timeout = timeout
        self.error_messages_callback = error_messages_callback

    def __call__(self, sentence):
        try:
            translated_sentence = []
            # handle the special case: empty string.
            if not sentence:
                return None
            translated_sentence = self.GoogleTranslate(sentence, src=self.src, dst=self.dst, timeout=self.timeout)
            fail_to_translate = translated_sentence[-1] == '\n'
            while fail_to_translate and patience:
                translated_sentence = self.GoogleTranslate(translated_sentence, src=self.src, dst=self.dst, timeout=self.timeout).text
                if translated_sentence[-1] == '\n':
                    if patience == -1:
                        continue
                    patience -= 1
                else:
                    fail_to_translate = False

            return translated_sentence

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return

    def GoogleTranslate(self, text, src, dst, timeout=30):
        url = 'https://translate.googleapis.com/translate_a/'
        params = 'single?client=gtx&sl='+src+'&tl='+dst+'&dt=t&q='+text;
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://translate.google.com',}

        try:
            response = requests.get(url+params, headers=headers, timeout=self.timeout)
            if response.status_code == 200:
                response_json = response.json()[0]
                length = len(response_json)
                translation = ""
                for i in range(length):
                    translation = translation + response_json[i][0]
                return translation
            return

        except requests.exceptions.ConnectionError:
            with httpx.Client() as client:
                response = client.get(url+params, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    response_json = response.json()[0]
                    length = len(response_json)
                    translation = ""
                    for i in range(length):
                        translation = translation + response_json[i][0]
                    return translation
                return

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return


class SubtitleFormatter:
    supported_formats = ['srt', 'vtt', 'json', 'raw']

    def __init__(self, format_type, error_messages_callback=None):
        self.format_type = format_type.lower()
        self.error_messages_callback = error_messages_callback
        
    def __call__(self, subtitles, padding_before=0, padding_after=0):
        try:
            if self.format_type == 'srt':
                return self.srt_formatter(subtitles, padding_before, padding_after)
            elif self.format_type == 'vtt':
                return self.vtt_formatter(subtitles, padding_before, padding_after)
            elif self.format_type == 'json':
                return self.json_formatter(subtitles)
            elif self.format_type == 'raw':
                return self.raw_formatter(subtitles)
            else:
                if error_messages_callback:
                    error_messages_callback(f'Unsupported format type: {self.format_type}')
                else:
                    raise ValueError(f'Unsupported format type: {self.format_type}')

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return

    def srt_formatter(self, subtitles, padding_before=0, padding_after=0):
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

    def vtt_formatter(self, subtitles, padding_before=0, padding_after=0):
        """
        Serialize a list of subtitles according to the VTT format, with optional time padding.
        """
        text = self.srt_formatter(subtitles, padding_before, padding_after)
        text = 'WEBVTT\n\n' + text.replace(',', '.')
        return text

    def json_formatter(self, subtitles):
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

    def raw_formatter(self, subtitles):
        """
        Serialize a list of subtitles as a newline-delimited string.
        """
        return ' '.join(text for (_rng, text) in subtitles)


class SubtitleWriter:
    def __init__(self, regions, transcripts, format, error_messages_callback=None):
        self.regions = regions
        self.transcripts = transcripts
        self.format = format
        self.timed_subtitles = [(r, t) for r, t in zip(self.regions, self.transcripts) if t]
        self.error_messages_callback = error_messages_callback

    def get_timed_subtitles(self):
        return self.timed_subtitles

    def write(self, declared_subtitle_filepath):
        try:
            formatter = SubtitleFormatter(self.format)
            formatted_subtitles = formatter(self.timed_subtitles)
            saved_subtitle_filepath = declared_subtitle_filepath
            if saved_subtitle_filepath:
                subtitle_file_base, subtitle_file_ext = os.path.splitext(saved_subtitle_filepath)
                if not subtitle_file_ext:
                    saved_subtitle_filepath = "{base}.{format}".format(base=subtitle_file_base, format=self.format)
                else:
                    saved_subtitle_filepath = declared_subtitle_filepath
            with open(saved_subtitle_filepath, 'wb') as f:
                f.write(formatted_subtitles.encode("utf-8"))

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return


#----------------------------------------------------------- MISC FUNCTIONS ------------------------------------------------------------#


def change_code_page(code_page):
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(code_page)
    kernel32.SetConsoleCP(code_page)


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


def worker_recognize(src, dst, error_messages_callback=None):
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
    model = None

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
            #print(f"device_info = {device_info}")
            SampleRate = int(device_info["default_samplerate"])
            #print(f"SampleRate = {SampleRate}")

        if wav_filepath:
            dump_fn = open(wav_filepath, "wb")
        else:
            dump_fn = None

        vosk_language = VoskLanguage()
        model = Model(lang=vosk_language.lang_of_code[src])

        with sd.RawInputStream(samplerate=SampleRate, blocksize=8192, device=Device, dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, SampleRate)

            while recognizing:
                if not recognizing:
                    rec = None
                    data = None
                    result = None
                    results_text = None
                    partial_results_text = None
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

                            tmp_src_txt_transcription_filename = f"record.{src}.txt"
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
        #sys.exit(0)

    except Exception as e:
        if error_messages_callback:
            error_messages_callback(e)
        else:
            print(e)
        #sys.exit(type(e).__name__ + ": " + str(e))

'''
def GoogleTranslate(text, src, dst, timeout=30, error_messages_callback=None):
    url = 'https://translate.googleapis.com/translate_a/'
    params = 'single?client=gtx&sl='+src+'&tl='+dst+'&dt=t&q='+text;
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://translate.google.com',}

    try:
        response = requests.get(url+params, headers=headers, timeout=timeout)
        if response.status_code == 200:
            response_json = response.json()[0]
            length = len(response_json)
            translation = ""
            for i in range(length):
                translation = translation + response_json[i][0]
            return translation
        return

    except requests.exceptions.ConnectionError:
        with httpx.Client() as client:
            response = client.get(url+params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                response_json = response.json()[0]
                length = len(response_json)
                translation = ""
                for i in range(length):
                    translation = translation + response_json[i][0]
                return translation
            return

    except KeyboardInterrupt:
        if error_messages_callback:
            error_messages_callback("Cancelling all tasks")
        else:
            print("Cancelling all tasks")
        return

    except Exception as e:
        if error_messages_callback:
            error_messages_callback(e)
        else:
            print(e)
        return
'''

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

            sentence_translator = SentenceTranslator(src=src, dst=dst, error_messages_callback=show_error_messages)
            n_words = len(partial_result.split())
            if partial_result and n_words % 3 == 0:
                #translated_text = GoogleTranslate(partial_result, src, dst, error_messages_callback=show_error_messages)
                translated_text = sentence_translator(partial_result)
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
    global main_window, ffmpeg_start_run_time, first_streaming_duration_recorded, recognizing

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
    global main_window, recognizing, ffmpeg_start_run_time, first_streaming_duration_recorded

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


def stop_ffmpeg_windows(error_messages_callback=None):
    try:
        tasklist_output = subprocess.check_output(['tasklist'], creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8')
        ffmpeg_pid = None
        for line in tasklist_output.split('\n'):
            if "ffmpeg" in line:
                ffmpeg_pid = line.split()[1]
                break
        if ffmpeg_pid:
            devnull = open(os.devnull, 'w')
            subprocess.Popen(['taskkill', '/F', '/T', '/PID', ffmpeg_pid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)

    except KeyboardInterrupt:
        if error_messages_callback:
            error_messages_callback("Cancelling all tasks")
        else:
            print("Cancelling all tasks")
        return

    except Exception as e:
        if error_messages_callback:
            error_messages_callback(e)
        else:
            print(e)
        return


def stop_ffmpeg_linux(error_messages_callback=None):
    process_name = 'ffmpeg'
    try:
        output = subprocess.check_output(['ps', '-ef'])
        pid = [line.split()[1] for line in output.decode('utf-8').split('\n') if process_name in line][0]
        subprocess.call(['kill', '-9', str(pid)])
        #print(f"{process_name} has been killed")
    except IndexError:
        #print(f"{process_name} is not running")
        pass

    except KeyboardInterrupt:
        if error_messages_callback:
            error_messages_callback("Cancelling all tasks")
        else:
            print("Cancelling all tasks")
        return

    except Exception as e:
        if error_messages_callback:
            error_messages_callback(e)
        else:
            print(e)
        return


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


def extract_audio(media_filepath, n_channels=1, rate=48000):
    global main_window, not_transcribing

    def which(program):
        def is_exe(file_path):
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

    def ffprobe_check():
        if WavConverter.which("ffprobe"):
            return "ffprobe"
        if WavConverter.which("ffprobe.exe"):
            return "ffprobe.exe"
        return None

    def ffmpeg_check():
        if WavConverter.which("ffmpeg"):
            return "ffmpeg"
        if WavConverter.which("ffmpeg.exe"):
            return "ffmpeg.exe"
        return None

    if not_transcribing: return

    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

    if "\\" in media_filepath:
        media_filepath = media_filepath.replace("\\", "/")

    if not os.path.isfile(media_filepath):
        not_transcribing=True
        window_key = '-OUTPUT-MESSAGES-'
        msg = f"The given file does not exist: '{media_filepath}'"
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    if not ffprobe_check():
        not_transcribing=True
        window_key = '-OUTPUT-MESSAGES-'
        msg = 'ffprobe executable is not found'
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    if not ffmpeg_check():
        not_transcribing=True
        window_key = '-OUTPUT-MESSAGES-'
        msg = 'ffmpeg executable is not found'
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    ffmpeg_command = [
                        'ffmpeg',
                        '-hide_banner',
                        '-loglevel', 'error',
                        '-v', 'error',
                        '-y',
                        '-i', media_filepath,
                        '-ac', str(channels),
                        '-ar', str(rate),
                        '-progress', '-', '-nostats',
                        temp.name
                     ]

    media_file_display_name = os.path.basename(media_filepath).split('/')[-1]

    try:
        info = f"Converting '{media_file_display_name}' to a temporary WAV file"
        total = 100
        start_time = time.time()

        ffprobe_command = [
                            'ffprobe',
                            '-hide_banner',
                            '-v', 'error',
                            '-loglevel', 'error',
                            '-show_entries',
                            'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1',
                            media_filepath
                          ]

        ffprobe_process = None
        if sys.platform == "win32":
            ffprobe_process = subprocess.check_output(ffprobe_command, stdin=open(os.devnull), universal_newlines=True, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            ffprobe_process = subprocess.check_output(ffprobe_command, stdin=open(os.devnull), universal_newlines=True)

        total_duration = float(ffprobe_process.strip())

        process = None
        if sys.platform == "win32":
            process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        while True:
            if process.stdout is None:
                continue

            stderr_line = (process.stdout.readline().decode("utf-8", errors="replace").strip())
 
            if stderr_line == '' and process.poll() is not None:
                break

            if "out_time=" in stderr_line:
                time_str = stderr_line.split('time=')[1].split()[0]
                current_duration = sum(float(x) * 1000 * 60 ** i for i, x in enumerate(reversed(time_str.split(":"))))

                if current_duration>0 and current_duration<=total_duration*1000:
                    progress = int(current_duration*100/(int(float(total_duration))*1000))
                    percentage = f'{progress}%'
                    if progress > 0:
                        elapsed_time = time.time() - start_time
                        eta_seconds = (elapsed_time / progress) * (total - progress)
                    else:
                        eta_seconds = 0

                    eta_time = timedelta(seconds=int(eta_seconds))
                    eta_str = str(eta_time)
                    hour, minute, second = eta_str.split(":")
                    time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second
                    main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, percentage, progress, time_str))

        elapsed_time = time.time() - start_time
        elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
        elapsed_time_str = str(elapsed_time_seconds)
        hour, minute, second = elapsed_time_str.split(":")
        time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
        main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, percentage, progress, time_str))

    except Exception as e:
        not_transcribing = True
        msg = [e, n_channels, str(n_channels)]
        sg.Popup(msg, title="Error", line_width=50)
        main_window.write_event_value('-EXCEPTION-', e)

    if not_transcribing: return

    temp.close()

    return temp.name, rate


class NoConsoleProcess(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
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


def vosk_transcribe(src, dst, media_filepath, subtitle_format):
    global main_window, all_transcribe_threads, thread_vosk_transcribe, not_transcribing, regions, transcriptions, \
        created_regions, created_subtitles, translated_transcriptions, saved_src_subtitle_filepath, saved_dst_subtitle_filepath

    if not os.path.isfile(media_filepath):
        FONT = ("Helvetica", 10)
        sg.set_options(font=FONT)
        sg.Popup(f"'{media_filepath}' is not exist!", title="Error", line_width=50)
        return

    if not_transcribing: return

    media_file_display_name = os.path.basename(media_filepath).split('/')[-1]

    pool = multiprocessing.Pool(10, initializer=NoConsoleProcess)
    wav_filepath = None
    SampleRate = None
    src_subtitle_file = None
    dst_subtitle_file = None
    google_language = GoogleLanguage()

    if not_transcribing: return

    main_window.write_event_value('-EVENT-TRANSCRIBE-PREPARE-WINDOW-', True)
    main_window.write_event_value('-EVENT-TRANSCRIBE-SHOW-PROGRESS-BAR-', False)

    window_key = '-OUTPUT-MESSAGES-'
    msg = f"Processing '{media_file_display_name}' :\n"
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    window_key = '-OUTPUT-MESSAGES-'
    msg = f"Converting '{media_file_display_name}' to a temporary WAV file...\n"
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))
    main_window.write_event_value('-EVENT-TRANSCRIBE-SHOW-PROGRESS-BAR-', True)

    info = f"Converting '{media_file_display_name}' to a temporary WAV file"
    total = 100
    time_str = "ETA  : --:--:--"
    main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, "0%", 0, time_str))

    wav_converter = WavConverter(progress_callback=show_progress, error_messages_callback=show_error_messages)
    wav_filepath, SampleRate = wav_converter(media_filepath)

    window_key = '-OUTPUT-MESSAGES-'
    msg = f"'{media_file_display_name}' temporary WAV file saved as:\n  '{wav_filepath}'\n"
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))
    main_window.write_event_value('-EVENT-TRANSCRIBE-SHOW-PROGRESS-BAR-', True)

    if not_transcribing: return

    regions = []
    transcriptions = []
    created_regions = []
    created_subtitles = []
    translated_transcriptions = []

    if not_transcribing: return

    if os.path.isfile(wav_filepath):

        window_key = '-OUTPUT-MESSAGES-'
        msg = f"Creating '{media_file_display_name}' transcriptions...\n"
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

        vosk_recognizer = VoskRecognizer(loglevel=-1, language_code=src, block_size=4096, progress_callback=show_progress)
        regions, transcriptions = vosk_recognizer(wav_filepath)

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

        base, ext = os.path.splitext(media_filepath)
        src_subtitle_filepath = f"{base}.{src}.{subtitle_format}"

        if not_transcribing: return

        writer = SubtitleWriter(regions, transcriptions, subtitle_format, error_messages_callback=show_error_messages)
        writer.write(src_subtitle_filepath)

        if not_transcribing: return

        if (not is_same_language(src, dst)):
            timed_subtitles = writer.timed_subtitles

            dst_subtitle_filepath = f"{base}.{dst}.{subtitle_format}"

            if not_transcribing: return

            for entry in timed_subtitles:
                created_regions.append(entry[0])
                created_subtitles.append(entry[1])

            if not_transcribing: return

            window_key = '-OUTPUT-MESSAGES-'
            msg = f"Translating '{media_file_display_name}' subtitles from '{google_language.name_of_code[src]}' ('{src}') to '{google_language.name_of_code[dst]}' ('{dst}') ...\n"
            append_flag = True
            main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

            info = f"Translating '{media_file_display_name}' subtitles from '{google_language.name_of_code[src]}' ('{src}') to '{google_language.name_of_code[dst]}' ('{dst}')"
            total = 100
            time_str = "ETA  : --:--:--"

            start_time = time.time()
            
            main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, "0%", 0, time_str))

            transcript_translator = SentenceTranslator(src=src, dst=dst, error_messages_callback=show_error_messages)
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

                if progress > 0:
                    elapsed_time = time.time() - start_time
                    eta_seconds = (elapsed_time / progress) * (total - progress)
                else:
                    eta_seconds = 0

                eta_time = timedelta(seconds=int(eta_seconds))
                eta_str = str(eta_time)
                hour, minute, second = eta_str.split(":")
                time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second
                main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, percentage, progress, time_str))

            elapsed_time = time.time() - start_time
            elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
            elapsed_time_str = str(elapsed_time_seconds)
            hour, minute, second = elapsed_time_str.split(":")
            time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
            main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, "100%", 100, time_str))

            if not_transcribing: return

            translation_writer = SubtitleWriter(created_regions, translated_transcription, subtitle_format, error_messages_callback=show_error_messages)
            translation_writer.write(dst_subtitle_filepath)

            if not_transcribing: return

        window_key = '-OUTPUT-MESSAGES-'
        msg = f"'{media_file_display_name}' temporary subtitles file saved as :\n  '{src_subtitle_filepath}'\n"
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

        if (not is_same_language(src, dst)):
            window_key = '-OUTPUT-MESSAGES-'
            msg = f"'{media_file_display_name}' temporary translated subtitles file saved as :\n  '{dst_subtitle_filepath}'\n"
            append_flag = True
            main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

        if not_transcribing: return

    window_key = '-OUTPUT-MESSAGES-'
    msg = f"Saving '{media_file_display_name}' temporary transcriptions text files...\n"
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    # SAVING TEMPORARY TRANSCRIPTIONS record.txt AND TRANSLATED TRANSCRIPTIONS record.traslated.txt
    tmp_src_txt_transcription_filename = f"record.{src}.txt"
    tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
    tmp_dst_txt_transcription_filename = f"record.{dst}.txt"
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
    msg = f"'{media_file_display_name}' temporary transcriptions text file saved as :\n  '{tmp_src_txt_transcription_filepath}'\n"
    append_flag = True
    main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    if (not is_same_language(src, dst)):
        window_key = '-OUTPUT-MESSAGES-'
        msg = f"'{media_file_display_name}' temporary translated transcriptions text file saved as :\n  '{tmp_dst_txt_transcription_filepath}'\n"
        append_flag = True
        main_window.write_event_value('-EVENT-TRANSCRIBE-SEND-MESSAGES-', (window_key, msg, append_flag))

    window_key = '-OUTPUT-MESSAGES-'
    msg = "Done."
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
    global main_window, thread_save_temporary_transcriptions, regions, transcriptions, \
        created_regions, created_subtitles, translated_transcriptions

    tmp_src_subtitle_filename = f"record.{src}.{subtitle_format}"
    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)

    writer = SubtitleWriter(regions, transcriptions, subtitle_format, error_messages_callback=show_error_messages)
    writer.write(tmp_src_subtitle_filepath)

    # SAVING REGIONS AND TRANSCRIPTIONS VALUES IN JSON FORMAT
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

    # SAVING TEMPORARY TRANSLATED TRANSCRIPTIONS AS FORMATTED SUBTITLE FILE
    timed_subtitles = writer.timed_subtitles
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
        start_time = time.time()
        time_str = "ETA  : --:--:--"

        main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, "0%", 0, time_str))

        transcript_translator = SentenceTranslator(src=src, dst=dst)
        translated_transcriptions = []
        pool = multiprocessing.Pool(10, initializer=NoConsoleProcess)

        for i, translated_transcription in enumerate(pool.imap(transcript_translator, created_subtitles)):
            translated_transcriptions.append(translated_transcription)
            progress = int(i*100/len(created_subtitles))
            percentage = f'{progress}%'
            if progress > 0:
                elapsed_time = time.time() - start_time
                eta_seconds = (elapsed_time / progress) * (total - progress)
            else:
                eta_seconds = 0
            eta_time = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta_time)
            hour, minute, second = eta_str.split(":")
            time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second
            main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, percentage, progress, time_str))

            if progress == total:
                elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
                elapsed_time_str = str(elapsed_time_seconds)
                hour, minute, second = elapsed_time_str.split(":")
                time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
                main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, "100%", total, time_str))

        elapsed_time = time.time() - start_time
        elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
        elapsed_time_str = str(elapsed_time_seconds)
        hour, minute, second = elapsed_time_str.split(":")
        time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
        main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, "100%", total, time_str))

        pool.close()
        pool.join()
        pool = None

        tmp_dst_subtitle_filename = f"record.{dst}.{subtitle_format}"
        tmp_dst_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_subtitle_filename)

        translation_writer = SubtitleWriter(created_regions, translated_transcriptions, subtitle_format, error_messages_callback=show_error_messages)
        translation_writer.write(tmp_dst_subtitle_filepath)

        # SAVING TRANSLATED TRANSCRIPTIONS VALUES IN JSON FORMAT
        translated_transcriptions_filename = "translated_transcriptions"
        translated_transcriptions_filepath = os.path.join(tempfile.gettempdir(), translated_transcriptions_filename)
        translated_transcriptions_file = open(translated_transcriptions_filepath, "w")
        json.dump(translated_transcriptions, translated_transcriptions_file)
        translated_transcriptions_file.close()

   # SAVING TEMPORARY TRANSLATED TRANSCRIPTION AS TEXT FILE
    tmp_dst_txt_transcription_filename = f"record.{dst}.txt"
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


def save_as_declared_subtitle_filename(declared_src_subtitle_filename, src, dst):

    regions_filename = "regions"
    regions_filepath = os.path.join(tempfile.gettempdir(), regions_filename)

    transcriptions_filename = "transcriptions"
    transcriptions_filepath = os.path.join(tempfile.gettempdir(), transcriptions_filename)

    translated_transcriptions_filename = "translated_transcriptions"
    translated_transcriptions_filepath = os.path.join(tempfile.gettempdir(), translated_transcriptions_filename)

    regions = None
    transcriptions = None
    translated_transcriptions = None

    transcribe_is_done_filename = "transcribe_is_done"
    transcribe_is_done_filepath = os.path.join(tempfile.gettempdir(), transcribe_is_done_filename)

    tmp_src_txt_transcription_filename = f"record.{src}.txt"
    tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)

    tmp_dst_txt_transcription_filename = f"record.{dst}.txt"
    tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)

    declared_src_subtitle_file_base, declared_src_subtitle_file_extension = os.path.splitext(declared_src_subtitle_filename)
    declared_src_subtitle_file_extension = declared_src_subtitle_file_extension[1:]

    
    while True:
        if os.path.isfile(regions_filepath):
            regions_file = open(regions_filepath, "r")
            regions = tuple(json.load(regions_file))
            regions_file.close()

        if os.path.isfile(transcriptions_filepath):
            transcriptions_file = open(transcriptions_filepath, "r")
            transcriptions = tuple(json.load(transcriptions_file))
            transcriptions_file.close()

        if declared_src_subtitle_file_extension and declared_src_subtitle_file_extension in SubtitleFormatter.supported_formats and len(regions)>0 and len(transcriptions)>0:

            declared_src_subtitle_filepath = f"{declared_src_subtitle_file_base}.{declared_src_subtitle_file_extension}"

            writer = SubtitleWriter(regions, transcriptions, subtitle_format, error_messages_callback=show_error_messages)
            writer.write(declared_src_subtitle_filepath)

            timed_subtitles = writer.timed_subtitles
            created_regions = []
            created_subtitles = []
            for entry in timed_subtitles:
                created_regions.append(entry[0])
                created_subtitles.append(entry[1])

        else:
            FONT = ("Helvetica", 10)
            sg.set_options(font=FONT)
            sg.Popup("Subtitle format you typed is not supported, subtitle file will be saved in TXT format.", title="Info", line_width=50)
            declared_src_subtitle_filepath = f"{declared_src_subtitle_file_base}.{src}.txt"
            shutil.copy(tmp_src_txt_transcription_filepath, declared_src_subtitle_filepath)

        # SAVING TRANSLATED SUBTITLE FILE WITH FILENAME BASED ON SUBTITLE FILENAME DECLARED IN COMMAND LINE ARGUMENTS
        if not is_same_language(src, dst):

            if os.path.isfile(translated_transcriptions_filepath):
                translated_transcriptions_file = open(translated_transcriptions_filepath, "r")
                translated_transcriptions = tuple(json.load(translated_transcriptions_file))
                translated_transcriptions_file.close()

            declared_dst_subtitle_filepath = f"{declared_src_subtitle_file_base}.{dst}.{declared_src_subtitle_file_extension}"

            if declared_src_subtitle_file_extension and declared_src_subtitle_file_extension in SubtitleFormatter.supported_formats and len(created_regions)>0 \
                    and len(translated_transcriptions)>0:

                translation_writer = SubtitleWriter(created_regions, translated_transcriptions, subtitle_format, error_messages_callback=show_error_messages)
                translation_writer.write(declared_dst_subtitle_filepath)

            else:
                shutil.copy(tmp_dst_txt_transcription_filepath, declared_dst_subtitle_filepath)

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


def download_vosk_model(src, url, folder):
    global main_window

    # Create the specified folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Extract the filename from the URL
    filename = os.path.basename(url)

    # Specify the path where the file will be saved
    save_path = os.path.join(folder, filename)

    # Create a progress bar widget
    vosk_language = VoskLanguage()
    info = f"Downloading '{vosk_language.name_of_code[src]}' vosk model"
    total = 100
    time_str = "ETA  : --:--:--"
    start_time = time.time()

    main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, "0%", 0, time_str))

    def progress_hook(block_count, block_size, total_size):
        progress = int(100*block_count*block_size/total_size)
        percentage = f'{progress}%'
        if progress > 0:
            elapsed_time = time.time() - start_time
            eta_seconds = (elapsed_time / progress) * (total - progress)
        else:
            eta_seconds = 0
        eta_time = timedelta(seconds=int(eta_seconds))
        eta_str = str(eta_time)
        hour, minute, second = eta_str.split(":")
        time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second
        main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, percentage, progress, time_str))
        if progress == total:
            elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
            elapsed_time_str = str(elapsed_time_seconds)
            hour, minute, second = elapsed_time_str.split(":")
            time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
            main_window.write_event_value('-EVENT-UPDATE-GENERIC-PROGRESS-BAR-', (info, total, "100%", total, time_str))

    # Start the download with progress
    urlretrieve(url, save_path, progress_hook)

    with ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(folder)
    os.remove(save_path)



#----------------------------------------------------------- GUI FUNCTIONS ------------------------------------------------------------#

def handle_focus(event):
    if event.widget == window:
        window.focus_set()


def steal_focus():
    global main_window, overlay_translation_window

    if overlay_translation_window:
        overlay_translation_window.close()
        overlay_translation_window=None

    if(sys.platform == "win32"):
        main_window.TKroot.attributes('-topmost', True)
        main_window.TKroot.attributes('-topmost', False)
        main_window.TKroot.deiconify()

    else:
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
    FONT = ("Helvetica", 10)
    sg.set_options(font=FONT)

    layout = [
        [sg.Text(text, size=(50,1))],
        [sg.Push(), sg.Button('Yes'), sg.Button('No')],
    ]
    return sg.Window(title if title else text, layout, resizable=True).read(close=True)


def make_progress_bar_window(info, total):

    FONT = ("Helvetica", 10)
    sg.set_options(font=FONT)

    layout = [
                [sg.Text(info, key='-INFO-')],
                [
                    sg.ProgressBar(total, orientation='h', size=(40, 10), key='-PROGRESS-'),
                    sg.Text(size=(5,1), key='-PERCENTAGE-'),
                    sg.Text(f"ETA  : --:--:--", size=(14, 1), expand_x=False, expand_y=False, key='-ETA-', justification='r')
                ]
             ]

    progress_bar_window = sg.Window("Progress", layout, no_titlebar=False, finalize=True)
    progress_bar_window['-PROGRESS-'].update(max=total)
    move_center(progress_bar_window)

    return progress_bar_window


def show_progress(info, media_file_display_name, progress, start_time):
    global main_window

    total = 100
    percentage = f'{progress}%'
    if progress > 0:
        elapsed_time = time.time() - start_time
        eta_seconds = (elapsed_time / progress) * (total - progress)
    else:
        eta_seconds = 0

    eta_time = timedelta(seconds=int(eta_seconds))
    eta_str = str(eta_time)
    hour, minute, second = eta_str.split(":")
    time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second
    main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, percentage, progress, time_str))
    if progress == total:
        elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
        elapsed_time_str = str(elapsed_time_seconds)
        hour, minute, second = elapsed_time_str.split(":")
        time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
        main_window.write_event_value('-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-', (media_file_display_name, info, total, "100%", total, time_str))


def show_error_messages(messages):
    global main_window, not_transcribing

    not_transcribing = True
    main_window.write_event_value("-EXCEPTION-", messages)


def make_transcribe_window(info, total):
    global not_transcribing

    FONT = ("Helvetica", 10)
    sg.set_options(font=FONT)

    layout = [
                [sg.Text("File to procees", size=(50,1), expand_x=False, expand_y=False, key='-FILE-DISPLAY-NAME-')],
                [sg.Text("Progress info", key='-INFO-')],
                [
                    sg.ProgressBar(100, orientation='h', size=(40, 10), key='-PROGRESS-'),
                    sg.Text("0%", size=(5,1), key='-PERCENTAGE-'),
                    sg.Text(f"ETA  : --:--:--", size=(14, 1), expand_x=False, expand_y=False, key='-ETA-', justification='r')
                ],
                [sg.Multiline(size=(50, 10), expand_x=True, expand_y=True, key='-OUTPUT-MESSAGES-')],
                [sg.Button('Cancel', font=FONT, expand_x=True, expand_y=True, key='-CANCEL-')]
             ]

    transcribe_window = sg.Window('Transcribe', layout, font=FONT, resizable=True, keep_on_top=True, finalize=True)
    transcribe_window['-PROGRESS-'].update(max=total)
    move_center(transcribe_window)

    return transcribe_window


def make_overlay_translation_window(translated_text):
    xmax, ymax = sg.Window.get_screen_size()

    max_columns = int(os.environ.get('COLUMNS', 80))
    max_lines = int(os.environ.get('LINES', 24))

    columns = max_columns
    rows = max_lines

    FONT = ("Helvetica", 16)
    sg.set_options(font=FONT)

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

    #window_height = int(nl*LINE_SIZE)
    window_height = int((nl+1)*LINE_SIZE)
    window_x=int((xmax-window_width)/2)

    if nl == 1:
        window_y = int(528*ymax/720)
    else:
        window_y = int((528-(nl-1)*28)*ymax/720)

    if xmax > 1280: FONT_SIZE = 14
    if xmax <= 1280: FONT_SIZE = 16

    FONT_TYPE = "Helvetica"
    sg.set_options(font=FONT)

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
    window.refresh()


def get_clipboard_text():
    try:
        clipboard_data = subprocess.check_output(['xclip', '-selection', 'clipboard', '-o'], universal_newlines=True)
        return clipboard_data.strip()
    except subprocess.CalledProcessError:
        # Handle the case when clipboard is empty or unsupported
        return None


def set_clipboard_text(text):
    try:
        subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode())
    except subprocess.CalledProcessError as e:
        pass


#------------------------------------------------------------ MAIN PROGRAM ------------------------------------------------------------#


def main():
    global main_window, thread_recognize, thread_timed_translate, thread_record_streaming, text, recognizing, \
        Device, SampleRate, wav_filepath, tmp_recorded_streaming_filepath, regions, transcriptions, \
        created_regions, created_subtitles, translated_transcriptions, start_button_click_time, ffmpeg_start_run_time, \
        get_tmp_recorded_streaming_filepath_time, first_streaming_duration_recorded, is_valid_url_streaming, \
        transcribe_window, thread_vosk_transcribe, all_transcribe_threads, not_transcribing, \
        saved_src_subtitle_filepath, saved_dst_subtitle_filepath


#------------------------------------------------------------- GUI PARTS -------------------------------------------------------------#


    xmax, ymax = sg.Window.get_screen_size()
    main_window_width = int(0.5*xmax)
    main_window_height = int(0.5*ymax)
    multiline_width = int(0.15*main_window_width)
    multiline_height = int(0.0125*main_window_height)
    FONT = ("Helvetica", 10)
    sg.set_options(font=FONT)

    vosk_language = VoskLanguage()
    google_language = GoogleLanguage()

    layout =  [
                [sg.Frame('Hints',[
                                    [sg.Text('Click \"Start\" button to start listening and printing subtitles on screen.', expand_x=True, expand_y=True)],
                                    [sg.Text('Paste the streaming URL into URL inputbox then check the \"Record Streaming\" checkbox to record the streaming.', expand_x=True, expand_y=True)],
                                  ],
                                  border_width=2, expand_x=True, expand_y=True)
                ],
                [
                    sg.Text('URL', size=(12, 1)),
                    sg.Input(size=(16, 1), expand_x=True, expand_y=True, key='-URL-', enable_events=True, right_click_menu=['&Edit', ['&Copy','&Paste',]]),
                    sg.Checkbox("Record Streaming", key='-RECORD-STREAMING-', enable_events=True)
                ],
                [
                    sg.Text('Thread status', size=(12, 1)),
                    sg.Text('NOT RECORDING', size=(20, 1), background_color='green1', text_color='black', expand_x=True, expand_y=True, key='-RECORD-STREAMING-STATUS-'),
                    sg.Text('Duration recorded', size=(16, 1)),
                    sg.Text('0:00:00.000000', size=(14, 1), background_color='green1', text_color='black', expand_x=True, expand_y=True, key='-STREAMING-DURATION-RECORDED-'),
                    sg.Text('', size=(16, 1), expand_x=True, expand_y=True)
                ],
                [sg.Text('', expand_x=True, expand_y=True),
                 sg.Button('SAVE RECORDED STREAMING', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-RECORDED-STREAMING-'),
                 sg.Text('', expand_x=True, expand_y=True)],
                [sg.Text('Audio language', size=(10, 1), expand_x=True, expand_y=True),
                 sg.Combo(list(vosk_language.code_of_name), size=(12, 1), default_value="English", enable_events=True, expand_x=True, expand_y=True, key='-SRC-')],
                [sg.Multiline(size=(96, 5), expand_x=True, expand_y=True, right_click_menu=['&Edit', ['&Copy','&Paste',]], key='-ML-SRC-RESULTS-')],
                [sg.Multiline(size=(96, 3), expand_x=True, expand_y=True, right_click_menu=['&Edit', ['&Copy','&Paste',]], key='-ML-SRC-PARTIAL-RESULTS-')],
                [sg.Text('', expand_x=True, expand_y=True),
                 sg.Button('SAVE TRANSCRIPTION', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-SRC-TRANSCRIPTION-'),
                 sg.Text('', expand_x=True, expand_y=True)],
                [sg.Text('Translation language', size=(10, 1),expand_x=True, expand_y=True),
                 sg.Combo(list(google_language.code_of_name), size=(12, 1), default_value="Indonesian", enable_events=True, expand_x=True, expand_y=True, key='-DST-')],
                [sg.Multiline(size=(96, 5), expand_x=True, expand_y=True, right_click_menu=['&Edit', ['&Copy','&Paste',]], key='-ML-DST-RESULTS-')],
                [sg.Multiline(size=(96, 3), expand_x=True, expand_y=True, right_click_menu=['&Edit', ['&Copy','&Paste',]], key='-ML-DST-PARTIAL-RESULTS-')],
                [sg.Text('', expand_x=True, expand_y=True, key='-SPACES3-'),
                 sg.Button('SAVE TRANSLATED TRANSCRIPTION', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-DST-TRANSCRIPTION-'),
                 sg.Text('', expand_x=True, expand_y=True, key='-SPACES4-')],
                [sg.Button('Start', expand_x=True, expand_y=True, button_color=('white', '#283b5b'), key='-START-BUTTON-'),
                 sg.Button('Exit', expand_x=True, expand_y=True)],
            ]


#--------------------------------------------------------- NON GUI PARTS -------------------------------------------------------------#

    # VOSK LogLevel DISABLED
    SetLogLevel(-1)

    if sys.platform == "win32":
        change_code_page(65001)
        stop_ffmpeg_windows(error_messages_callback=show_error_messages)
    else:
        stop_ffmpeg_linux(error_messages_callback=show_error_messages)

    last_selected_src = None
    last_selected_dst = None

    src_filename = "src"
    src_filepath = os.path.join(tempfile.gettempdir(), src_filename)
    if os.path.isfile(src_filepath):
        src_file = open(src_filepath, "r")
        last_selected_src = src_file.read()

    if last_selected_src:
        src = last_selected_src
    else:
        src = "en"

    dst_filename = "dst"
    dst_filepath = os.path.join(tempfile.gettempdir(), dst_filename)
    if os.path.isfile(dst_filepath):
        dst_file = open(dst_filepath, "r")
        last_selected_dst = dst_file.read()

    if last_selected_dst:
        dst = last_selected_dst
    else:
        dst = "id"

    subtitle_format = "srt"

    recognizing = False
    not_transcribing = True

    tmp_recorded_streaming_filename = "record.mp4"
    tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)
    if os.path.isfile(tmp_recorded_streaming_filepath): os.remove(tmp_recorded_streaming_filepath)

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

    video_file_types = [
        ('MP4 Files', '*.mp4'),
    ]

    tmp_src_txt_transcription_filename = f"record.{src}.txt"
    tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
    if os.path.isfile(tmp_src_txt_transcription_filepath): os.remove(tmp_src_txt_transcription_filepath)

    tmp_dst_txt_transcription_filename = f"record.{dst}.txt"
    tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)
    if os.path.isfile(tmp_dst_txt_transcription_filepath): os.remove(tmp_dst_txt_transcription_filepath)

    tmp_src_subtitle_filename = f"record.{src}.{subtitle_format}"
    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)
    if os.path.isfile(tmp_src_subtitle_filepath): os.remove(tmp_src_subtitle_filepath)

    tmp_dst_subtitle_filename = f"record.{dst}.{subtitle_format}"
    tmp_dst_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_subtitle_filename)
    if os.path.isfile(tmp_dst_subtitle_filepath): os.remove(tmp_dst_subtitle_filepath)

    regions_filename = "regions"
    regions_filepath = os.path.join(tempfile.gettempdir(), regions_filename)
    if os.path.isfile(regions_filepath): os.remove(regions_filepath)

    transcriptions_filename = "transcriptions"
    transcriptions_filepath = os.path.join(tempfile.gettempdir(), transcriptions_filename)
    if os.path.isfile(transcriptions_filepath): os.remove(transcriptions_filepath)

    translated_transcriptions_filename = "translated_transcriptions"
    translated_transcriptions_filepath = os.path.join(tempfile.gettempdir(), translated_transcriptions_filename)
    if os.path.isfile(translated_transcriptions_filepath): os.remove(translated_transcriptions_filepath)

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

    parser.add_argument('-lls', '--list-src-languages', help="List all available source languages", action='store_true')
    parser.add_argument('-lld', '--list-dst-languages', help="List all available destination languages", action='store_true')

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
    parser.add_argument("-r", "--samplerate", type=int, help="sampling rate in Hertz for example 8000, 16000, 44100, or 48000", default=48000)

    parser.add_argument("-u", "--url", type=str, metavar="URL", help="URL of live streaming if you want to record the streaming")
    parser.add_argument("-vf", "--video-filename", type=str, metavar="VIDEO_FILENAME", help="video file to store recording to", default=None)

    parser.add_argument('-v', '--version', action='version', version=VERSION)

    args = parser.parse_args(remaining)
    args = parser.parse_args()

    if args.src_language not in vosk_language.name_of_code.keys():
        print("Audio language not supported. Run with --list-languages-src to see all supported source languages.")
        parser.exit(0)
    else:
        src = args.src_language
        last_selected_src = src
        src_file = open(src_filepath, "w")
        src_file.write(src)
        src_file.close()

    if args.dst_language not in google_language.name_of_code.keys():
        print("Destination language not supported. Run with --list-languages-dst to see all supported destination languages.")
        parser.exit(0)
    else:
        dst = args.dst_language
        last_selected_dst = dst
        dst_file = open(dst_filepath, "w")
        dst_file.write(dst)
        dst_file.close()

    if args.list_src_languages:
        print("List of all source languages:")
        for code, language in sorted(vosk_language.name_of_code.items()):
            print("{code}\t{language}".format(code=code, language=language))
        parser.exit(0)

    if args.list_dst_languages:
        print("List of all destination languages:")
        for code, language in sorted(google_language.name_of_code.items()):
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


    subtitle_format = args.subtitle_format

    if subtitle_format not in SubtitleFormatter.supported_formats:
        FONT = ("Helvetica", 10)
        sg.set_options(font=FONT)
        sg.Popup("Subtitle format is not supported, you can choose it later when you save transcriptions.", title="Info", line_width=50)

    if args.video_filename:
        saved_recorded_streaming_basename, saved_recorded_streaming_extension = os.path.splitext(os.path.basename(args.video_filename))
        saved_recorded_streaming_extension = saved_recorded_streaming_extension[1:]
        if saved_recorded_streaming_extension != "mp4":
            saved_recorded_streaming_filename = "{base}.{format}".format(base=saved_recorded_streaming_basename, format="mp4")
        else:
            saved_recorded_streaming_filename = args.video_filename


#------------------------------------ REMAINING GUI PARTS NEEDED TO LOAD AT LAST BEFORE MAIN LOOP ------------------------------------#


    main_window = sg.Window('PyVOSKLiveSubtitles-'+VERSION, layout, resizable=True, keep_on_top=True, finalize=True)
    main_window['-SRC-'].block_focus()
    main_window.bind("<Button-3>", "right_click")
    main_window['-URL-'].bind("<FocusIn>", "FocusIn")
    main_window['-ML-SRC-RESULTS-'].bind("<FocusIn>", "FocusIn")
    main_window['-ML-SRC-PARTIAL-RESULTS-'].bind("<FocusIn>", "FocusIn")
    main_window['-ML-DST-RESULTS-'].bind("<FocusIn>", "FocusIn")
    main_window['-ML-DST-PARTIAL-RESULTS-'].bind("<FocusIn>", "FocusIn")

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

    FONT = ("Helvetica", 10)
    sg.set_options(font=FONT)

    transcribe_window = None

    progressbar = make_progress_bar_window("", 100)
    progressbar.Hide()

    main_window['-URL-'].update("")
    main_window['-RECORD-STREAMING-'].update(False)

    if args.src_language:
        combo_src = vosk_language.name_of_code[args.src_language]
        main_window['-SRC-'].update(combo_src)

    if args.dst_language:
        combo_dst = google_language.name_of_code[args.dst_language]
        main_window['-DST-'].update(combo_dst)

    if args.url:
        is_valid_url_streaming = is_streaming_url(args.url)
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

    # IF VOSK MODEL IS NOT EXIST THEN WE DOWNLOAD THE MODEL
    vosk_cache_dir = None
    if sys.platform == "win32":
        vosk_cache_dir = os.path.expanduser('~\\') + '.cache' + '\\' + 'vosk'
    elif sys.platform == "linux":
        vosk_cache_dir = os.path.expanduser('~/.cache/vosk')
    elif sys.platform == "darwin":
        vosk_cache_dir = os.path.expanduser('~/Library/Caches/vosk')

    vosk_model_dir = vosk_cache_dir + os.sep + vosk_language.model_of_code[src]

    if not os.path.isdir(vosk_model_dir):
        FONT = ("Helvetica", 10)
        sg.set_options(font=FONT)
        sg.Popup(f"Vosk model for '{vosk_language.name_of_code[src]}' language is not exist, please wait until download progress completed", title="Info", line_width=50)

        os.makedirs(vosk_cache_dir, exist_ok=True)
        url = MODEL_PRE_URL + vosk_language.model_of_code[src] + ".zip"
        filename = os.path.basename(url)
        save_path = os.path.join(vosk_cache_dir, filename)

        info = f"Downloading '{vosk_language.name_of_code[src]}' vosk model"
        total = 100
        download_progressbar = make_progress_bar_window(info, total)
        start_time = time.time()

        def progress_hook(block_count, block_size, total_size):
            progress = int(100*block_count*block_size/total_size)
            percentage = f'{progress}%'

            if progress > 0:
                elapsed_time = time.time() - start_time
                eta_seconds = (elapsed_time / progress) * (total - progress)
            else:
                eta_seconds = 0

            eta_time = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta_time)
            hour, minute, second = eta_str.split(":")
            time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second

            download_progressbar['-INFO-'].update(info)
            download_progressbar['-PERCENTAGE-'].update(percentage)
            download_progressbar['-PROGRESS-'].update(progress)
            download_progressbar['-ETA-'].update(time_str)

            if progress == total:
                elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
                elapsed_time_str = str(elapsed_time_seconds)
                hour, minute, second = elapsed_time_str.split(":")
                time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
                download_progressbar['-INFO-'].update(info)
                download_progressbar['-PERCENTAGE-'].update(percentage)
                download_progressbar['-PROGRESS-'].update(progress)
                download_progressbar['-ETA-'].update(time_str)
                download_progressbar.refresh()
                time.sleep(1)
                download_progressbar.close()

        try:
            urlretrieve(url, save_path, progress_hook)
        except Exception as e:
            sg.Popup(e, title="Error", line_width=50)

        with ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(vosk_cache_dir)
        os.remove(save_path)

        #url = MODEL_PRE_URL + vosk_language.model_of_code[src] + ".zip"
        #download_vosk_model(src, url, vosk_cache_dir)


#-------------------------------------------------------------- MAIN LOOP -------------------------------------------------------------#


    while True:
        window, event, values = sg.read_all_windows()

        src = vosk_language.code_of_name[str(main_window['-SRC-'].get())]
        last_selected_src = src
        src_file = open(src_filepath, "w")
        src_file.write(src)
        src_file.close()

        dst = google_language.code_of_name[str(main_window['-DST-'].get())]
        last_selected_dst = dst
        dst_file = open(dst_filepath, "w")
        dst_file.write(dst)
        dst_file.close()

        vosk_model_dir = vosk_cache_dir + os.sep + vosk_language.model_of_code[src]
        if not os.path.isdir(vosk_model_dir):
            FONT = ("Helvetica", 10)
            sg.set_options(font=FONT)
            sg.Popup(f"Vosk model for '{vosk_language.name_of_code[src]}' language is not exist, please wait until download progress completed", title="Info", line_width=50)

            os.makedirs(vosk_cache_dir, exist_ok=True)
            url = MODEL_PRE_URL + vosk_language.model_of_code[src] + ".zip"
            filename = os.path.basename(url)
            save_path = os.path.join(vosk_cache_dir, filename)

            info = f"Downloading '{vosk_language.name_of_code[src]}' vosk model"
            total = 100
            download_progressbar = make_progress_bar_window(info, total)
            start_time = time.time()

            def progress_hook(block_count, block_size, total_size):
                progress = int(100*block_count*block_size/total_size)
                percentage = f'{progress}%'

                if progress > 0:
                    elapsed_time = time.time() - start_time
                    eta_seconds = (elapsed_time / progress) * (total - progress)
                else:
                    eta_seconds = 0
                eta_time = timedelta(seconds=int(eta_seconds))
                eta_str = str(eta_time)
                hour, minute, second = eta_str.split(":")
                time_str = "ETA  : " + hour.zfill(2) + ":" + minute + ":" + second

                download_progressbar['-INFO-'].update(info)
                download_progressbar['-PERCENTAGE-'].update(percentage)
                download_progressbar['-PROGRESS-'].update(progress)
                download_progressbar['-ETA-'].update(time_str)

                if progress == total:
                    elapsed_time_seconds = timedelta(seconds=int(elapsed_time))
                    elapsed_time_str = str(elapsed_time_seconds)
                    hour, minute, second = elapsed_time_str.split(":")
                    time_str = "Time : " + hour.zfill(2) + ":" + minute + ":" + second
                    download_progressbar['-INFO-'].update(info)
                    download_progressbar['-PERCENTAGE-'].update(percentage)
                    download_progressbar['-PROGRESS-'].update(progress)
                    download_progressbar['-ETA-'].update(time_str)
                    download_progressbar.refresh()
                    time.sleep(1)
                    download_progressbar.close()

            try:
                urlretrieve(url, save_path, progress_hook)
            except Exception as e:
                sg.Popup(e, title="Error", line_width=50)

            with ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(vosk_cache_dir)
            os.remove(save_path)

            #url = MODEL_PRE_URL + vosk_language.model_of_code[src] + ".zip"
            #download_vosk_model(src, url, vosk_cache_dir)


        if event == 'right_click':

            x, y = main_window.TKroot.winfo_pointerxy()
            widget = main_window.TKroot.winfo_containing(x, y)
            widget.focus_set()


        if event == 'Copy':

            key = main_window.find_element_with_focus().Key
            widget = main_window[key].Widget
            selected_text = None
            try:
                selected_text = widget.selection_get()
            except:
                pass
            if sys.platform == "win32":
                if selected_text:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(selected_text)
                    win32clipboard.CloseClipboard()
            else:
                if selected_text:
                    set_clipboard_text(selected_text)


        elif event == 'Paste':

            key = main_window.find_element_with_focus().Key
            current_value = values[key]
            text_elem = main_window[key]
            element_type = type(text_elem)
            strings = str(text_elem.Widget.index('insert'))
            cursor_position_strings = ""
            if "Input" in str(element_type):
                cursor_position_strings = strings
            elif "Multiline" in str(element_type):
                cursor_position_strings = strings[2:]
            cursor_position = int(cursor_position_strings)
            clipboard_data = None
            if sys.platform == "win32":
                try:
                    win32clipboard.OpenClipboard()
                    clipboard_data = win32clipboard.GetClipboardData()
                except Exception as e:
                    #show_error_messages(e)
                    pass
            else:
                try:
                    clipboard_data = get_clipboard_text()
                except:
                    pass
                    
            if clipboard_data:
                new_value = current_value[:cursor_position] + clipboard_data + current_value[cursor_position:]
                main_window[key].update(new_value)
                cursor_position += len(clipboard_data)
                if "Multiline" in str(element_type):
                    text_elem.Widget.mark_set('insert', f'1.{cursor_position}')
                    text_elem.Widget.see(f'1.{cursor_position}')
                elif "Input" in str(element_type):
                    text_elem.Widget.icursor(cursor_position)
                    text_elem.Widget.xview_moveto(1.0)


        if event == 'Exit' or event == sg.WIN_CLOSED:

            if transcribe_window is not None and window == transcribe_window:

                if not not_transcribing:
                    FONT = ("Helvetica", 10)
                    sg.set_options(font=FONT)
                    answer = popup_yes_no('Are you sure?', title='Confirm')
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

                        else:
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

            else:
                text = get_clipboard_text()
                if text:
                    main_window['-URL-'].update(text.strip())


        elif event == '-RECORD-STREAMING-':

            if values['-RECORD-STREAMING-']:
                is_valid_url_streaming = is_streaming_url(str(values['-URL-']).strip())
                if not is_valid_url_streaming:
                    FONT = ("Helvetica", 10)
                    sg.set_options(font=FONT)
                    sg.Popup('Invalid URL, please enter a valid URL', title="Info", line_width=50)
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

                        else:
                            thread_record_streaming = Thread(target=record_streaming_linux, args=(stream_url.url, tmp_recorded_streaming_filepath))
                            thread_record_streaming.start()

                    else:
                        FONT = ("Helvetica", 10)
                        sg.set_options(font=FONT)
                        sg.Popup('Invalid URL, please enter a valid URL', title="Info", line_width=50)
                        recognizing = False
                        main_window['-START-BUTTON-'].update(('Stop','Start')[not recognizing], button_color=(('white', ('red', '#283b5b')[not recognizing])))
                        main_window['-RECORD-STREAMING-'].update(False)

                regions = []
                transcriptions = []

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:
                    time.sleep(1)

                tmp_src_txt_transcription_filename = f"record.{src}.txt"
                tmp_src_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_src_txt_transcription_filename)
                tmp_src_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "w")
                tmp_src_txt_transcription_file.write("")
                tmp_src_txt_transcription_file.close()

                tmp_dst_txt_transcription_filename = f"record.{dst}.txt"
                tmp_dst_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), tmp_dst_txt_transcription_filename)
                tmp_dst_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "w")
                tmp_dst_txt_transcription_file.write("")
                tmp_dst_txt_transcription_file.close()

                main_window['-ML-SRC-PARTIAL-RESULTS-'].update("")
                main_window['-ML-SRC-RESULTS-'].update("")
                main_window['-ML-DST-PARTIAL-RESULTS-'].update("")
                main_window['-ML-DST-RESULTS-'].update("")

                thread_recognize = Thread(target=worker_recognize, args=(src, dst, show_error_messages), daemon=True)
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

                # IF RECORD STREAMING
                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:

                    if sys.platform == "win32":
                        #print("thread_record_streaming.is_alive() = {}".format(thread_record_streaming.is_alive()))
                        stop_record_streaming_windows()

                    else:
                        #print("thread_record_streaming.is_alive() = {}".format(thread_record_streaming.is_alive()))
                        stop_record_streaming_linux()

                    tmp_src_subtitle_filename = f"record.{src}.{subtitle_format}"
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

                    else:
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

            sentence_translator = SentenceTranslator(src=src, dst=dst, error_messages_callback=show_error_messages)
            line = str(values[event]).strip().lower()
            main_window['-ML-SRC-RESULTS-'].update(line + "\n", background_color_for_value='yellow1', append=True, autoscroll=True)
            translated_line = sentence_translator(line)
            main_window['-ML-DST-RESULTS-'].update(translated_line + "\n", background_color_for_value='yellow1', append=True, autoscroll=True)

            if overlay_translation_window:
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
                if overlay_translation_window:
                    overlay_translation_window.UnHide()
                    overlay_translation_window['-ML-OVERLAY-DST-PARTIAL-RESULTS-'].update(translated_text,background_color_for_value='black', visible=True, autoscroll=True)
                else:
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


        elif event == '-EVENT-UPDATE-GENERIC-PROGRESS-BAR-':

            pb = values[event]
            info = pb[0]
            total = pb[1]
            percentage = pb[2]
            progress = pb[3]
            time_str = pb[4]

            FONT = ("Helvetica", 10)
            sg.set_options(font=FONT)

            if progressbar:
                progressbar.UnHide()
            else:
                progressbar = make_progress_bar_window("", 100)

            progressbar['-INFO-'].update(info)
            progressbar['-PERCENTAGE-'].update(percentage)
            progressbar['-PROGRESS-'].update(progress)
            progressbar['-ETA-'].update(time_str)

            if progress == total:
                progressbar['-ETA-'].update(time_str)
                progressbar.refresh()
                time.sleep(1)
                progressbar.Hide()


        elif event == '-EVENT-TRANSCRIBE-PREPARE-WINDOW-':

            prepare = values[event]

            if prepare:

                FONT = ("Helvetica", 10)
                sg.set_options(font=FONT)
                transcribe_window = make_transcribe_window("", 100)

                for element in transcribe_window.element_list():
                    if isinstance(element, sg.Text) or isinstance(element, sg.Multiline):
                        element.update(font=FONT)
                    if isinstance(element, sg.Button):
                        transcribe_window['-CANCEL-'].Widget.config(font=FONT)

                transcribe_window['-OUTPUT-MESSAGES-'].update("")
                transcribe_window['-FILE-DISPLAY-NAME-'].update("File to process")
                transcribe_window['-INFO-'].update("Progress info")
                transcribe_window['-PROGRESS-'].update(0)
                transcribe_window['-PERCENTAGE-'].update("0%")
                transcribe_window['-ETA-'].update('ETA  : --:--:--')


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
                transcribe_window['-FILE-DISPLAY-NAME-'].update(visible=True)
                transcribe_window['-INFO-'].update(visible=True)
                transcribe_window['-PROGRESS-'].update(visible=True)
                transcribe_window['-PERCENTAGE-'].update(visible=True)
                transcribe_window['-ETA-'].update(visible=True)
            elif not show:
                transcribe_window['-FILE-DISPLAY-NAME-'].update(visible=False)
                transcribe_window['-INFO-'].update(visible=False)
                transcribe_window['-PROGRESS-'].update(visible=False)
                transcribe_window['-PERCENTAGE-'].update(visible=False)
                transcribe_window['-ETA-'].update(visible=False)


        elif event == '-EVENT-TRANSCRIBE-HIDE-PROGRESS-BAR-':

            hide = values[event]

            transcribe_window['-FILE-DISPLAY-NAME-'].update(visible=False)
            transcribe_window['-INFO-'].update(visible=False)
            transcribe_window['-PROGRESS-'].update(visible=False)
            transcribe_window['-PERCENTAGE-'].update(visible=False)
            transcribe_window['-ETA-'].update(visible=False)


        elif event == '-EVENT-TRANSCRIBE-UPDATE-PROGRESS-BAR-':

            pb = values[event]
            media_file_display_name = pb[0]
            info = pb[1]
            total = pb[2]
            percentage = pb[3]
            progress = pb[4]
            time_str = pb[5]

            processing_string = ""
            if media_file_display_name == "...":
                processing_string = "File to process"
            else:
                processing_string = f"Processing '{media_file_display_name}'"

            transcribe_window['-FILE-DISPLAY-NAME-'].update(processing_string)
            transcribe_window['-INFO-'].update(info)
            transcribe_window['-PERCENTAGE-'].update(percentage)
            transcribe_window['-PROGRESS-'].update(progress)
            transcribe_window['-ETA-'].update(time_str)

            if progress == total:
                transcribe_window['-ETA-'].update(time_str)
                transcribe_window.refresh()
                time.sleep(1)


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

            if not recognizing:

                if not args.video_filename:
                    tmp_recorded_streaming_filename = "record.mp4"
                else:
                    tmp_recorded_streaming_filename = args.video_filename

                tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)

                if os.path.isfile(tmp_recorded_streaming_filepath):

                    saved_recorded_streaming_filename = sg.popup_get_file('', no_window=True, save_as=True, font=FONT, default_path=saved_recorded_streaming_filename, file_types=video_file_types)

                    if saved_recorded_streaming_filename:
                        shutil.copy(tmp_recorded_streaming_filepath, saved_recorded_streaming_filename)

                else:
                    FONT = ("Helvetica", 10)
                    sg.set_options(font=FONT)
                    sg.Popup("No streaming was recorded", title="Info", line_width=50)

            else:
                pass


        elif event == '-EVENT-SAVE-SRC-TRANSCRIPTION-':

            if not recognizing:

                if os.path.isfile(tmp_src_txt_transcription_filepath):

                    saved_src_subtitle_filename = sg.popup_get_file('', no_window=True, save_as=True, font=FONT, file_types=subtitle_file_types)

                    if saved_src_subtitle_filename:

                        selected_subtitle_format = saved_src_subtitle_filename.split('.')[-1]

                        if selected_subtitle_format in SubtitleFormatter.supported_formats:

                            writer = SubtitleWriter(regions, transcriptions, subtitle_format, error_messages_callback=show_error_messages)
                            writer.write(saved_src_subtitle_filename)

                        else:

                            saved_src_subtitle_file = open(saved_src_subtitle_filename, "w")
                            tmp_src_txt_transcription_file = open(tmp_src_txt_transcription_filepath, "r")
                            for line in tmp_src_txt_transcription_file:
                                if line:
                                    saved_src_subtitle_file.write(line)
                            saved_src_subtitle_file.close()
                            tmp_src_txt_transcription_file.close()

                else:
                    FONT = ("Helvetica", 10)
                    sg.set_options(font=FONT)
                    sg.Popup("No transcriptions was recorded", title="Info", line_width=50)

            else:
                pass


        elif event == '-EVENT-SAVE-DST-TRANSCRIPTION-':

            if not recognizing:

                if os.path.isfile(tmp_dst_txt_transcription_filepath):

                    saved_dst_subtitle_filename = sg.popup_get_file('', no_window=True, save_as=True, font=FONT, file_types=subtitle_file_types)

                    if saved_dst_subtitle_filename:

                        selected_subtitle_format = saved_dst_subtitle_filename.split('.')[-1]

                        if selected_subtitle_format in SubtitleFormatter.supported_formats:
                            translation_writer = SubtitleWriter(created_regions, translated_transcriptions, subtitle_format, error_messages_callback=show_error_messages)
                            translation_writer.write(saved_dst_subtitle_filename)

                        else:
                            saved_dst_subtitle_file = open(saved_dst_subtitle_filename, "w")
                            tmp_dst_txt_transcription_file = open(tmp_dst_txt_transcription_filepath, "r")
                            for line in tmp_dst_txt_transcription_file:
                                if line:
                                    saved_dst_subtitle_file.write(line)
                            saved_dst_subtitle_file.close()
                            tmp_dst_txt_transcription_file.close()
                else:
                    FONT = ("Helvetica", 10)
                    sg.set_options(font=FONT)
                    sg.Popup("No translated transcriptions was recorded", title="Info", line_width=50)

            else:
                pass


        elif event == '-EXCEPTION-':
            e = str(values[event]).strip()
            sg.Popup(e, title="Error", line_width=50)
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
    if os.path.isfile(translated_transcriptions_filepath): os.remove(translated_transcriptions_filepath)
    if os.path.isfile(transcribe_is_done_filepath): os.remove(transcribe_is_done_filepath)

    remove_temp_files("wav")
    remove_temp_files("mp4")

    main_window.close()
    sys.exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
