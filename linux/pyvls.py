import sys
import os
import time
import threading
from threading import Timer
import argparse
import sounddevice as sd
import json
import httpx
import PySimpleGUI as sg
try:
    import queue  # Python 3 import
except ImportError:
    import Queue  # Python 2 import

import tempfile
from datetime import datetime, timedelta
import subprocess
import ctypes
import platform
from streamlink import Streamlink
from streamlink.exceptions import NoPluginError
import six
import pysrt
import shutil
import re
import select


#---------------------------------------------------------VOSK PART---------------------------------------------------------------------#

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

last_selected_src = None
src_file = None
last_selected_dst = None
dst_file = None

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

#-------------------------------------------------------------------------------------------------------------------------------------#

q = queue.Queue()

SampleRate = None
Device = None
wav_file_name = None
recognizing = False

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

thread_record_streaming = None
thread_recognize = None
thread_timed_translate = None
text=''


#-------------------------------------------------------------MISC FUNCTIONS-------------------------------------------------------------#


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


def check_thread_status():
    global thread_recognize
    if thread_recognize != None:
        if thread_recognize.is_alive():
            main_window.TKroot.after(100, check_thread_status) # check again in 100 milliseconds
    else:
        pass


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


def worker_recognize(src):
    global main_window, recognizing, text, Device, SampleRate, wav_file_name, regions, transcripts, start_button_click_time, ffmpeg_start_run_time, get_tmp_recorded_streaming_filepath_time, loading_time, first_streaming_duration_recorded

    p = 0
    f = 0
    l = 0

    start_time = None
    end_time = None
    first_streaming_duration_recorded = None
    absolute_start_time = None

    time_value_filename = "time_value"
    time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
    if os.path.isfile(time_value_filepath):
        time_value_file = open(time_value_filepath, "r")
        time_value_string = time_value_file.read()
        first_streaming_duration_recorded = datetime.strptime(time_value_string, "%H:%M:%S.%f") - datetime(1900, 1, 1)
        time_value_file.close()

    try:
        if SampleRate is None:
            #soundfile expects an int, sounddevice provides a float:
            device_info = sd.query_devices(Device, "input")
            SampleRate = int(device_info["default_samplerate"])

        if wav_file_name:
            dump_fn = open(wav_file_name, "wb")
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
                    print("BREAK")
                    break

                data = q.get()
                results_text = ''
                partial_results_text = ''

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result["text"] and absolute_start_time:
                        text = result["text"]
                        if len(text) > 0:
                            main_window.write_event_value('-EVENT-VOICE-RECOGNIZED-', text)
                            transcripts.append(text)

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

                            src_tmp_txt_transcription_filename = "record.txt"
                            src_tmp_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), src_tmp_txt_transcription_filename)
                            src_tmp_txt_transcription_file = open(src_tmp_txt_transcription_filepath, "a")
                            src_tmp_txt_transcription_file.write(time_stamp_string + " " + text + "\n")
                            src_tmp_txt_transcription_file.close()

                else:
                    result = json.loads(rec.PartialResult())
                    if result["partial"]:
                        text = result["partial"]
                        if len(text)>0:
                            main_window.write_event_value('-EVENT-VOICE-RECOGNIZED-', text)

                            if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:
                                if ffmpeg_start_run_time and l == 0:

                                    loading_time = datetime.now() - ffmpeg_start_run_time - (ffmpeg_start_run_time - get_tmp_recorded_streaming_filepath_time)
                                    if first_streaming_duration_recorded:

                                        # A ROUGH GUESS OF THE FIRST PARTIAL RESULT START TIME :)
                                        start_time = first_streaming_duration_recorded + loading_time
                                        #start_time = first_streaming_duration_recorded + timedelta(seconds=1)

                                        # MAKE SURE THAT IT'S EXECUTED ONLY ONCE
                                        l += 1

                                else:
                                    if p == 0:
                                        absolute_start_time = datetime.now()

                                        if end_time:
                                            start_time = end_time
                                        p += 1

                            else:
                                if l == 0:
                                    now_datetime = datetime.now()
                                    now_time = now_datetime.time()
                                    microseconds = (now_time.hour * 3600 + now_time.minute * 60 + now_time.second) * 1000000 + now_time.microsecond
                                    start_time = timedelta(microseconds=microseconds)
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


class TranscriptionTranslator(object):
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


def GoogleTranslate(text, src, dst):
    url = 'https://translate.googleapis.com/translate_a/'
    params = 'single?client=gtx&sl='+src+'&tl='+dst+'&dt=t&q='+text;
    with httpx.Client(http2=True) as client:
        client.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://translate.google.com',})
        response = client.get(url+params)
        #print('response.status_code = {}'.format(response.status_code))
        if response.status_code == 200:
            response_json = response.json()[0]
            #print('response_json = {}'.format(response_json))
            if response_json!= None:
                length = len(response_json)
                #print('length = {}'.format(length))
                translation = ""
                for i in range(length):
                    #print("{} {}".format(i, response_json[i][0]))
                    translation = translation + response_json[i][0]
                return translation
        return


def worker_timed_translate(src, dst):
    global main_window, recognizing

    while (recognizing==True):
        phrase = str(main_window['-ML1-'].get())
        if len(phrase)>0:
            translated_text = GoogleTranslate(phrase, src, dst)
            main_window.write_event_value('-EVENT-VOICE-TRANSLATED-', translated_text)

        if not recognizing:
            break


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
    global ffmpeg_start_run_time, first_streaming_duration_recorded, recognizing

    #ffmpeg_cmd = ['ffmpeg', '-y', '-i', hls_url,  '-movflags', '+frag_keyframe+separate_moof+omit_tfhd_offset+empty_moov', '-fflags', 'nobuffer', '-loglevel', 'error',  f'{filename}']
    ffmpeg_cmd = ['ffmpeg', '-y', '-i', f'{hls_url}',  '-movflags', '+frag_keyframe+separate_moof+omit_tfhd_offset+empty_moov', '-fflags', 'nobuffer', f'{filename}']

    ffmpeg_start_run_time = datetime.now()
    #print("ffmpeg_start_run_time = {}".format(ffmpeg_start_run_time))

    first_streaming_duration_recorded = None

    i = 0

    process = subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)

    msg = "RUNNING"
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
    global ffmpeg_start_run_time, first_streaming_duration_recorded

    #ffmpeg_cmd = ['ffmpeg', '-y', '-i', url, '-c', 'copy', '-bsf:a', 'aac_adtstoasc', '-f', 'mp4', output_file]
    ffmpeg_cmd = ['ffmpeg', '-y', '-i', f'{url}',  '-movflags', '+frag_keyframe+separate_moof+omit_tfhd_offset+empty_moov', '-fflags', 'nobuffer', f'{output_file}']

    ffmpeg_start_run_time = datetime.now()
    #first_streaming_duration_recorded = None

    # Define a function to run the ffmpeg process in a separate thread
    def run_ffmpeg():
        i = 0
        line = None

        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        msg = "RUNNING"
        main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

        # Set up a timer to periodically check for new output
        timeout = 1.0
        timer = threading.Timer(timeout, lambda: None)
        timer.start()

        # Read output from ffmpeg process
        while True:
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

                    time_value_filename = "time_value"
                    time_value_filepath = os.path.join(tempfile.gettempdir(), time_value_filename)
                    time_value_file = open(time_value_filepath, "w")
                    time_value_file.write(str(time_value))
                    time_value_file.close()

                streaming_duration_recorded = datetime.strptime(str(time_value), "%H:%M:%S.%f") - datetime(1900, 1, 1)
                #print("streaming_duration_recorded = {}".format(streaming_duration_recorded))
                main_window.write_event_value('-EVENT-STREAMING-DURATION-RECORDED-', streaming_duration_recorded)

            # Restart the timer to check for new output
            timer.cancel()
            timer = threading.Timer(timeout, lambda: None)
            timer.start()

        # Close the ffmpeg process and print the return code
        process.stdout.close()
        process.stderr.close()
        #print(f"FFmpeg exited with return code {process.returncode}")

    # Start the thread to run ffmpeg
    thread = threading.Thread(target=run_ffmpeg)
    thread.start()

    # Return the thread object so that the caller can join it if needed
    return thread


def stop_thread(thread):
    if thread and thread.is_alive():
        if platform.system() == 'Windows':
            # Use ctypes to call the TerminateThread() function from the Windows API
            # This forcibly terminates the thread, which can be unsafe in some cases
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.OpenThread(1, False, thread.ident)
            if thread_handle:
                ret = kernel32.TerminateThread(thread_handle, 0)
                if ret == 0:
                    msg = thread.name + "TERMINATION ERROR!"
                else:
                    msg = thread.name + "TERMINATED"
                    thread = None
                #print(msg)
        elif platform.system() == 'Linux':
            libpthread = ctypes.CDLL('libpthread.so.0')
            thread_id = thread.ident
            if thread_id:
                print('thread_id = {}'.format(thread_id))
                ret = libpthread.pthread_cancel(thread_id)
                if ret == 0:
                    msg = thread.name + "TERMINATION ERROR!"
                else:
                    msg = thread.name + "TERMINATED"
                    thread = None
                #print(msg)


def stop_record_streaming_linux():
    global main_window

    msg = "TERMINATED"
    main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

    ffmpeg_pid = subprocess.check_output(['pgrep', '-f', 'ffmpeg']).strip()
    if ffmpeg_pid:
        subprocess.Popen(['kill', ffmpeg_pid])
    else:
        msg = 'FFMPEG HAS TERMINATED'
        main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)


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
        main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

        tasklist_output = subprocess.check_output(['tasklist'], creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8')

        ffmpeg_pid = None
        for line in tasklist_output.split('\n'):
            if 'ffmpeg' in line:
                ffmpeg_pid = line.split()[1]
                break
        if ffmpeg_pid:
            devnull = open(os.devnull, 'w')
            subprocess.Popen(['taskkill', '/F', '/T', '/PID', ffmpeg_pid], stdout=devnull, stderr=devnull, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            msg = 'FFMPEG HAS TERMINATED'
            main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)

    else:
        msg = "NOT RUNNING"
        main_window.write_event_value('-EVENT-THREAD-RECORD-STREAMING-STATUS-', msg)


#-------------------------------------------------------------GUI FUNCTIONS-------------------------------------------------------------#


def handle_focus(event):
    if event.widget == window:
        window.focus_set()


def steal_focus():
    global main_window, overlay_translation_window

    #print('debug')
    if overlay_translation_window:
        overlay_translation_window.close()
        overlay_translation_window=None
    if(platform.system() == "Windows"):
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


def make_progress_bar_window(total, info):
    FONT_TYPE = "Helvetica"
    FONT_SIZE = 9
    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    layout = [[sg.Text(info, key='-INFO-')],
              [sg.ProgressBar(total, orientation='h', size=(30, 10), key='-PROGRESS-'), sg.Text(size=(5,1), key='-PERCENTAGE-')]]

    window = sg.Window("Progress", layout, no_titlebar=False, finalize=True)

    return window


def make_overlay_translation_window(translated_text):
    xmax, ymax = sg.Window.get_screen_size()
    max_columns = None
    max_lines = None

    if platform.system() == 'Windows':
        SM_CXSCREEN = 0
        SM_CYSCREEN = 1
        width = ctypes.windll.user32.GetSystemMetrics(SM_CXSCREEN)
        height = ctypes.windll.user32.GetSystemMetrics(SM_CYSCREEN)

        max_columns = int(width/8.5)
        max_lines = int(height/18)

    elif platform.system() == 'Linux':
        output = subprocess.check_output(['xrandr'])
        primary_line = [line for line in output.decode().splitlines() if ' connected primary' in line][0]
        primary_size = primary_line.split()[3].replace("+"," ").split()[0]

        max_columns, max_lines = int(primary_size.split('x')[0]) // 8, int(primary_size.split('x')[1]) // 18

    columns = max_columns
    rows = max_lines

    FONT_TYPE = 'Arial'
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

    if xmax > 1280: FONT_SIZE = 16
    if xmax <= 1280: FONT_SIZE = 16

    sg.set_options(font=("Arial", FONT_SIZE))

    multiline_width = len(translated_text)
    multiline_height = nl

    TRANSPARENT_BLACK='#add123'

    if not (platform.system() == "Windows"):
        layout = [[sg.Multiline(default_text=translated_text, size=(multiline_width, multiline_height), text_color='yellow1', border_width=0,
            background_color='black', no_scrollbar=True,
            justification='l', expand_x=True, expand_y=True,  key='-ML-LS2-')]]

        overlay_translation_window = sg.Window('text', layout, no_titlebar=True, keep_on_top=True, size=(window_width,window_height), location=(window_x,window_y), 
            margins=(0, 0), background_color='black', alpha_channel=0.7, return_keyboard_events=True, finalize=True)

        overlay_translation_window.TKroot.attributes('-type', 'splash')
        overlay_translation_window.TKroot.attributes('-topmost', 1)
        #overlay_translation_window.TKroot.attributes('-topmost', 0)
    else:
        layout = [[sg.Multiline(default_text=translated_text, size=(multiline_width, multiline_height), text_color='yellow1', border_width=0,
            background_color='white', no_scrollbar=True,
            justification='l', expand_x=True, expand_y=True,  key='-ML-LS2-')]]

        overlay_translation_window = sg.Window('Translation', layout, no_titlebar=True, keep_on_top=True, size=(window_width,window_height), location=(window_x,window_y), 
            margins=(0, 0), background_color='white', transparent_color='white', grab_anywhere=True, return_keyboard_events=True, 
            finalize=True)

    #event, values = overlay_translation_window.read(timeout=TIMEOUT, close=True)
    event, values = overlay_translation_window.read(500)

    return overlay_translation_window


def move_center(window):
    screen_width, screen_height = window.get_screen_dimensions()
    win_width, win_height = window.size
    x, y = (screen_width - win_width)//2, (screen_height - win_height)//2-30
    window.move(x, y)


#=============================================================MAIN PROGRAM=============================================================#

def main():
    global main_window, thread_recognize, thread_timed_translate, thread_record_streaming, text, recognizing, \
        Device, SampleRate, wav_file_name, tmp_recorded_streaming_filepath, regions, transcripts, \
        start_button_click_time, ffmpeg_start_run_time, get_tmp_recorded_streaming_filepath_time, \
        first_streaming_duration_recorded, is_valid_url_streaming, stop_event

#--------------------------------------------------------------GUI PARTS--------------------------------------------------------------#

    xmax, ymax = sg.Window.get_screen_size()
    main_window_width = int(0.5*xmax)
    main_window_height = int(0.5*ymax)
    multiline_width = int(0.15*main_window_width)
    multiline_height = int(0.0125*main_window_height)
    FONT_TYPE = "Helvetica"
    FONT_SIZE = 9
    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    layout = [[sg.Frame('Hints',[
                                [sg.Text('Click \"Start\" button to start listening and printing subtitles on screen.', expand_x=True, expand_y=True)],
                                [sg.Text('Paste the streaming URL into URL inputbox then check the \"Record Streaming\" checkbox to record the streaming.', expand_x=True, expand_y=True)],
                                [sg.Text('If you record the streaming, when you save transcriptions, all timestamps will be relative to the \"Start\" button clicked time.', expand_x=True, expand_y=True)],
                                [sg.Text('If you don\'t record the streaming, all timestamps will be based on your system clock.', expand_x=True, expand_y=True)],
                                [sg.Text('If you need accurate subtitles sync for the video, please use PyAutoSRT (https://github.com/botbahlul/PyAutoSRT) ', expand_x=True, expand_y=True, enable_events=True)],
                                ], border_width=2, expand_x=True, expand_y=True)],
              [sg.Text('URL'), sg.Input(size=(12, 1), expand_x=True, expand_y=True, key='-URL-'),
               sg.Checkbox("Record Streaming", key='-RECORD-STREAMING-', enable_events=True)],
              [sg.Text('', size=(3, 1)),
               sg.Text('Thread status'),
               sg.Text('NOT RUNNING', size=(24, 1), background_color='green1', text_color='black', expand_x=True, expand_y=True, key='-RECORD-STREAMING-STATUS-'),
               sg.Text('Duration recorded', size=(16, 1)),
               sg.Text('0:00:00.000000', size=(9, 1), background_color='green1', text_color='black', expand_x=True, expand_y=True, key='-STREAMING-DURATION-RECORDED-'),
               sg.Text('', size=(14, 1), expand_x=True, expand_y=True)],
              [sg.Text('', expand_x=True, expand_y=True),
               sg.Button('SAVE RECORDED STREAMING', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-RECORDED-STREAMING-'),
               sg.Text('', expand_x=True, expand_y=True)],
              [sg.Text('Audio language', size=(12, 1), expand_x=True, expand_y=True),
               sg.Combo(list(map_src_of_language), size=(int(0.2*multiline_width), 1), default_value="English", expand_x=True, expand_y=True, key='-SRC-')],
              [sg.Multiline(size=(multiline_width, multiline_height), expand_x=True, expand_y=True, key='-ML1-')],
              [sg.Text('', expand_x=True, expand_y=True),
               sg.Button('SAVE TRANSCRIPTION', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-SRC-TRANSCRIPTION-'),
               sg.Text('', expand_x=True, expand_y=True)],
              [sg.Text('Translation language', size=(12, 1),expand_x=True, expand_y=True),
               sg.Combo(list(map_dst_of_language), size=(int(0.2*multiline_width), 1), default_value="Indonesian", expand_x=True, expand_y=True, key='-DST-')],
              [sg.Multiline(size=(multiline_width, multiline_height), expand_x=True, expand_y=True, key='-ML2-')],
              [sg.Text('', expand_x=True, expand_y=True, key='-SPACES3-'),
               sg.Button('SAVE TRANSLATED TRANSCRIPTION', size=(31,1), expand_x=True, expand_y=True, key='-EVENT-SAVE-DST-TRANSCRIPTION-'),
               sg.Text('', expand_x=True, expand_y=True, key='-SPACES4-')],
              [sg.Button('Start', expand_x=True, expand_y=True, button_color=('white', '#283b5b'), key='-START-BUTTON-'),
               sg.Button('Exit', expand_x=True, expand_y=True)]]


#----------------------------------------------------------NON GUI PARTS--------------------------------------------------------------#

    # VOSK LogLevel
    SetLogLevel(-1)

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

    tmp_recorded_streaming_filename = "record.mp4"
    tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)
    if os.path.isfile(tmp_recorded_streaming_filepath):
        os.remove(tmp_recorded_streaming_filepath)

    saved_recorded_streaming_filename = None

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

    src_tmp_txt_transcription_filename = "record.txt"
    src_tmp_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), src_tmp_txt_transcription_filename)
    if os.path.isfile(src_tmp_txt_transcription_filepath):
        os.remove(src_tmp_txt_transcription_filepath)

    dst_tmp_txt_transcription_filename = "record.translated.txt"
    dst_tmp_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), dst_tmp_txt_transcription_filename)
    if os.path.isfile(dst_tmp_txt_transcription_filepath):
        os.remove(dst_tmp_txt_transcription_filepath)

    tmp_src_subtitle_filepath = None
    tmp_dst_subtitle_filepath = None

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

    parser.add_argument("-af", "--audio-filename", type=str, metavar="AUDIO_FILENAME", help="audio file to store recording to")
    parser.add_argument("-d", "--device", type=int_or_str, help="input device (numeric ID or substring)")
    parser.add_argument("-r", "--samplerate", type=int, help="sampling rate in Hertz for example 8000, 16000, 44100, or 48000")
    parser.add_argument('-v', '--version', action='version', version='0.1.2')

    parser.add_argument("-u", "--url", type=str, metavar="URL", help="URL of live streaming if you want to record the streaming")
    parser.add_argument("-vf", "--video-filename", type=str, metavar="VIDEO_FILENAME", help="video file to store recording to", default=None)

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
        wav_file_name = args.audio_filename
        dump_fn = open(args.audio_filename, "wb")
    else:
        dump_fn = None

    if args.subtitle_format not in FORMATTERS.keys():
        FONT_TYPE = "Helvetica"
        FONT_SIZE = 9
        sg.set_options(font=(FONT_TYPE, FONT_SIZE))
        sg.Popup("Subtitle format is not supported, you can choose it later when you save transcriptions.", title="Info")

    if args.video_filename:
        saved_recorded_streaming_basename, saved_recorded_streaming_extension = os.path.splitext(os.path.basename(args.video_filename))
        saved_recorded_streaming_extension = saved_recorded_streaming_extension[1:]
        if saved_recorded_streaming_extension != "mp4":
            saved_recorded_streaming_filename = "{base}.{format}".format(base=saved_recorded_streaming_basename, format="mp4")
        else:
            saved_recorded_streaming_filename = args.video_filename


#-------------------------------------REMAINING GUI PARTS NEEDED TO LOAD AT LAST BEFORE MAIN LOOP-------------------------------------#


    main_window = sg.Window('VOSK Live Subtitles', layout, resizable=True, keep_on_top=True, finalize=True)
    main_window['-SRC-'].block_focus()

    if (platform.system() == "Windows"):
        if main_window.TKroot is not None:
            main_window.TKroot.attributes('-topmost', True)
            main_window.TKroot.attributes('-topmost', False)

    if not (platform.system() == "Windows"):
        if main_window.TKroot is not None:
            main_window.TKroot.attributes('-topmost', 1)
            main_window.TKroot.attributes('-topmost', 0)

    main_window.finalize()
    overlay_translation_window = None
    move_center(main_window)
    
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


#---------------------------------------------------------------MAIN LOOP--------------------------------------------------------------#


    while True:
        window, event, values = sg.read_all_windows(500)

        #print('event =', event)
        #print('recognizing =', recognizing)

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
            break


        elif event == '-RECORD-STREAMING-':
            if values['-RECORD-STREAMING-']:
                is_valid_url_streaming = is_streaming_url(str(values['-URL-']).strip())
                if not is_valid_url_streaming:
                    FONT_TYPE = "Helvetica"
                    FONT_SIZE = 9
                    sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                    sg.Popup('Invalid URL, please enter a valid URL.                   ', title="Info")
                    main_window['-RECORD-STREAMING-'].update(False)


        elif event == '-START-BUTTON-':
            recognizing = not recognizing
            #print('Start button clicked, changing recognizing status')
            #print('recognizing =', recognizing)

            main_window['-START-BUTTON-'].update(('Stop','Start')[not recognizing], button_color=(('white', ('red', '#283b5b')[not recognizing])))

            if recognizing:
                #print('VOSK Live Subtitle is START LISTENING now')
                #print('recognizing =', recognizing)

                start_button_click_time = datetime.now()
                #print("start_button_click_time = {}".format(start_button_click_time))

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:

                    if is_valid_url_streaming:

                        url = values['-URL-']

                        tmp_recorded_streaming_filename = "record.mp4"
                        tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)

                        get_tmp_recorded_streaming_filepath_time = datetime.now()
                        #print("get_tmp_recorded_streaming_filepath_time = {}".format(get_tmp_recorded_streaming_filepath_time))

                        #NEEDED FOR streamlink MODULE TO RUN AS PYINSTALLER COMPILED BINARY
                        os.environ['STREAMLINK_DIR'] = './streamlink/'
                        os.environ['STREAMLINK_PLUGINS'] = './streamlink/plugins/'
                        os.environ['STREAMLINK_PLUGIN_DIR'] = './streamlink/plugins/'

                        streamlink = Streamlink()
                        streams = streamlink.streams(url)
                        stream_url = streams['360p']

                        # WINDOWS AND LINUX HAS DIFFERENT BEHAVIOR WHEN RUNNING FFMPEG AS THREAD
                        if platform.system() == "Windows":
                            thread_record_streaming = threading.Thread(target=record_streaming_windows, args=(stream_url.url, tmp_recorded_streaming_filepath), daemon=True)
                            thread_record_streaming.start()

                        elif platform.system() == "Linux":
                            thread_record_streaming = threading.Thread(target=record_streaming_linux, args=(stream_url.url, tmp_recorded_streaming_filepath))
                            thread_record_streaming.start()

                    else:
                        FONT_TYPE = "Helvetica"
                        FONT_SIZE = 9
                        sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                        sg.Popup('Invalid URL, please enter a valid URL.                   ', title="Info")
                        recognizing = False
                        main_window['-START-BUTTON-'].update(('Stop','Start')[not recognizing], button_color=(('white', ('red', '#283b5b')[not recognizing])))
                        main_window['-RECORD-STREAMING-'].update(False)

                # SHOWING OVERLAY TRANSLATION WINDOW
                if not overlay_translation_window:
                    overlay_translation_window = make_overlay_translation_window(100*' ')
                    overlay_translation_window.Hide()

                regions = []
                transcripts = []

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:
                    time.sleep(1)

                src_tmp_txt_transcription_filename = "record.txt"
                src_tmp_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), src_tmp_txt_transcription_filename)
                src_tmp_txt_transcription_file = open(src_tmp_txt_transcription_filepath, "w")
                src_tmp_txt_transcription_file.write("")
                src_tmp_txt_transcription_file.close()

                src_tmp_txt_transcription_filename = "record.translated.txt"
                src_tmp_txt_transcription_filepath = os.path.join(tempfile.gettempdir(), src_tmp_txt_transcription_filename)
                src_tmp_txt_transcription_file = open(src_tmp_txt_transcription_filepath, "w")
                src_tmp_txt_transcription_file.write("")
                src_tmp_txt_transcription_file.close()

                thread_recognize = threading.Thread(target=worker_recognize, args=(src,), daemon=True)
                thread_recognize.start()

                thread_timed_translate = threading.Timer(0.2, worker_timed_translate, args=(src, dst))
                thread_timed_translate.daemon = True
                thread_timed_translate.start()

                main_window.TKroot.after(100, check_thread_status)

            else:
                #print('VOSK Live Subtitle is STOP LISTENING now')
                #print('recognizing =', recognizing)

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:

                    if platform.system() == "Windows":
                        #print("thread_record_streaming.is_alive() = {}".format(thread_record_streaming.is_alive()))
                        stop_record_streaming_windows()

                    elif platform.system() == "Linux":
                        #print("thread_record_streaming.is_alive() = {}".format(thread_record_streaming.is_alive()))
                        stop_record_streaming_linux()

                text=''
                main_window['-ML1-'].update(text, background_color_for_value='yellow1', autoscroll=True)
                translated_text=''
                main_window['-ML2-'].update(translated_text, background_color_for_value='yellow1', autoscroll=True)

                if overlay_translation_window:
                    overlay_translation_window.close()
                    overlay_translation_window=None

                # SAVING TRANSCRIPTIONS IN A TEMPORARY SUBTITLE FILE
                #print("Saving transcriptions")
                timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
                subtitle_format = "srt"
                formatter = FORMATTERS.get(subtitle_format)
                formatted_subtitles = formatter(timed_subtitles)

                if main_window['-URL-'].get() != (None or "") and main_window['-RECORD-STREAMING-'].get() == True:
                    tmp_src_subtitle_basename, extension = os.path.splitext(os.path.basename(tmp_recorded_streaming_filepath))
                    tmp_src_subtitle_filename = tmp_src_subtitle_basename + "." + subtitle_format
                    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)
                else:
                    tmp_src_subtitle_filename = "record" + "." + subtitle_format
                    tmp_src_subtitle_filepath = os.path.join(tempfile.gettempdir(), tmp_src_subtitle_filename)

                tmp_src_subtitle_file = open(tmp_src_subtitle_filepath, 'wb')
                tmp_src_subtitle_file.write(formatted_subtitles.encode("utf-8"))
                tmp_src_subtitle_file.close()

                # SAVING TRANSCRIPTIONS AS FORMATTED SUBTITLE FILE WITH FILENAME DECLARED IN COMMAND LINE ARGUMENTS
                if args.subtitle_filename:
                    subtitle_file_base, subtitle_file_extension = os.path.splitext(args.subtitle_filename)
                    subtitle_file_extension = subtitle_file_extension[1:]
                    #print("subtitle_file_base = {}".format(subtitle_file_base))
                    #print("subtitle_file_extension = {}".format(subtitle_file_extension))

                    if subtitle_file_extension and subtitle_file_extension in FORMATTERS.keys():
                        #print("subtitle_file_extension in FORMATTERS.keys()")
                        subtitle_filepath = "{base}.{format}".format(base=subtitle_file_base, format=subtitle_file_extension)
                        timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
                        subtitle_format = subtitle_file_extension
                        formatter = FORMATTERS.get(subtitle_format)
                        formatted_subtitles = formatter(timed_subtitles)
                        subtitle_file = open(subtitle_filepath, 'wb')
                        subtitle_file.write(formatted_subtitles.encode("utf-8"))
                        subtitle_file.close()
                    else:
                        #print("subtitle_file_extension not in FORMATTERS.keys()")
                        FONT_TYPE = "Helvetica"
                        FONT_SIZE = 9
                        sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                        sg.Popup("Subtitle format you typed is not supported, subtitle file will be saved in TXT format.", title="Info")
                        subtitle_filepath = "{base}.{format}".format(base=subtitle_file_base, format="txt")
                        shutil.copy(src_tmp_txt_transcription_filepath, subtitle_filepath)

                if thread_record_streaming and thread_record_streaming.is_alive():
                    if platform.system() == "Windows":
                        stop_record_streaming_windows()

                    elif platform.system() == "Linux":
                        stop_record_streaming_linux()

                # SAVING TEMPORARY TRANSLATED TRANSCRIPTIONS AS FORMATTED SUBTITLE FILE
                #print("Saving translated transcriptions")
                created_regions = []
                created_subtitles = []
                for entry in timed_subtitles:
                    created_regions.append(entry[0])
                    created_subtitles.append(entry[1])

                translated_transcriptions = []
                i = 0
                progressbar = make_progress_bar_window(len(created_subtitles), "Saving translated transcriptions...")
                for created_subtitle in created_subtitles:
                    translated_transcription = GoogleTranslate(created_subtitle, src, dst)
                    translated_transcriptions.append(translated_transcription)
                    i += 1
                    progressbar['-INFO-'].update('Saving translated transcriptions...')
                    progressbar['-PERCENTAGE-'].update(f'{int(i*100/len(created_subtitles))}%')
                    progressbar['-PROGRESS-'].update(i)
                progressbar['-INFO-'].update('Saving translated transcriptions...')
                progressbar['-PERCENTAGE-'].update(f'{int(len(created_subtitles)*100/len(created_subtitles))}%')
                progressbar['-PROGRESS-'].update(len(created_subtitles))
                time.sleep(1)
                progressbar.close()

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
                with open(dst_tmp_txt_transcription_filepath, 'w') as f:
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


                # SAVING TRANSLATED SUBTITLE FILE WITH FILENAME BASED ON SUBTITLE FILENAME DECLARED IN COMMAND LINE ARGUMENTS
                if args.subtitle_filename:

                    saved_dst_subtitle_filepath = subtitle_filepath[ :-4] + ".translated." + subtitle_format

                    if subtitle_file_extension and subtitle_file_extension in FORMATTERS.keys():
                        timed_translated_subtitles = [(r, t) for r, t in zip(created_regions, translated_transcriptions) if t]
                        subtitle_format = subtitle_file_extension
                        formatter = FORMATTERS.get(subtitle_format)
                        formatted_translated_subtitles = formatter(timed_translated_subtitles)
                        saved_dst_subtitle_file = open(saved_dst_subtitle_filepath, 'wb')
                        saved_dst_subtitle_file.write(formatted_translated_subtitles.encode("utf-8"))
                        saved_dst_subtitle_file.close()
                    else:
                        shutil.copy(dst_tmp_txt_transcription_filepath, saved_dst_subtitle_filepath)

                if args.video_filename:
                    shutil.copy(tmp_recorded_streaming_filepath, saved_recorded_streaming_filename)

            if not (platform.system() == "Windows"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event == '-EVENT-VOICE-RECOGNIZED-' and recognizing==True:

            text=str(values[event]).strip().lower()
            main_window['-ML1-'].update(text,background_color_for_value='yellow1',autoscroll=True)
            
            if not (platform.system() == "Windows"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event == '-EVENT-VOICE-TRANSLATED-' and recognizing==True:

            overlay_translation_window.UnHide()
            translated_text=str(values[event]).strip().lower()
            main_window['-ML2-'].update(translated_text,background_color_for_value='yellow1',autoscroll=True)

            if (overlay_translation_window):
                overlay_translation_window['-ML-LS2-'].update(translated_text,background_color_for_value='black', autoscroll=True)

            if not (platform.system() == "Windows"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event == '-EVENT-STREAMING-DURATION-RECORDED-':

            record_duration = str(values[event]).strip()
            main_window['-STREAMING-DURATION-RECORDED-'].update(record_duration)


        elif event == '-EVENT-THREAD-RECORD-STREAMING-STATUS-':

            msg = str(values[event]).strip()
            main_window['-RECORD-STREAMING-STATUS-'].update(msg)

            if "RUNNING" in msg:
                main_window['-RECORD-STREAMING-STATUS-'].update(text_color='white', background_color='red')
                main_window['-STREAMING-DURATION-RECORDED-'].update(text_color='white', background_color='red')
            else:
                main_window['-RECORD-STREAMING-STATUS-'].update(text_color='black', background_color='green1')
                main_window['-STREAMING-DURATION-RECORDED-'].update(text_color='black', background_color='green1')


        elif event == '-EVENT-SAVE-RECORDED-STREAMING-':

            tmp_recorded_streaming_filename = "record.mp4"
            tmp_recorded_streaming_filepath = os.path.join(tempfile.gettempdir(), tmp_recorded_streaming_filename)

            if os.path.isfile(tmp_recorded_streaming_filepath):

                saved_recorded_streaming_filename = sg.popup_get_file('', no_window=True, save_as=True, font=(FONT_TYPE, FONT_SIZE), default_path=saved_recorded_streaming_filename, file_types=video_file_types)

                if saved_recorded_streaming_filename:
                    shutil.copy(tmp_recorded_streaming_filepath, saved_recorded_streaming_filename)

            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("No streaming was recorded.                             ", title="Info")


        elif event == '-EVENT-SAVE-SRC-TRANSCRIPTION-':
            if os.path.isfile(src_tmp_txt_transcription_filepath):

                saved_src_subtitle_filename = sg.popup_get_file('', no_window=True, save_as=True, font=(FONT_TYPE, FONT_SIZE), file_types=subtitle_file_types)

                if saved_src_subtitle_filename:

                    selected_subtitle_format = saved_src_subtitle_filename.split('.')[-1]

                    if selected_subtitle_format in FORMATTERS.keys():
                        timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
                        subtitle_format = selected_subtitle_format
                        formatter = FORMATTERS.get(subtitle_format)
                        formatted_subtitles = formatter(timed_subtitles)
                        tmp_src_subtitle_filename = saved_src_subtitle_filename
                        subtitle_file = open(tmp_src_subtitle_filename, 'wb')
                        subtitle_file.write(formatted_subtitles.encode("utf-8"))
                        subtitle_file.close()
                    else:
                        saved_src_subtitle_file = open(saved_src_subtitle_filename, "w")
                        src_tmp_txt_transcription_file = open(src_tmp_txt_transcription_filepath, "r")
                        for line in src_tmp_txt_transcription_file:
                            if line:
                                saved_src_subtitle_file.write(line)
                        saved_src_subtitle_file.close()
                        src_tmp_txt_transcription_file.close()

            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("No transcriptions was recorded.                             ", title="Info")


        elif event == '-EVENT-SAVE-DST-TRANSCRIPTION-':

            if os.path.isfile(dst_tmp_txt_transcription_filepath):

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
                        dst_tmp_txt_transcription_file = open(dst_tmp_txt_transcription_filepath, "r")
                        for line in dst_tmp_txt_transcription_file:
                            if line:
                                saved_dst_subtitle_file.write(line)
                        saved_dst_subtitle_file.close()
                        dst_tmp_txt_transcription_file.close()
            else:
                FONT_TYPE = "Helvetica"
                FONT_SIZE = 9
                sg.set_options(font=(FONT_TYPE, FONT_SIZE))
                sg.Popup("No translated transcriptions was recorded.                 ", title="Info")


    if thread_recognize and thread_recognize.is_alive():
        stop_thread(thread_recognize)

    if thread_timed_translate and thread_timed_translate.is_alive():
        stop_thread(thread_timed_translate)

    main_window.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
