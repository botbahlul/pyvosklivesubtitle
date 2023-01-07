import sys
import os
import threading
from threading import Timer
import argparse
import sounddevice as sd
import json
from pygoogletranslation import Translator
import PySimpleGUI as sg
try:
    import queue  # Python 3 import
except ImportError:
    import Queue  # Python 2 import

#---------------------------------------------------------VOSK PART---------------------------------------------------------------------#

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

def open_dll():
    dlldir = os.path.abspath(os.path.dirname("."))
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
Filename = None
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
arraylist_dst.append("af");
arraylist_dst.append("sq");
arraylist_dst.append("am");
arraylist_dst.append("ar");
arraylist_dst.append("hy");
arraylist_dst.append("as");
arraylist_dst.append("ay");
arraylist_dst.append("az");
arraylist_dst.append("bm");
arraylist_dst.append("eu");
arraylist_dst.append("be");
arraylist_dst.append("bn");
arraylist_dst.append("bho");
arraylist_dst.append("bs");
arraylist_dst.append("bg");
arraylist_dst.append("ca");
arraylist_dst.append("ceb");
arraylist_dst.append("ny");
arraylist_dst.append("zh-CN");
arraylist_dst.append("zh-TW");
arraylist_dst.append("co");
arraylist_dst.append("cr");
arraylist_dst.append("cs");
arraylist_dst.append("da");
arraylist_dst.append("dv");
arraylist_dst.append("nl");
arraylist_dst.append("doi");
arraylist_dst.append("en");
arraylist_dst.append("eo");
arraylist_dst.append("et");
arraylist_dst.append("ee");
arraylist_dst.append("fil");
arraylist_dst.append("fi");
arraylist_dst.append("fr");
arraylist_dst.append("fy");
arraylist_dst.append("gl");
arraylist_dst.append("ka");
arraylist_dst.append("de");
arraylist_dst.append("el");
arraylist_dst.append("gn");
arraylist_dst.append("gu");
arraylist_dst.append("ht");
arraylist_dst.append("ha");
arraylist_dst.append("haw");
arraylist_dst.append("he");
arraylist_dst.append("hi");
arraylist_dst.append("hmn");
arraylist_dst.append("hu");
arraylist_dst.append("is");
arraylist_dst.append("ig");
arraylist_dst.append("ilo");
arraylist_dst.append("id");
arraylist_dst.append("ga");
arraylist_dst.append("it");
arraylist_dst.append("ja");
arraylist_dst.append("jv");
arraylist_dst.append("kn");
arraylist_dst.append("kk");
arraylist_dst.append("km");
arraylist_dst.append("rw");
arraylist_dst.append("kok");
arraylist_dst.append("ko");
arraylist_dst.append("kri");
arraylist_dst.append("kmr");
arraylist_dst.append("ckb");
arraylist_dst.append("ky");
arraylist_dst.append("lo");
arraylist_dst.append("la");
arraylist_dst.append("lv");
arraylist_dst.append("ln");
arraylist_dst.append("lt");
arraylist_dst.append("lg");
arraylist_dst.append("lb");
arraylist_dst.append("mk");
arraylist_dst.append("mg");
arraylist_dst.append("ms");
arraylist_dst.append("ml");
arraylist_dst.append("mt");
arraylist_dst.append("mi");
arraylist_dst.append("mr");
arraylist_dst.append("mni");
arraylist_dst.append("lus");
arraylist_dst.append("mn");
arraylist_dst.append("mmr");
arraylist_dst.append("ne");
arraylist_dst.append("no");
arraylist_dst.append("or");
arraylist_dst.append("om");
arraylist_dst.append("ps");
arraylist_dst.append("fa");
arraylist_dst.append("pl");
arraylist_dst.append("pt");
arraylist_dst.append("pa");
arraylist_dst.append("qu");
arraylist_dst.append("ro");
arraylist_dst.append("ru");
arraylist_dst.append("sm");
arraylist_dst.append("sa");
arraylist_dst.append("gd");
arraylist_dst.append("nso");
arraylist_dst.append("sr");
arraylist_dst.append("st");
arraylist_dst.append("sn");
arraylist_dst.append("sd");
arraylist_dst.append("si");
arraylist_dst.append("sk");
arraylist_dst.append("sl");
arraylist_dst.append("so");
arraylist_dst.append("es");
arraylist_dst.append("su");
arraylist_dst.append("sw");
arraylist_dst.append("sv");
arraylist_dst.append("tg");
arraylist_dst.append("ta");
arraylist_dst.append("tt");
arraylist_dst.append("te");
arraylist_dst.append("th");
arraylist_dst.append("ti");
arraylist_dst.append("ts");
arraylist_dst.append("tr");
arraylist_dst.append("tk");
arraylist_dst.append("tw");
arraylist_dst.append("ug");
arraylist_dst.append("uk");
arraylist_dst.append("ur");
arraylist_dst.append("uz");
arraylist_dst.append("vi");
arraylist_dst.append("cy");
arraylist_dst.append("xh");
arraylist_dst.append("yi");
arraylist_dst.append("yo");
arraylist_dst.append("zu");

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
arraylist_dst_languages.append("Bengali (Bangla)");
arraylist_dst_languages.append("Bhojpuri");
arraylist_dst_languages.append("Bosnian");
arraylist_dst_languages.append("Bulgarian");
arraylist_dst_languages.append("Catalan");
arraylist_dst_languages.append("Cebuano");
arraylist_dst_languages.append("Chichewa, Nyanja");
arraylist_dst_languages.append("Chinese (Simplified)");
arraylist_dst_languages.append("Chinese (Traditional)");
arraylist_dst_languages.append("Corsican");
arraylist_dst_languages.append("Croatian");
arraylist_dst_languages.append("Czech");
arraylist_dst_languages.append("Danish");
arraylist_dst_languages.append("Divehi, Maldivian");
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
arraylist_dst_languages.append("Kinyarwanda (Rwanda)");
arraylist_dst_languages.append("Konkani");
arraylist_dst_languages.append("Korean");
arraylist_dst_languages.append("Krio");
arraylist_dst_languages.append("Kurdish (Kurmanji)");
arraylist_dst_languages.append("Kurdish (Sorani)");
arraylist_dst_languages.append("Kyrgyz");
arraylist_dst_languages.append("Lao");
arraylist_dst_languages.append("Latin");
arraylist_dst_languages.append("Latvian (Lettish)");
arraylist_dst_languages.append("Lingala");
arraylist_dst_languages.append("Lithuanian");
arraylist_dst_languages.append("Luganda, Ganda");
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
arraylist_dst_languages.append("Oriya");
arraylist_dst_languages.append("Oromo (Afaan Oromo)");
arraylist_dst_languages.append("Pashto, Pushto");
arraylist_dst_languages.append("Persian (Farsi)");
arraylist_dst_languages.append("Polish");
arraylist_dst_languages.append("Portuguese");
arraylist_dst_languages.append("Punjabi (Eastern)");
arraylist_dst_languages.append("Quechua");
arraylist_dst_languages.append("Romanian, Moldavian");
arraylist_dst_languages.append("Russian");
arraylist_dst_languages.append("Samoan");
arraylist_dst_languages.append("Sanskrit");
arraylist_dst_languages.append("Scots Gaelic");
arraylist_dst_languages.append("Sepedi");
arraylist_dst_languages.append("Serbian");
arraylist_dst_languages.append("Sesotho");
arraylist_dst_languages.append("Shona");
arraylist_dst_languages.append("Sindhi");
arraylist_dst_languages.append("Sinhalese");
arraylist_dst_languages.append("Slovak");
arraylist_dst_languages.append("Slovenian");
arraylist_dst_languages.append("Somali");
arraylist_dst_languages.append("Spanish");
arraylist_dst_languages.append("Sundanese");
arraylist_dst_languages.append("Swahili (Kiswahili)");
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
arraylist_dst_languages.append("Twi");
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

listening_thread=None
text=''


#-------------------------------------------------------------MISC FUNCTIONS-------------------------------------------------------------#

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


def listen_worker_thread(src, dst):
    global main_window, listening_thread, text, recognizing, Device, SampleRate, Filename

    try:
        if SampleRate is None:
            #soundfile expects an int, sounddevice provides a float:
            device_info = sd.query_devices(Device, "input")
            SampleRate = int(device_info["default_samplerate"])
        
        if Filename:
            dump_fn = open(Filename, "wb")
        else:
            dump_fn = None

        model = Model(lang = src)

        with sd.RawInputStream(samplerate=SampleRate, blocksize = 8000, device=Device, dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, SampleRate)

            while recognizing==True:
                if recognizing==False:
                    rec = None
                    data = None
                    break

                data = q.get()
                results_text = ''
                partial_results_text = ''
                if rec.AcceptWaveform(data):
                    text = ((((((rec.Result().replace("text", "")).replace("{", "")).replace("}", "")).replace(":", "")).replace("partial", "")).replace("\"", "")).lower().strip()
                else:
                    text = ((((((rec.PartialResult().replace("text", "")).replace("{", "")).replace("}", "")).replace(":", "")).replace("partial", "")).replace("\"", "")).lower().strip()
                    if len(text)>0:
                        main_window.write_event_value('-VOICE-RECOGNIZED-', text)

                if dump_fn is not None:
                    dump_fn.write(data)

    except KeyboardInterrupt:
        recognizing==False
        rec = None
        data = None
        #print("\nDone")
        parser.exit(0)

    except Exception as e:
        parser.exit(type(e).__name__ + ": " + str(e))


def translate(phrase, src, dest):
    translator = Translator()
    translated_phrase = translator.translate(phrase, src=src, dest=dest).text
    return translated_phrase


def timed_translate(src, dst):
    global main_window

    WAIT_SECONDS=0.2
    phrase = str(main_window['-ML1-'].get())
    if (len(phrase)>0):
        translated_text = translate(phrase, src, dst)
        main_window.write_event_value('-VOICE-TRANSLATED-', translated_text)
    threading.Timer(WAIT_SECONDS, timed_translate, args=(src, dst)).start()


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


def make_overlay_voice_window(voice_text):
    xmax,ymax=sg.Window.get_screen_size()
    columns, rows = os.get_terminal_size()

    FONT_TYPE='Arial'
    if xmax>1280: FONT_SIZE=14
    if xmax<=1280: FONT_SIZE=16

    if len(voice_text)<=96:
        wszx=int(9.9*(xmax/1280)*len(voice_text) + 60)
        #wszx=font_length(voice_text, FONT_TYPE, FONT_SIZE)
    else:
        wszx=int(960*xmax/1280)

    nl=int((len(voice_text)/96)+1)
    #print('nl =', nl)

    if nl==1:
        LINE_SIZE=32
    else:
        LINE_SIZE=28*ymax/720

    wszy=int(nl*LINE_SIZE)

    #mlszx=int(0.15*wszx)
    #mlszy=int(0.018*wszy)

    wx=int((xmax-wszx)/2)
    wy=int(2*ymax/720)

    sg.set_options(font=(FONT_TYPE, FONT_SIZE))

    mlszx=len(voice_text)
    mlszy=nl

    if not (sys.platform == "win32"):
        layout = [[sg.Multiline(default_text=translated_text, size=(mlszx, mlszy), text_color='yellow1', border_width=0,
            background_color='black', no_scrollbar=True, justification='l', expand_x=True, expand_y=True,  key='-ML-LS1-')]]

        overlay_voice_window = sg.Window('voice_text', layout, no_titlebar=True, keep_on_top=True, size=(wszx,wszy), location=(wx,wy), 
            margins=(0, 0), background_color='black', alpha_channel=0.7, return_keyboard_events=True, finalize=True)

        #overlay_voice_window.set_alpha(0.6)

        overlay_translation_window.TKroot.attributes('-type', 'splash')
        overlay_translation_window.TKroot.attributes('-topmost', 1)
        #overlay_translation_window.TKroot.attributes('-topmost', 0)
    else:
        layout = [[sg.Multiline(default_text=voice_text, size=(mlszx, mlszy), text_color='yellow1', border_width=0,
            background_color='white', no_scrollbar=True, justification='l', expand_x=True, expand_y=True,  key='-ML-LS1-')]]

        overlay_voice_window = sg.Window('voice_text', layout, no_titlebar=True, keep_on_top=True, size=(wszx,wszy), location=(wx,wy), 
            margins=(0, 0), background_color='white', transparent_color='white', grab_anywhere=True, return_keyboard_events=True, 
            finalize=True)

    #TIMEOUT=50*len(voice_text)
    #TIMEOUT=500

    #event, values = overlay_voice_window.read(timeout=TIMEOUT, close=True)
    event, values = overlay_voice_window.read(500)

    return overlay_voice_window


def make_overlay_translation_window(translated_text):
    xmax,ymax=sg.Window.get_screen_size()
    columns, rows = os.get_terminal_size()

    FONT_TYPE='Arial'
    #if xmax>1280: FONT_SIZE=14
    #if xmax<=1280: FONT_SIZE=16
    FONT_SIZE=16

    if len(translated_text)<=96:
        wszx=int(9.9*(xmax/1280)*len(translated_text) + 60)
        #wszx=font_length(translated_text, FONT_TYPE, FONT_SIZE)
    else:
        wszx=int(960*xmax/1280)

    #wszx=int(960*xmax/1280)

    nl=int((len(translated_text)/96)+1)
    #print('nl =', nl)

    if nl==1:
        LINE_SIZE=32
    else:
        LINE_SIZE=28*ymax/720
    #LINE_SIZE=28*ymax/720

    wszy=int(nl*LINE_SIZE)

    #mlszx=int(0.15*wszx)
    #mlszy=int(0.018*wszy)

    wx=int((xmax-wszx)/2)

    if nl==1:
        wy=int(528*ymax/720)
    else:
        wy=int((528-(nl-1)*28)*ymax/720)
    #wy=int((528-(nl-1)*28)*ymax/720)

    if xmax>1280: FONT_SIZE=16
    if xmax<=1280: FONT_SIZE=16

    sg.set_options(font=("Arial", FONT_SIZE))

    mlszx=len(translated_text)
    mlszy=nl

    TRANSPARENT_BLACK='#add123'

    if not (sys.platform == "win32"):
        layout = [[sg.Multiline(default_text=translated_text, size=(mlszx, mlszy), text_color='yellow1', border_width=0,
            background_color='black', no_scrollbar=True,
            justification='l', expand_x=True, expand_y=True,  key='-ML-LS2-')]]

        overlay_translation_window = sg.Window('text', layout, no_titlebar=True, keep_on_top=True, size=(wszx,wszy), location=(wx,wy), 
            margins=(0, 0), background_color='black', alpha_channel=0.7, return_keyboard_events=True, finalize=True)

        overlay_translation_window.TKroot.attributes('-type', 'splash')
        overlay_translation_window.TKroot.attributes('-topmost', 1)
        #overlay_translation_window.TKroot.attributes('-topmost', 0)
    else:
        layout = [[sg.Multiline(default_text=translated_text, size=(mlszx, mlszy), text_color='yellow1', border_width=0,
            background_color='white', no_scrollbar=True,
            justification='l', expand_x=True, expand_y=True,  key='-ML-LS2-')]]

        overlay_translation_window = sg.Window('Translation', layout, no_titlebar=True, keep_on_top=True, size=(wszx,wszy), location=(wx,wy), 
            margins=(0, 0), background_color='white', transparent_color='white', grab_anywhere=True, return_keyboard_events=True, 
            finalize=True)

    #TIMEOUT=50*len(translated_text)
    #TIMEOUT=5000

    #event, values = overlay_translation_window.read(timeout=TIMEOUT, close=True)
    event, values = overlay_translation_window.read(500)

    return overlay_translation_window


#-------------------------------------------------------------MAIN PROGRAM-------------------------------------------------------------#

def main():
    global main_window, listening_thread, text, recognizing, Device, SampleRate, Filename

    #parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-S', '--src-language', help="Spoken language", default="en")
    parser.add_argument('-D', '--dst-language', help="Desired language for translation", default="id")
    parser.add_argument('-ll', '--list-languages', help="List all available source/destination languages", action='store_true')

    parser.add_argument("-ld", "--list-devices", action="store_true", help="show list of audio devices and exit")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])
    parser.add_argument("-f", "--filename", type=str, metavar="FILENAME", help="audio file to store recording to")
    parser.add_argument("-d", "--device", type=int_or_str, help="input device (numeric ID or substring)")
    parser.add_argument("-r", "--samplerate", type=int, help="sampling rate in Hertz for example 8000, 16000, 44100, or 48000")
    parser.add_argument('-v', '--version', action='version', version='0.0.6')
    args = parser.parse_args(remaining)
    args = parser.parse_args()

    if args.src_language:
        src = args.src_language

    if args.dst_language:
        dst = args.dst_language

    if args.src_language not in map_language_of_src.keys():
        print("Audio language not supported. Run with --list-languages to see all supported languages.")
        sys.exit(0)

    if args.dst_language not in map_language_of_dst.keys():
        print("Destination language not supported. Run with --list-languages to see all supported languages.")
        sys.exit(0)

    if args.list_languages:
        print("List of all languages:")
        for code, language in sorted(map_src_of_language.items()):
            print("{code}\t{language}".format(code=code, language=language))
        sys.exit(0)

    if args.device:
        Device = args.device

    if args.samplerate:
        SampleRate = args.samplerate

    if args.filename:
        Filename = args.filename


#-----------------------------------------------------------GUI STARTS-----------------------------------------------------------#

    xmax,ymax=sg.Window.get_screen_size()
    wsizex=int(0.5*xmax)
    wsizey=int(0.5*ymax)
    mlszx=int(0.15*wsizex)
    mlszy=int(0.018*wsizey)

    if not args.src_language==None:
        combo_src=map_language_of_src[args.src_language]
    else:
        combo_src='English'

    if not args.dst_language==None:
        combo_dst=map_language_of_dst[args.dst_language]
    else:
        combo_dst='Indonesian'

    layout = [[sg.Text('Click Start to start listening and printing it on screen', expand_x=True, expand_y=True, key='-MAIN-')],
              [sg.Text('Audio language      ', expand_x=True, expand_y=True),
               sg.Combo(list(map_src_of_language), default_value=combo_src, expand_x=True, expand_y=True, key='-SRC-')],
              [sg.Multiline(size=(mlszx, mlszy), expand_x=True, expand_y=True, key='-ML1-')],
              [sg.Text('Translation language', expand_x=True, expand_y=True),
               sg.Combo(list(map_dst_of_language), default_value=combo_dst, expand_x=True, expand_y=True, key='-DST-')],
              [sg.Multiline(size=(mlszx, mlszy), expand_x=True, expand_y=True, key='-ML2-')],
              [sg.Button('Start', expand_x=True, expand_y=True, button_color=('white', '#283b5b'), key='-START-BUTTON-'),
               sg.Button('Exit', expand_x=True, expand_y=True)]]

    main_window = sg.Window('VOSK Live Subtitles', layout, resizable=True, keep_on_top=True, finalize=True)
    main_window['-SRC-'].block_focus()

    if  (sys.platform == "win32"):
        main_window.TKroot.attributes('-topmost', True)
        main_window.TKroot.attributes('-topmost', False)

    if not (sys.platform == "win32"):
        main_window.TKroot.attributes('-topmost', 1)
        main_window.TKroot.attributes('-topmost', 0)

    overlay_translation_window = None
    recognizing = False


#---------------------------------------------------------------MAIN LOOP--------------------------------------------------------------#

    while True:
        window, event, values = sg.read_all_windows(500)

        #print('event =', event)
        #print('recognizing =', recognizing)

        src=map_src_of_language[str(main_window['-SRC-'].get())]
        dst=map_dst_of_language[str(main_window['-DST-'].get())]

        if event == 'Exit' or sg.WIN_CLOSED:
            break

        elif event == '-START-BUTTON-':
            recognizing = not recognizing
            #print('Start button clicked, changing recognizing status')
            #print('recognizing =', recognizing)
            main_window['-START-BUTTON-'].update(('Stop','Start')[not recognizing], button_color=(('white', ('red', '#283b5b')[not recognizing])))

            if recognizing:
                #print('VOSK Live Subtitle is START LISTENING now')
                #print('recognizing =', recognizing)

                if not overlay_translation_window:
                    overlay_translation_window = make_overlay_translation_window(100*' ')
                    overlay_translation_window.Hide()

                listening_thread=threading.Thread(target=listen_worker_thread, args=(src,dst), daemon=True)
                listening_thread.start()

                timer = threading.Thread(target=timed_translate, args=(src, dst), daemon=True)
                timer.start()

            else:
                #print('VOSK Live Subtitle is STOP LISTENING now')
                #print('recognizing =', recognizing)

                text=''
                main_window['-ML1-'].update(text,background_color_for_value='yellow1',autoscroll=True)
                translated_text=''
                main_window['-ML2-'].update(translated_text,background_color_for_value='yellow1',autoscroll=True)

                if overlay_translation_window:
                    overlay_translation_window.close()
                    overlay_translation_window=None

            if not (sys.platform == "win32"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event=='-VOICE-RECOGNIZED-' and recognizing==True:
            text=str(values[event]).strip().lower()
            main_window['-ML1-'].update(text,background_color_for_value='yellow1',autoscroll=True)
            
            #if overlay_voice_window: overlay_voice_window['-ML-LS1-'].update(text,background_color_for_value='black',autoscroll=True)

            if not (sys.platform == "win32"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)


        elif event=='-VOICE-TRANSLATED-' and recognizing==True:
            overlay_translation_window.UnHide()
            translated_text=str(values[event]).strip().lower()
            main_window['-ML2-'].update(translated_text,background_color_for_value='yellow1',autoscroll=True)
            
            if (overlay_translation_window):
                overlay_translation_window['-ML-LS2-'].update(translated_text,background_color_for_value='black',autoscroll=True)

            if not (sys.platform == "win32"):
                main_window.TKroot.attributes('-topmost', 1)
                main_window.TKroot.attributes('-topmost', 0)

    main_window.close()


if __name__ == "__main__":
    main()
