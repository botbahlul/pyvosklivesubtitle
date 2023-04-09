from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ['streamlink']
datas = collect_data_files('streamlink.plugins')
