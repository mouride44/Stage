# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:52:36 2016

@author: khalil
"""

#from cx_Freeze import setup, Executable

# On appelle la fonction setup
import sys
from cx_Freeze import setup, Executable
"""base=None
# Dependencies are automatically detected, but it might need fine tuning.
#build_exe_options = {"packages": ["os"], "includes": ["pandas"]}
options = dict(compressed=True,includes=['numpy'],excludes=['Tkinter','tcl','ttk','tkinter‌​'],optimize=2)
setup( name = "numpybug", version = "0.1", description = "Sample cx_Freeze script", options=dict(build_exe=options), executables = [Executable("TXT_to_CSV_to_DataBase.py", base=base)])
# GUI applications require a different base on Windows (the default is for a
# console application)"""
base = None
build_exe_options = {"packages": ["os"], "includes": ["pandas"]}
setup(  name = "diao",
        version = "0.1",
        description = "application!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("TXT_to_CSV_to_DataBase.py", base=base)])

