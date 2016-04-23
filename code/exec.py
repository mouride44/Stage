# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:52:36 2016

@author: khalil
"""

from cx_Freeze import setup, Executable

# On appelle la fonction setup
setup(
    name = "votre_programme",
    version = "1",
    description = "Votre programme",
    executables = [Executable("TXT_to_CSV_to_DataBase.py")],
)