# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:22:21 2016

@author: khalil
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import scatter_matrix
import pandas.io.sql as sql
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
# tester la lecture
df = pd.read_csv("testok.csv",usecols=[1,2,3])
print df
