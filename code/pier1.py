# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:29:29 2016

@author: khalil
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as sql
import mysql.connector

conn2 = mysql.connector.connect(host='localhost',port='3306',
                                       database='gaulois',
                                       user='root',
                                       password='773221024')
df=sql.read_frame("select * from spectre1",conn2)
print df
df1=df[df.classe==2].iloc[:,20:40]
df=df[df.classe==1]
df_A=df.iloc[:,20:40]
print df_A
d=list()
for row in (xrange(len(df_A.index))):
   print row
   d.append(sum(df_A.iloc[row,::]))
df_A["energie1"]=d

df_A.index=df.classe
#df_A.energie.plot(kind='bar')
#df_A.energie.plot(kind='bar')
d1=list()
for row in (xrange(len(df1.index))):
   print row
   d1.append(sum(df1.iloc[row,::]))
df1["energie2"]=d1
print df1
df12=df_A[["energie1"]]
df12.index=xrange(len(df12.index))

print df12

