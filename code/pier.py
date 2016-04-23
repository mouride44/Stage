# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:33:52 2016

@author: khalil
"""


import csv
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as sql
import mysql.connector
conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                           database='symao',
                                           user='symao',
                                           password='symao')
    
if conn.is_connected():
    print('Connected to MySQL database')
#df.to_sql(con=conn, name='spectretest', if_exists='append', flavor='mysql',index=False)
df=sql.read_frame("select * from spectre",conn)

df["target"]=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
df=df[df.target==0]
df_A=df.iloc[:,20:36]
dfe=df_A.T
dff=df.iloc[:,0:20]
dfh=df.iloc[:,20:40]
print dff.var()
dfh=dfh.drop_duplicates()
#dfh.var().plot(kind='bar')

#ax = dfh.var().plot(kind='bar')
#fig = ax.get_figure()
#fig.savefig('water5.png')
d=list()
print len(dfh.index)
for row in (xrange(len(dfh.index))):
   print row
   d.append(sum(dfh.iloc[row,::]))
print d
dfh["Energie"]=d
dfen=dfh["Energie"]
dfen.index = ['eau'+ str(i) for i in range(len(dfen.index))]
print dfen
#ax=dfen.plot(kind="bar")
# s is an instance of Series
#fig = ax.get_figure()
#fig.savefig('water2.png')
pier=list()
for i in (xrange(len(dfen.index))):
    pier.append( (dfen.iloc[i]- dfen.iloc[0])*100/ dfen.iloc[0])     
dfh["pourc"]=pier
print dfen
dfp=dfh["pourc"]
dfp.index = ['perror0'+ str(i) for i in range(len(dfen.index))]
#ax1=dfp.plot(kind="bar")
# s is an instance of Series
#fig = ax1.get_figure()
#fig.savefig('waterp.png')
"""dfe.columns=["ech1","ech2","ech3","ech4","ech5","ech6","ech7"]
dfe=dfe.iloc[:,0:6]

#dfe["ech12"]=dfe["ech1"]-dfe["ech2"]
dfe["ech13"]=abs(dfe["ech1"]-dfe["ech3"])
dfe["ech14"]=abs(dfe["ech1"]-dfe["ech4"])
dfe["ech15"]=abs(dfe["ech1"]-dfe["ech5"])
dfe["ech16"]=abs(dfe["ech1"]-dfe["ech6"])
#dfe["ech23"]=dfe["ech2"]-dfe["ech3"]
#dfe["ech24"]=dfe["ech2"]-dfe["ech4"]
#dfe["ech25"]=dfe["ech2"]-dfe["ech5"]
#dfe["ech26"]=dfe["ech2"]-dfe["ech6"]
dfe["ech34"]=abs(dfe["ech3"]-dfe["ech4"])
dfe["ech35"]=abs(dfe["ech3"]-dfe["ech5"])
dfe["ech36"]=abs(dfe["ech3"]-dfe["ech6"])
dfe["ech45"]=abs(dfe["ech4"]-dfe["ech5"])
dfe["ech56"]=abs(dfe["ech5"]-dfe["ech6"])
df12=dfe.iloc[:,6:19]
#df12.plot(kind='bar')
ax = df12.plot(kind='bar') # s is an instance of Series
fig = ax.get_figure()
fig.savefig('water2.png')"""
dfh=dfh.iloc[:,0:20]
dg1=dfh.iloc[:,0:5]
dg2=dfh.iloc[:,5:10]
dg3=dfh.iloc[:,10:15]
dg4=dfh.iloc[:,15:20]
print dg4
d1=list()
for row in (xrange(len(dg2.index))):
   print row
   d1.append(sum(dg4.iloc[row,::]))
dg4["eng1"]=d1
dg4.index = ['Eng4_'+ str(i) for i in range(len(dg4.index))]
ax3 = dg4["eng1" ].plot(kind='bar') # s is an instance of Series
fig = ax3.get_figure()
fig.savefig('Eng4.png')
print dg1

   


