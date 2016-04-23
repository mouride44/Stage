# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:26:07 2016

@author: khalil
"""

import pandas as pd
import numpy as np
import matplotlib
import pandas.io.sql as sql
import mysql.connector
# tester la lecture
df = pd.read_csv("20659test.csv",usecols=[0,1,2],
names=["Frequency","Peak","phase"])
df=df.dropna(how ="all")
print df
 
s=df["Frequency"]
s0=s[0]
#es=df["espece"]
cpt2=0
for a in df["Frequency"] :
    if a==s0:
        cpt2=cpt2+1
print("Dans ce fichier il y a",cpt2,"especes sur",df.index.shape,"frequence")
#on va diviser le tableau en cpt

infor=[]
e=np.array_split(df[["phase"]], cpt2)
f=np.array_split(df[["Peak"]], cpt2)
fr=np.array_split(df[["Frequency"]], cpt2)
#tableau contenant la tranpose des dataframe des phase
freq=[]
for rowf in fr:
    freq.append(rowf.T)
c=freq[0]
tabf=['freq'+str(i) for i in range(len(c.columns))]    
for b in freq:
    b.columns=tabf
dff=pd.concat(freq)
dff.index = ['esp'+ str(i) for i in range(len(dff.index))]
#print dff
for row1 in e :
    infor.append(row1.T)
#tableau contenant dataframe des amplutudes#print infor    
infg=[]
for row2 in f:
    infg.append(row2.T)
b=infg[0]
tab=['Am'+ str(i) for i in range(len(b.columns))]
for row3 in infg:
    row3.columns=tab
df1=pd.concat(infg)
#print df1
df1.index = ['esp'+ str(i) for i in range(len(df1.index))]
#print df1
tab1=['Ph'+ str(i) for i in range(len(b.columns))]
for row4 in infor :
     row4.columns=tab1
df2=pd.concat(infor)
df2.index = ['esp'+ str(i) for i in range(len(df2.index))]
#print df2
#ax1=fig.add_subplot(2,2,1)
dfok=pd.concat([dff,df1,df2],axis=1)
dfok["espece"]=['w',245,255,265]
dfok["NumeroM"]=[2070,2070,2070,2070]
conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                       database='symao',
                                       user='symao',
                                       password='symao')

if conn.is_connected():
    print('Connected to MySQL database')
#dfok.to_sql(con=conn, name='spectre', if_exists='append', flavor='mysql',index=False)

print dfok