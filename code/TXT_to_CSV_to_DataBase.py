# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:33:32 2016
@author: kvdweiden,ibrahima (symao)
"""

import pandas as pd
import numpy as np
import pandas.io.sql as sql
import mysql.connector
import os 
from os import chdir
import sys
chdir("/home/khalil/Bureau/testsy")
try :
   fichiertxt=raw_input("veuillez introduire  un fichier txt: ")
   cFicIn = open(fichiertxt,'r')
except IOError :
     print "Ce fichier n\'existe pas dans le répertoire"
     sys.exit(1)
except NameError:
     print "Verifier que le fichier existe"
     sys.exit(1) 
    
def saisi():
    num_array = list()
    num_mes=list()
    numM=raw_input("Veuillez saisir le numero de mesure : ")
    num = raw_input("Veuillez saisir le nombre d'echantillons: ")
    for i in range(int(num)):
        n = raw_input("num :")
        num_array.append(n)
        num_mes.append(numM)
    
    #print 'ARRAY: ',num_array
    #print 'mesure: ',num_mes
    return num_mes,num_array
fichiercsv="2071test.csv"
#cFicIn = open(fichiertxt,'r')
csv_file = open(fichiercsv,'w')



try:
    from itertools import izip_longest  # added in Py 2.6
except ImportError:
    from itertools import zip_longest as izip_longest  # name change in Py 3.x

try:
    from itertools import accumulate  # added in Py 3.2
except ImportError:
    def accumulate(iterable):
        'Return running totals (simplified version).'
        total = next(iterable)
        yield total
        for value in iterable:
            total += value
            yield total

def make_parser(fieldwidths):
    cuts = tuple(cut for cut in accumulate(abs(fw) for fw in fieldwidths))
    pads = tuple(fw < 0 for fw in fieldwidths) # bool values for padding fields
    flds = tuple(izip_longest(pads, (0,)+cuts, cuts))[:-1]  # ignore final one
    parse = lambda line: tuple(line[i:j] for pad, i, j in flds if not pad)
    # optional informational function attributes
    parse.size = sum(abs(fw) for fw in fieldwidths)
    parse.fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's')
                                                for fw in fieldwidths)
    return parse
    
def csv_to_database(fichir,mesu,nesp):
  try:  
    df = pd.read_csv(fichir,usecols=[0,1,2],
    names=["Frequency","Peak","phase"])
    df=df.dropna(how ="all")
    #print df
     
    s=df["Frequency"]
    s0=s[0]
    #es=df["espece"]
    cpt2=0
    for a in df["Frequency"] :
        if a==s0:
            cpt2=cpt2+1
    print("Dans ce fichier il y\'a %s especes "% cpt2)
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
    
    dfok["espece"]=nesp
    dfok["NumeroM"]=mesu
    dfok["classe"]=[0,1,2,3]
    return dfok
  except ValueError:
        print "saisir correctement le numero des echantillons"
  except IndexError:
         print "Fichier vide"
    
  
     


try:
  line=cFicIn.readline()
  
  a=0
  copy = False    
  for line in cFicIn:

          
     fieldwidths = (22, 22, 22)  # negative widths represent ignored padding fields
     parse = make_parser(fieldwidths)
     fields = parse(line)
    
     if line.strip() == "Frequency [Hz]           Peak amplitude           Phase [degrees]":
        bucket = []
        copy = True

     elif line.strip() == "Peak interpolation: Numeric":
        for strings in bucket:
            csv_file.write( strings + '\n')
        copy = False

     elif copy:
        bucket.append(fields[0].strip()+','+fields[1].strip()+','+fields[2].strip()+','+'\n')
except NameError ,e:
     if e:
      print "Verifier que le fichier existe"
     
csv_file.close()

#converter csv to database mysql passing in dataframe
mesur,ne=saisi()
df=csv_to_database(fichiercsv,mesur ,ne)
"""conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                           database='symao',
                                           user='symao',
                                           password='symao')"""


conn = mysql.connector.connect(host='localhost',port='3306',
                                       database='gaulois',
                                       user='root',
                                       password='773221024')    
if conn.is_connected():
    print('Connected to MySQL database')
try :
  df.to_sql(con=conn, name='spectre1', if_exists='append', flavor='mysql',index=False)
except AttributeError:
    print "Le fichier n\'est pas enregistré dans la base"
except FutureWarning:
    print ""
else:
    print "Le fichier est enregistré avec succés"
os.remove("2071test.csv")
#print sql.read_frame("select * from spectretest",conn)
#
        
        
        
        
