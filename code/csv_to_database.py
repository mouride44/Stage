# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:42:25 2016

@author: khalil
"""

import pandas as pd
import numpy as np
import sqlite3 as sqlite
import csv
import pandas.io.sql as sql
import mysql.connector
fichier='testok.csv'
hoste='localhost'
por='3306'
datab='gaulois'
users='root'
passewords='773221024'
table='dmu'
def open_csv(fichier):
    df=pd.read_csv(fichier,usecols=[1,2,3])
    df2=df.dropna(how='all')
    return df2
def espece_count(frames):

    s= frames["Frequency"]
    s0=s[0]
    cpt2=0
    for a in frames["Frequency"] :
       if a==s0:
          cpt2=cpt2+1
    return cpt2
          
def ajout_index(datf)  :
    datf["indexes"]=datf.index
    
    
def split_table(fichier,esp):
      dataframe=open_csv(fichier)
      cpt=espece_count(dataframe)
      f=np.array_split(dataframe, cpt)
      
      #tableau contenant la tranpose des dataframe des pha
      for row in f :
          row["Harmonie"]=xrange(20)  
      tab=range(esp+1,(cpt+esp)+1,1)
      for a in xrange(cpt):
              f[a]["espece"]='Ec'+ str(tab[a])
              
      df1=pd.concat(f) 
      return df1
      
def connect_database(data) : 
     connection=sqlite.connect(data) 
     return connection  
     
def split_tablef(fichier) :
      dataframe=open_csv(fichier)
      cpt=espece_count(dataframe)
      f=np.array_split(dataframe, 4)
      
      #tableau contenant la tranpose des dataframe des pha
      for row in f :
          row["Harmonie"]=xrange(20)  
      for a in xrange(cpt):
              f[a]["espece"]='Ec'+ str(a)          
      df1=pd.concat(f) 
      return df1    
      
def first_table(fichier,connection,names):
    datafirst=split_tablef(fichier)
    datafirst["indexes"]=datafirst.index
    datafirst.to_sql(con=connection, name=names, if_exists='append', flavor='mysql',index=True)
    
def frame_to_sql(frame,connection,tab) :
     frame.to_sql(con=connection, name=tab, if_exists='append', flavor='mysql',index=False)
     
def max_index  (con,tableau)  :
    con.execute("SELECT max(indexes) FROM %s"%(tableau,))
    max_id = con.fetchone()[0]
    return max_id 
def change_index(fram,m):
    inf=[]
    n=m+1
    taille=((fram.index.shape[0]+m))+1
    for n in range(n,taille,1):
        inf.append(n)        
    fram["indexes"]=inf
    return fram
def check_table(cursor):
    cursor.execute(""" SELECT COUNT(*) FROM sqlite_master WHERE name = ?  """, ('tab1', ))
    res =cursor.fetchone()
    return  bool(res[0]) # True if exists
def check_table_my(cursor,tablename):
    cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name =%s ",[tablename])
    if cursor.fetchone()[0] == 1:
         cursor.close()
         return True

    cursor.close()
    return False 
def recupere_espece(cursor,maxid,table):
    cursor.execute("SELECT espece  FROM %s WHERE  indexes=%s"%(table,maxid,))
    es=cursor.fetchone()[0]

    es.split()
    return int(es[2])

    #num=response[0]
   # num.split()
   # return int(num[2])
def connect_mysql(hosts,ports,databases,users,passewords):
     conn = mysql.connector.connect(host=hosts,port=ports,
                                       database=databases,
                                       user=users,
                                       password=passewords)
     if conn.is_connected():
        print('Connected to MySQL database')
     return conn
    

#The firt table inirst_t(.indexable(fichier,data)da the ba11se
#connet=connect_database(data)
#conn=connet.cursor()
#f=check_table(conn)
connect_my=connect_mysql(hoste,por,datab,users,passewords)
conn=connect_my.cursor()
f=check_table_my(conn,table)

#df2=sql.read_frame("select * from tab1",connect_my)
print f
df=open_csv(fichier)
print df
print espece_count(df)
f2=sql.read_frame("select * from dmu",connect_my)
print f2
#on cree le nouveau table
if f==False: 
    
     first_table(fichier,connect_my,table)
     print("table créé")
    
else :  
    connect_my=connect_mysql(hoste,por,datab,users,passewords)
    conn=connect_my.cursor()
    maxi=max_index (conn,table)
    esp= recupere_espece(conn,maxi,table)
    print maxi
    print esp
    
    
    spl=split_table(fichier,esp)
    ajout_index(spl)
    info=change_index(spl,maxi)
  
   
    frame_to_sql(info,connect_my,table) 
print info 