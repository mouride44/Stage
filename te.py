
import csv
import struct
import pandas as pd
import numpy as np
import matplotlib
import pandas.io.sql as sql
import mysql.connector
import matplotlib.pyplot as plt
conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                           database='symao',
                                           user='symao',
                                           password='symao')

"""conn = mysql.connector.connect(host='localhost',port='3306',
                                       database='gaulois',
                                       user='root',
                                       password='773221024') """

if conn.is_connected():
    print('Connected to MySQL database')
#df.to_sql(con=conn, name='spectretest', if_exists='append', flavor='mysql',index=False)
df=sql.read_frame("select * from spectreok",conn)
df["calib"]=[2066,2066,2066,2066,2066,2066,2066,2066,2077,2077,2077,2077]
fg=np.array_split(df,4)
groupe=df.groupby(["calib","reglage"])
#tab array in grouP
tab=list()
print groupe.groups
for name, group in groupe:
    tab.append(group)
for i in tab:
    i.index=i.classe
tabb=list()
for j in tab :
     if (len(j)>4) :
         tabb.append(j)  
ob=list()
for row in tab:
    e=np.array_split(row,len(row)/4)   
    ob.append(e) 
print ob
"""for row  in xrange(len(tab)):
      tabg=list()
      tab1=tab[row].iloc[:,20:40]
      
      dg1=tab1.iloc[:,0:5]
      
      dg1["energie"]=dg1.sum(axis=1)
      dg1["percent"]=(dg1["energie"] -dg1.energie.iloc[0])*100/dg1.energie.iloc[0]
      print dg1
      dg2=tab1.iloc[:,5:10]
      dg2["energie"]=dg2.sum(axis=1)
      dg2["percent"]=(dg2["energie"] -dg2.energie.iloc[0])*100/dg2.energie.iloc[0]
      dg3=tab1.iloc[:,10:15]
      dg3["energie"]=dg3.sum(axis=1)
      dg3["percent"]=(dg3["energie"] -dg3.energie.iloc[0])*100/dg3.energie.iloc[0]
      dg4=tab1.iloc[:,15:20]
      dg4["energie"]=dg4.sum(axis=1)
      dg4["percent"]=(dg4["energie"] -dg4.energie.iloc[0])*100/dg4.energie.iloc[0]
      fig, axes = plt.subplots(nrows=2, ncols=2)
      dg1.percent.plot(kind='bar',ax=axes[0,0],color="rgb")
      dg2.percent.plot(kind='bar',ax=axes[0,1])
      dg3.percent.plot(kind='bar',ax=axes[1,0])
      dg4.percent.plot(kind='bar',ax=axes[1,1])
      plt.savefig("percentage"+str(row))"""
      