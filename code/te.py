
import csv
import struct
import pandas as pd
import numpy as np
import matplotlib
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
print df