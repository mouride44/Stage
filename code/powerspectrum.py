# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:41:12 2016

@author: khalil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as sql
import mysql.connector
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                           database='symao',
                                           user='symao',
                                           password='symao')
    
if conn.is_connected():
    print('Connected to MySQL database')
#df.to_sql(con=conn, name='spectretest', if_exists='append', flavor='mysql',index=False)
df=sql.read_frame("select * from spectre",conn)
df["target"]=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
df=df[df.target!=0]
df_A=df.iloc[:,20:40]
print df_A.columns
#x=df.iloc[:4,0:20]
#plt.matshow(df_A.corr())

#print df_A.corr()
y=df["target"]
cr=df_A.corr()
"""cr.plot(kind='barh',stacked=True)
print cr
from matplotlib import cm as cm

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
cax = ax1.matshow(df_A.corr(), cmap=cmap)
ax1.grid(True)
plt.title(' Feature Correlation')
labels=df_A.columns
ax1.set_xticklabels(labels,fontsize=10)
ax1.set_yticklabels(labels,fontsize=10)
ax1.set_xlim(1,8)
ax1.set_ylim(1,8)
# Add colrbar, mak sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax)
plt.savefig("touba2.png")
plt.show()

dte= (df.iloc[:3,0:20]).T
dte.index=xrange(20)
dte.columns=["freq1","freq2","freq3"]
dft= df_A.T
dft.index=xrange(20)
dft.columns=["he1","he2","he3"]
dfpl=pd.concat([dft,dte],axis=1,ignore_index=False)
dfpl.plot(x="freq1",y=["he1","he2","he3"])
print dfpl"""
X=df.iloc[:,20:40]
X=X[["Am1","Am9","Am17"]]
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("Am1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Am9")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Am17")
ax.w_zaxis.set_ticklabels([])
plt.show()
#Representation 3 d des Amplitudes
#feature importance 

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,
test_size=0.25,random_state=11)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=11)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
#plt.savefig("touba3.png")
plt.show()


############################"""""""ploy""""""""""""""##############"""""""

