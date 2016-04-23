# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:26:24 2016

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
df = pd.read_csv("test_lait.csv",skiprows=1,
header=None,usecols=[1,2,3],
names=["Frequency","Peak","phase"])
#df=df.ix[:,1:4]
print df
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

dfh=dfok[["Am0","Ph0"]]
dfh["classe"]=[0,1,2,3]
print dfok
#dfh.plot(kind='bar',stacked=False,alpha=0.5)
#dfh.boxplot()
#plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=2).fit_transform(dfh)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1],c=dfh.ix[:,2],
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
#ax.set_xlabel("freq")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("peak")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("phase")
ax.w_zaxis.set_ticklabels([])
plt.show()
plt.savefig("test1.png")
#print f
#corre=dfh.corr()
#print corre
#print cor
#corre.plot(kind='bar',stacked=True)
"""corr = dfh.corr().as_matrix()
axes = scatter_matrix(df1, alpha=0.5, diagonal='kde')
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.show()"""
X=dfh[["Am0","Ph0"]]
y=dfh["classe"]
#see the features importance
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X,y)
importance=rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importance)[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importance[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


