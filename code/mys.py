# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:34:26 2016

@author: khalil
"""

import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas.io.sql as sql
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# tester la lecture
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                       database='symao',
                                       user='symao',
                                       password='symao')

if conn.is_connected():
    print('Connected to MySQL database')
conn2 = mysql.connector.connect(host='localhost',port='3306',
                                       database='gaulois',
                                       user='root',
                                       password='773221024')
if conn2.is_connected():
    print('Connected to MySQL database')

df = pd.read_csv("lait2.csv")
cursor = conn.cursor()
#df.to_sql(con=conn, name='tab1', if_exists='append', flavor='mysql',index=False)
#cursor.execute("""DROP TABLE tab1""")
df2=sql.read_frame("select * from tab1",conn2)
df3=df2.dropna(how='all')
df4=df3.dropna()
#print df4
df4["CelluleQ"]=pd.qcut(df4.Cellules,3,labels=["g1","g2","g3"])
#print pd.value_counts(df4["CelluleQ"])
g1=df4[df4.CelluleQ=="g1"]
g2=df4[df4.CelluleQ=="g2"]
g3=df4[df4.CelluleQ=="g3"]
g3["target"]=2
g1["target"]=0
g2["target"]=1

dfok=pd.concat([g1,g2,g3],axis=0,ignore_index=True)
dfq=dfok.ix[:,5:12]
df_q=dfq.drop('CelluleQ',axis=1)
print df_q
#plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(df_q)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=dfq.ix[:,6],
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("Lactose")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Cellules")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Uree")
ax.w_zaxis.set_ticklabels([])
#plt.savefig("touba.png")
#plt.show()

#the part of machine leaning
#cross validation
X=df_q.drop(["target"],axis=1)
y=df_q["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,
test_size=0.25,random_state=11)
print X_train
print y_train
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
#regression logistique
#combinaisons de paramètres à évaluer
parametres = [{'C':[0.1,1,10],'kernel':['rbf','linear']}]
logit = LogisticRegression()
titan_logit=logit.fit(X_train, y_train)
# Erreur
scl=titan_logit.score(X_test, y_test)


clf=svm.SVC()
grid = GridSearchCV(estimator=clf,param_grid=parametres,scoring='accuracy')
grille = grid.fit(X_train,y_train)
print "socore de ", grille.grid_scores_
print "best parametre",grille.best_params_
#meilleur performance – estimée en interne par validation croisée
print "best performance in cross validation",grid.best_score_
y_pred3 = grille.predict(X_test)
 
clfov = svm.SVC(decision_function_shape='ovo')
clf1=svm.SVC(kernel='linear')
clf1.fit(X_train,y_train)
fd=clf.fit(X_train,y_train)
fd=clfov.fit(X_train,y_train)
#prediction
y_pred=logit.predict(X_test)
matrix=confusion_matrix(y_test,y_pred)
#la performance
sv= clf.score(X_test,y_test)
sv2=clf1.score(X_test,y_test)
sv4=clfov.score(X_test,y_test)
cfr=metrics.accuracy_score(y_test,y_pred3)

print('The matrix of confusion')
print matrix
table=pd.crosstab(y_test,y_pred)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
print len(y)
#testons the k fold cross validation
kf=KFold(len(X),n_folds=3,shuffle=True)#,random_state=)
print kf
means=[]
for trai,test in kf:
    logit.fit(X.ix[trai], y.ix[trai])    
    y_pred=logit.predict(X.ix[test])
   # curmean = np.mean(y_pred == y[test])
   # X_train, X_test = X.ix[trai], X.ix[test]
   # print curmean
    #means.append(curmean)
    means.append(logit.fit(X_train, y_train).score(X_test, y_test))
print means    
#print("Mean accuracy: {:.1%}".format(np.mean(means)))
    
    
#plt.colorbartr
#print titan_logit.coef_"
# Coefficientss
#titan_logit.coef
#print dfq
#renommer les noms des variables
df.rename(columns={"N° d'élevage":"numeroEl","N° de position dans l'élevage":"numeroPos","Date d'analyse":"dataA","Urée":"Uree","Acétone":"Acetone"},inplace=True)
#print df2.columns
#df2.to_sql(con=conn, name='tableok', if_exists='append', flavor='mysql',index=False)
#change les clé
#df3=sql.read_frame("select * from tableok",conn)
#df3.to_sql(con=conn, name='ho', if_exists='append', flavor='mysql',index=False)
#print sql.read_frame("select * from ho",conn)
#print df
#cursor.execute("ALTER TABLE ho ADD PRIMARY KEY(numeroEl, dataA)")
#df=df.drop_duplicates()
#df=df.dropna()
df["num"]=df.index
#cursor.execute("DROP TABLE espece_lait")
#df.to_sql(con=conn, name='espece_lait', if_exists='append', flavor='mysql',index=False)
#cursor.execute("ALTER TABLE espece_lait ADD PRIMARY KEY(num,numeroEl,dataA)")
#crer un autre table qui prend les table 
#crer la table a inserer
#print df
#print sql.read_frame("select * from prudh",conn)
#print df
"""crer table liaison portant les deux mesure"""
cursor.execute("""
CREATE TABLE IF NOT EXISTS scale(
    id bigint(20) NOT NULL,
    numE varchar(63) NOT NULL ,
    day varchar(63) NOT NULL ,
    
    FOREIGN KEY (id,numE,day) references espece_lait(num,numeroEl,dataA))
    
    ENGINE=innoDB;
    """)
#print sql.read_frame("select * from Mesur",conn)
#cursor.execute("""INSERT INTO scale (id, numE, day) SELECT num, numeroEl, dataA FROM espece_lait""")
#print sql.read_frame("select * from espece_lait",conn)
#features importances 
"""import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

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
plt.savefig("touba1.png")
plt.show()"""
#test anova
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline



# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) svm
clfan = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)
print "le score regression univarie" ,anova_svm.score(X_test,y_test)

#fusionner deux fonctions pour selectionner les meillleurs variables
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest



# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X_train, y_train).transform(X)

svm = SVC(kernel="linear")
######################################################################
# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X_train, y_train)
yp=grid_search.predict(X_test)
cp=metrics.accuracy_score(y_test,y_pred3)
print "score features unions" ,cp
#print(grid_search.best_estimator_)
#################featues impotance and predict ######################################
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

cfpre=ExtraTreesClassifier()
cfpre=cfpre.fit(X_train ,y_train)
print cfpre.feature_importances_
model = SelectFromModel(cfpre, prefit=True)
X_new=model.transform(X)
X_trainn,X_testt,y_trainn,y_testt=train_test_split(X_new,y,
test_size=0.25,random_state=11)
clfn=SVC(kernel='linear')
clfnew=clfn.fit(X_trainn,y_trainn)
#########score####
print ("The score to svm dbf algorithm",sv)
print ("The score to svm grsearch use dbf algorithm",cfr)

print ("The score to svm linear algorithm",sv2)
print ("The score to svm ovo",sv4)
print ("The score to Regression logistic algorithm",scl)

print ("the score to use selectfromodel with svclinear",clfn.score(X_testt,y_testt))

print( "the score to use featureunion pca_and_reguniv with svclinear" ,cp)


