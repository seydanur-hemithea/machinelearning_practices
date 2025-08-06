# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 17:02:23 2025

@author: asus
"""

from sklearn.datasets import make_classification,make_moons,make_circles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
X,y=make_classification(n_features=2, n_redundant=0,n_informative=2,
                        n_clusters_per_class=1,random_state=42)
X+=1.2*np.random.uniform(size=X.shape)
Xy=(X,y)
# plt.scatter(X[:,0],X[:,1],c=y)

# X,y=make_moons(noise=0.2,random_state=42)
# plt.scatter(X[:,0],X[:,1],c=y)
# X,y=make_circles(noise=0.1, factor=0.3,random_state=42)
# plt.scatter(X[:,0],X[:,1],c=y)
datasets=[Xy,
          make_moons(noise=0.2, random_state=42,),
          make_circles(noise=0.1, factor=0.3,random_state=42)]
fig=plt.figure(figsize=(6,0))
i=1
for ds_ct,ds in enumerate(datasets):
    X,y=ds
    # if ds_ct==0:
    #     colors="darkred"
    # elif ds_ct==1:
    #     colors="darkblue"
    # else:
    #     colors="darkgreen"
    ax=plt.subplot(len(datasets),1,i)
   
    ax.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.coolwarm,edgecolors="black")
    i+=1  
plt.show()

names=["Nearest Neighbors","Linear SVN","Decision Tree","Random Forest","Naive Baayes"]
classifiers=[KNeighborsClassifier(),SVC(),DecisionTreeClassifier(), RandomForestClassifier (),GaussianNB()]
fig=plt.figure(figsize=(6,9))
i=1
for ds_ct,ds in enumerate(datasets):
    X,y=ds
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    cm_bright= ListedColormap(["darkred","darkblue"])
    ax=plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_ct==0:
        ax.set_title("input data")
    ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,edgecolors="black",alpha=0.6)
    i+=1
    for name,cf in zip(names,classifiers):
        ax=plt.subplot(len(datasets),len(classifiers)+1,i)
        
    
        cf= make_pipeline(StandardScaler(),cf)
        cf.fit(X_train,y_train)
        score=cf.score(X_test,y_test)
        DecisionBoundaryDisplay.from_estimator(cf, X, cmap=plt.cm.RdBu,alpha=0.7,ax=ax,eps=0.5)
        ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,edgecolors="black",alpha=0.6)
        ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,edgecolors="black",alpha=0.6)
        if ds_ct==0:
            ax.set_title(name)
        ax.text(
            X[:,0].max()+0.25,
            X[:,1].min()+0.45 ,str(score))
        i+=1
    
    