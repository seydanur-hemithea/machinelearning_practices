# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:54:46 2025

@author: asus
"""

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


iris=load_iris()
X=iris.data
y=iris.target
target_names=iris.target_names
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)

lda=LinearDiscriminantAnalysis(n_components=2)
X_lda=lda.fit_transform(X,y)
colors=["red","green","blue"]
plt.figure()

for color,i,target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_pca[y==i,0],X_pca[y==i,1],color=color,alpha=0.8,label=target_name)
plt.legend()
plt.title("pca of Iris dataset")
plt.figure()
for color,i,target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_lda[y==i,0],X_lda[y==i,1],color=color,alpha=0.8,label=target_name)
plt.legend()
plt.title("lda of Iris dataset")
