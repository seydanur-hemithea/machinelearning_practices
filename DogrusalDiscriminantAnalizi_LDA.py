# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 17:05:01 2025

@author: asus
"""

from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
mnist=fetch_openml("mnist_784", version=1)
X=mnist.data
y=mnist.target.astype(int)
lda=LinearDiscriminantAnalysis(n_components=2)
X_lda=lda.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap="tab10",alpha=0.6)
plt.title("LDA of MNIST dataset")
plt.xlabel("ld1")
plt.ylabel("ld2")
plt.colorbar(lable="Digits")
