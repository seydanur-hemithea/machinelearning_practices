# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:06:50 2025

@author: asus
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("örnek veri")

kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
labels=kmeans.labels_
plt.figure()
plt.scatter(X[:,0],X[:,1],c=labels,cmap="viridis")
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c="red",marker="X")