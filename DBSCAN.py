# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:46:29 2025

@author: asus
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("örnek veri")
linkage_methods=("ward","single","average","complete")
"""ward=küme içi varyansları minimize ediyoruz
single=iki kğüme içerisindeki en yakın iki nokta arasındasındaki messafe
avarage=iki küme arasındaki tüm noklar arasındaki mesafe
complete=iki küme arasındaki en uzak iki nokta arasındaki mesafe ölçülğyoır"""
plt.figure()
for i ,linkage_method in enumerate(linkage_methods,1):
    model=AgglomerativeClustering(n_clusters=4,linkage=linkage_method)
    cluster_labels=model.fit_predict(X)
    plt.subplot(2,4,i)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    dendrogram(linkage(X,method=linkage_method),no_labels=True)
    plt.xlabel("veri noktaları")
    plt.ylabel("uazaklılk")
    plt.subplot(2,4,i+4)
    plt.scatter(X[:,0],X[:,1],c=cluster_labels,cmap="viridis")
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    plt.xlabel("X")
    plt.ylabel("y")
    
    
    
    
    
    
    
    