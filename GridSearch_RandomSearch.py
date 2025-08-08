# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:20:50 2025

@author: asus
"""


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
KNN=KNeighborsClassifier()


knn_parametres_grid={"n_neighbors":np.arange(2,31)}

knn_grid_search=GridSearchCV(KNN, knn_parametres_grid)
knn_grid_search.fit(X_train,y_train)


print("KNN best parametress:",knn_grid_search.best_params_)
print("KNN best Accuracy:",knn_grid_search.best_score_)


knn_random_search=RandomizedSearchCV(KNN, knn_parametres_grid)
knn_random_search.fit(X_train,y_train)


print("KNN best parametress:",knn_random_search.best_params_)
print("KNN best Accuracy:",knn_random_search.best_score_)

