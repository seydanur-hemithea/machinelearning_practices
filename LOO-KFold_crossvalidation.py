# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:49:29 2025

@author: asus
"""


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

import numpy as np

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
tree=DecisionTreeClassifier()
tree_parametres_grid={"max_depth":[3,5,7]
                     }

#KFold Grid Search
kf=KFold(n_splits=10)
tree_grid_search_kf=GridSearchCV(tree, tree_parametres_grid,cv=kf)
tree_grid_search_kf.fit(X_train,y_train)
print("En iyi kf parametre:",tree_grid_search_kf.best_params_)
print("En iyi kf score:",tree_grid_search_kf.best_score_)

#leave one out
loo=LeaveOneOut()
tree_grid_search_loo=GridSearchCV(tree, tree_parametres_grid,cv=loo)
tree_grid_search_loo.fit(X_train,y_train)
print("En iyi loo parametre:",tree_grid_search_loo.best_params_)
print("En iyi loo score:",tree_grid_search_loo.best_score_)