# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 17:19:42 2025

@author: asus
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import numpy as np

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
dtc=DecisionTreeClassifier()
tree_parametres_grid={"max_depth":[3,5,7],
                      "max_leaf_nodes":[None,5,10,20,30,50]}
nb_cv=3
tree_grid_search=GridSearchCV(dtc, tree_parametres_grid)
tree_grid_search.fit(X_train,y_train)


print("tree best parametress:",tree_grid_search.best_params_)
print("tree best Accuracy:",tree_grid_search.best_score_)
for mean_score,params in zip(tree_grid_search.cv_results_["mean_test_score"],
                             tree_grid_search.cv_results_["params"]):
    print(f"ortalama test scoru:{mean_score},parametreler:{params}")
    
cv_result=tree_grid_search.cv_results_
for i ,params in enumerate((cv_result["params"])):
    print(f"parametrelerr:{params}")
    for j in range(nb_cv):
        accuracy=cv_result[f"split{j}_test_score"][i]
        print(f"\tfold{j+i}-accuracy:{accuracy}")
        
    
    
    