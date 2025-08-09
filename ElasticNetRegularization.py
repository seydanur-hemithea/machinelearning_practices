# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:49:29 2025

@author: asus
"""


from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.metrics import mean_squared_error


diabetes=load_diabetes()
X=diabetes.data
y=diabetes.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
elasticNet=ElasticNet()
EN_param_grid={"alpha":[0.1,1,10,100],"l1_ratio":[0.1,0.3,0.5,0.7,0.9]}
#1 e yaklastikca lasso 0 a yaklastikca ridge e saha egimli oluyor(L1 ratio)

EN_grid_search=GridSearchCV(elasticNet, EN_param_grid,cv=5)
EN_grid_search.fit(X_train,y_train)
print("En iyi en parametre:",EN_grid_search.best_params_)
print("En iyi en score:",EN_grid_search.best_score_)

best_en_model=EN_grid_search.best_estimator_
y_pred_en=best_en_model.predict(X_test)
en_mse=mean_squared_error(y_test,y_pred_en)
print("en_mse=",en_mse)

