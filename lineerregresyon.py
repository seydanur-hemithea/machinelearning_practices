# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:13:19 2025

@author: asus
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error,r2_score
diabets=load_diabetes()
diabets_X,diabets_y=load_diabetes(return_X_y=True)
diabets_X=diabets_X[:,np.newaxis,2]

diabets_X_train=diabets_X[:-20]
diabets_X_test=diabets_X[-20:]

diabets_y_train=diabets_y[:-20]
diabets_y_test=diabets_y[-20:]

lin_reg=LinearRegression()
lin_reg.fit(diabets_X_train,diabets_y_train)

diabets_y_pred=lin_reg.predict(diabets_X_test)
mse=mean_squared_error(diabets_y_test,diabets_y_pred)
print("mse:",mse)
r2=r2_score(diabets_y_test,diabets_y_pred)
print("r2:",r2)
