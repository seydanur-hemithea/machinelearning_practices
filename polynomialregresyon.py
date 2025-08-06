# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:56:12 2025

@author: asus
"""

import numpy as np
import  matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X=4*np.random.rand(100,1)
y=2+3*X**2
poly_fet=PolynomialFeatures(degree=2)
X_poly=poly_fet.fit_transform(X)

poly_reg=LinearRegression()
poly_reg.fit(X_poly,y)
plt.scatter(X,y,color="blue")
X_test=np.linspace(0,4,100).reshape(-1, 1)
X_test_poly=poly_fet.transform(X_test)
y_pred=poly_reg.predict(X_test_poly)


plt.plot(X_test,y_pred,color="red")
plt.xlabel("X")
plt.ylabel("y")

plt.title("ploinomial regresyon model")
