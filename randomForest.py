# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:24:45 2025

@author: asus
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

oli=fetch_olivetti_faces()
plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i+50],cmap="gray")
plt.show()    
X=oli.data 
y=oli.target 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rf_clf=RandomForestClassifier(n_estimators=100,random_state=42,)

rf_clf.fit(X_train,y_train)
y_pred=rf_clf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)
#%%

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

calif=fetch_california_housing()
X=calif.data
y=calif.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rf_calif=RandomForestRegressor(n_estimators=100,random_state=42)
rf_calif.fit(X_train,y_train)
y_pred=rf_calif.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
print(mse)
rmse=np.sqrt(mse)
print(rmse)





