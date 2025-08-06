# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:36:34 2025

@author: asus
"""

#logistic reg ile sınıflandırma yapabiliyoruz adı regresyon olmasına ragmen regresyıon yapılmkıyor
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
df=pd.DataFrame(data=heart_disease.data.features)
# data (as pandas dataframes) 
df["target"]=  heart_disease.data.targets

if df.isna().any().any():
    df.dropna(inplace=True)
    print("nan")
    
X=df.drop(["target"],axis=1).values

y=df.target.values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
log_reg=LogisticRegression(penalty="l2",C=1,solver="lbfgs",max_iter=100)
log_reg.fit(X_train,y_train)
acc=log_reg.score(X_test,y_test)

print(acc)