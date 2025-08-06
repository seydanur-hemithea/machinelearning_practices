# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:18:01 2025

@author: asus
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
digit=load_digits()
fig,axes=plt.subplots(nrows=2,ncols=5,figsize=(10,5),
                      subplot_kw={"xticks":[],"yticks":[]})
for i ,ax in enumerate(axes.flat):
    ax.imshow(digit.images[i],cmap="binary",interpolation="nearest")
    ax.set_title(digit.target[i])
plt.show()
X=digit.data
y=digit.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
svc=SVC(kernel="linear",random_state=42)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

print( classification_report(y_test,y_pred))