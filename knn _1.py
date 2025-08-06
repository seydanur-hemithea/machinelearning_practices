#dataset examination
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

cancer=load_breast_cancer()
df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target

X=cancer.data#features
y=cancer.target#target

#train_test split
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.3,random_state=30) 

#olceklendirme
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=4)#model creating 

knn.fit(X_train,y_train)#model training
y_pred=knn.predict(X_test)
acc=accuracy_score(y_test,y_pred)
print(acc)
cm=confusion_matrix(y_test,y_pred)
print(cm)
#hiperpAREMTER SETTİNG
acc_values=[]
k_values=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)#model creating 
    knn.fit(X_train,y_train)#model training
    y_pred=knn.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    acc_values.append(acc)
    k_values.append(k)
    
plt.figure() 
plt.plot(k_values,acc_values,marker="o",linestyle="--")
plt.title("k degree vs acc")
plt.xlabel("k degree")
plt.ylabel("acc")
plt.xticks(k_values)
plt.grid(True)

#%%
import numpy as np










for i ,weight in enumerate(["uniform","distance"]):
    knn=KNeighborsRegressor(n_neighbors=5,weights=weight)
    y_pred=knn.fit(X,y).predict(T)
    plt.subplot(2,1,i+1)
    plt.scatter(X,y,color="green",label="data")
    plt.plot(T,y_pred,color="blue",label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN regressor weights={}".format(weight))
    plt.tight_layout()
    plt.show()










