# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:18:14 2025

@author: asus
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Örnek veri seti (iris)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Korelasyon matrisi
corr_matrix = df.corr()

# Isı haritası ile görselleştirme
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi (Iris Verisi)')
plt.show()