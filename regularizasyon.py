# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:40:25 2025

@author: asus
"""

from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# Veri seti oluştur (10 özellikten sadece 3'ü anlamlı)
X, y, coef = make_regression(n_samples=100, n_features=10, noise=10, coef=True)

# Ridge ve Lasso modelleri
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)

# Modelleri eğit
ridge.fit(X, y)
lasso.fit(X, y)

# Ağırlıkları karşılaştır
plt.figure(figsize=(10,5))
plt.plot(ridge.coef_, label='Ridge Coefficients', marker='o')
plt.plot(lasso.coef_, label='Lasso Coefficients', marker='x')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Ridge vs Lasso Ağırlık Karşılaştırması')
plt.xlabel('Özellik Index')
plt.ylabel('Ağırlık (Coefficient)')
plt.legend()
plt.grid(True)
plt.show()