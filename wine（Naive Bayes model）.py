"""
wine（Naive Bayes model）~

Author:杜博文
Date：2022/8/2
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine
X_train, X_test, y_train, y_test = train_test_split(wine.date,wine.target, random_state=0)

gnb = GaussianNB()
gnb.fit(wine.date,wine.target)
GaussianNB()


















