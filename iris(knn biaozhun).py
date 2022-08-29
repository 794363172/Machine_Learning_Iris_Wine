import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score  # 划分训练集和测试集
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import neighbors
from sklearn.preprocessing import Normalizer, StandardScaler

iris = load_iris()
iris_X = iris.data  # X为数据
iris_Y = iris.target  # Y为特征
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=0.3,random_state=1)
# 先标准化再预测
scaler = StandardScaler()  #变量scaler接收标准化方法
# # 传入特征值进行标准化
X_train = scaler.fit_transform(X_train)  #对训练的特征值标准化
X_test = scaler.fit_transform(X_test)    #对测试的特征值标准化
knn = KNeighborsClassifier(n_neighbors=1,weights='uniform')
knn.fit(X_train, y_train)
score=knn.score(X_test, y_test)
print('预测得分:',score)

