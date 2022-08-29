"""
iris(svm)~

Author:dbw
Date：2022/8/12
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#导入数据，分训练集和测试集
iris = load_iris()
iris_X = iris.data#X为数据
iris_Y = iris.target#Y为特征
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y,
                                                    test_size=0.3,random_state=1)
#归一化
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
#网格搜索
svc = svm.SVC()
params = [{'kernel':['linear'], 'C':[1, 10, 100, 1000]},
    {'kernel':['poly'], 'C':[1], 'degree':[2, 3]},
    {'kernel':['rbf'], 'C':[1,10,100], 'gamma':[1, 0.1, 0.01]}]
gs = GridSearchCV(estimator=svc,param_grid=params, cv=2)
gs.fit(X_train,y_train)
print('最优参数: ',gs.best_params_)
print('最佳性能: ', gs.best_score_)
print('最佳模型: ',gs.best_estimator_)





clf = svm.SVC(kernel='rbf',gamma=11,                      # 核函数
             decision_function_shape='ovo',      # one vs one 分类问题
             C=10)
clf.fit(X_train,y_train)
print('测试数据得分: {:.2f}'.format(clf.score(X_test, y_test)))

score=clf.score(X_test, y_test)
print('测试数据得分: ',score)
print("Train_score:{0}\nTest_score:{1}".format(clf.score(X_train, y_train),
                                               clf.score(X_test, y_test)))
# 测试数据得分:  0.9777777777777777
# from sklearn.metrics import accuracy_score
# y_pred = clf.predict(X_test)
# accuracy_score(y_test, y_pred)
# score=accuracy_score(y_test, y_pred)
# print('测试数据得分: ',score)

