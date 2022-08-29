# knn 标准化
import numpy as np
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
                                            test_size=0.3,random_state=1)
from sklearn.preprocessing import Normalizer, StandardScaler

scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

"""
在需要设置random_state的地方给其赋一个值，当多次运行此段代码能够得到完全一样的结果，
别人运行此代码也可以复现你的过程。若不设置此参数则会随机选择一个种子，执行结果也会因此而不同了。
虽然可以对random_state进行调参，但是调参后在训练集上表现好的模型未必在陌生训练集上表现好，
所以一般会随便选取一个random_state的值作为参数。
X_train, X_test, y_train, y_test  记住这个顺序！！！！！！
"""
knn = KNeighborsClassifier(n_neighbors=1,weights='uniform')
knn.fit(X_train, y_train)
print('测试数据得分: {:.2f}'.format(knn.score(X_test, y_test)))

X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
prediction = knn.predict(X_new)
print('预测新红酒的分类为: {}'.format(wine['target_names'][prediction]))

#测试数据为0.74

parameters = {'n_neighbors': [1, 10],'weights':['uniform','distance']}
gs = GridSearchCV(estimator=knn,param_grid=parameters, refit = True, cv = 5, verbose = 1, n_jobs = -1)
gs.fit(X_train,y_train)
print('最优参数: ',gs.best_params_)
print('最佳性能: ', gs.best_score_)
print('最佳模型: ',gs.best_estimator_)
# 测试数据得分: 0.98
# 预测新红酒的分类为: ['class_0']
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# 最优参数:  {'n_neighbors': 10, 'weights': 'distance'}
# 最佳性能:  0.952
# 最佳模型:  KNeighborsClassifier(n_neighbors=10, weights='distance')

from sklearn.datasets import load_wine
wine_dataset = load_wine()
print(wine_dataset.data.mean(0))
print(wine_dataset.data.std(0))

