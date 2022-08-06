import numpy as np
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
                                            test_size=0.3,random_state=1)
"""
在需要设置random_state的地方给其赋一个值，当多次运行此段代码能够得到完全一样的结果，
别人运行此代码也可以复现你的过程。若不设置此参数则会随机选择一个种子，执行结果也会因此而不同了。
虽然可以对random_state进行调参，但是调参后在训练集上表现好的模型未必在陌生训练集上表现好，
所以一般会随便选取一个random_state的值作为参数。
X_train, X_test, y_train, y_test  记住这个顺序！！！！！！
"""
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print('测试数据得分: {:.2f}'.format(knn.score(X_test, y_test)))

X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
prediction = knn.predict(X_new)
print('预测新红酒的分类为: {}'.format(wine['target_names'][prediction]))

#测试数据为0.74

