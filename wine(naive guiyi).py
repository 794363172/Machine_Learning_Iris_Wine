import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=0,test_size=0.3)
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
clf = GaussianNB()
clf.fit(X_train, y_train)
GaussianNB()
print('测试数据得分: {:.2f}'.format(clf.score(X_test, y_test)))

X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
prediction = clf.predict(X_new)
print('预测新红酒的分类为: {}'.format(wine['target_names'][prediction]))