import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score  # 划分训练集和测试集
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import neighbors

iris = load_iris()
print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
print(iris.data[0])  # [ 5.1  3.5  1.4  0.2]
print(iris.target[0])  # 0
for i in range(len(iris.target)):
 print("example %d: label %s, feature %s" % (i,iris.target[i],iris.data[i]))
 """
 导入数据集的方法
 ① datasets.load_dataset_name（）：sklearn包自带的小数据集
 ②datasets.fetch_dataset_name（）：比较大的数据集，主要用于测试解决实际问题，支持在线下载
 ③datasets.make_dataset_name（）：构造数据集
 """
 iris_X = iris.data#X为数据
 iris_Y = iris.target#Y为特征
 X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=0.3,random_state=1)#将数据集分为训练集和测试集

 knn = neighbors.KNeighborsClassifier(n_neighbors=5)
 knn.fit(X_train, y_train)



print(cross_val_score(knn, X_train, y_train, cv=4))

y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)

#预测得分
score=knn.score(X_test, y_test)
print('预测得分:',score)
#预测得分: 0.95555
X_new = np.array([[6.2, 1.05, 3.33, 2.0]])
prediction = knn.predict(X_new)
print('预测鸢尾花的分类为: {}'.format(iris['target_names'][prediction]))
#预测新红酒的分类为: ['versicolor']