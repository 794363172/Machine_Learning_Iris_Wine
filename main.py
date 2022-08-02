import numpy as np
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X_train, X_test,y_train,y_test = train_test_split(wine.data,wine.target,
                                                 test_size=0.3,random_state=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print('测试数据得分: {:.2f}'.format(knn.score(X_test, y_test)))