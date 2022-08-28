from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
wine=load_wine()
x_train,x_test,y_train,y_test = train_test_split(wine.data, wine.target,test_size=0.3,random_state=1)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(x_train)
X_test = scalar.fit_transform(x_test)
knn = KNeighborsClassifier(n_neighbors=1,weights='uniform')
knn.fit(x_train, y_train)
print('测试数据得分: {:.2f}'.format(knn.score(x_test, y_test)))