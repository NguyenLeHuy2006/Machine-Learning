import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 


np.random.seed(7) 
iris = datasets.load_iris()
iris_X = iris.data   
iris_y = iris.target  

print('Labels:', np.unique(iris_y))
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)
print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])
print('-' * 40)



model_1nn = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
model_1nn.fit(X_train, y_train)
y_pred_1nn = model_1nn.predict(X_test)
print("Accuracy of 1NN: %.2f %%" % (100 * accuracy_score(y_test, y_pred_1nn)))



model_7nn = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2)
model_7nn.fit(X_train, y_train)
y_pred_7nn = model_7nn.predict(X_test)
print("Accuracy of 7NN with major voting: %.2f %%" % (100 * accuracy_score(y_test, y_pred_7nn)))




model_7nn_dist = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance')
model_7nn_dist.fit(X_train, y_train)
y_pred_7nn_dist = model_7nn_dist.predict(X_test)
print("Accuracy of 7NN (1/distance weights): %.2f %%" % (100 * accuracy_score(y_test, y_pred_7nn_dist)))


def myweight(distances):
    sigma2 = .4 
    return np.exp(-distances**2 / sigma2)

model_7nn_custom = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = myweight)
model_7nn_custom.fit(X_train, y_train)
y_pred_7nn_custom = model_7nn_custom.predict(X_test)
print("Accuracy of 7NN (customized weights): %.2f %%" % (100 * accuracy_score(y_test, y_pred_7nn_custom)))