from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
neigh = KNeighborsClassifier()
from time import time
X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y)
# print(len(x_train))
# print(len(x_test))
#t = time()
neigh.fit(x_train, y_train)
#print(time()-t)
count = 0
t2 = time()
for i in range(len(x_test)): 
    if y_test[i] == neigh.predict([x_test[i]])[0]: count += 1
print(time()-t2)
print(count/len(x_test))