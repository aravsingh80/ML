from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from time import time
t = time()
X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y)
k = 75
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(x_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = x_train[representative_digit_idx]
y_init = []
for y in representative_digit_idx: y_init.append(y_train[y])
y_representative_digits = np.array(y_init)
y_train_propagated = np.empty(len(x_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
    percentile_closest = 25
X_cluster_dist = X_digits_dist[np.arange(len(x_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1
    partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = x_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
print(time()-t)
log_reg = LogisticRegression()
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(x_test, y_test))