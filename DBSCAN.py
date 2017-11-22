from sklearn.datasets import make_moons 
import matplotlib.pyplot as plt 
import numpy as np 
from numpy.linalg import cholesky
import random 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN



x1,y1=make_moons(n_samples=1000,noise=0.1) 
plt.figure(1)


# plt.scatter(x1[:,0],x1[:,1],marker='o',c=y1) 
# plt.show()


# y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(x1)
# plt.scatter(x1[:, 0], x1[:, 1], c=y_pred)
# plt.show()


y_pred = DBSCAN(eps=0.1).fit_predict(x1)
plt.scatter(x1[:, 0], x1[:, 1], c=y_pred)
plt.show()