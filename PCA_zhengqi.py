import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X_train = np.loadtxt('D:/zhengqi_train.txt', skiprows=1)
# X_train = np.mat(X_train)
X_train = np.delete(X_train, 38, axis=1)
print(X_train)
pca = PCA(n_components=3)
new_X = pca.fit_transform(X_train)
print(new_X)
# print(pca.explained_variance_ratio_)
Y = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=38)
print(Y)
# 3D可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# print(new_X[0, 1])
color = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'w']   # 蓝，青绿，绿，黑，紫红，红，黄，白
print(len(Y))
step = (Y.max()-Y.min())/7
for i in range(len(Y)):
    index = int((Y[i]-Y.min())/step)      # Y[i]是最大值的时候这一点用白色表示了
    ax.scatter(new_X[i, 0], new_X[i, 1], new_X[i, 2], s=1, c=color[index])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

