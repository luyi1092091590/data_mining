import numpy as np
X_train = np.loadtxt('D:/zhengqi_train.txt', skiprows=1)
X_train = np.delete(X_train, 38, axis=1)
Y = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=38)
Y_list = list(Y)
temp_list = [Y_list.index(np.max(Y_list))]
step = (Y.max()-Y.min())/7
for i in range(len(Y)):
    if (Y[i] - Y.min()) / step > 6:
        temp_list.append(i)
X_train = np.mat(X_train)
Y = np.mat(Y)
Y = Y.T
for index in temp_list:
    X_train = np.delete(X_train, index, axis=0)
    Y = np.delete(Y, index, axis=0)
X_train = np.insert(X_train, 0, [1] * 2838, axis=1)
print(X_train.shape)
print(Y.shape)
A = np.eye(39)
temp = [2, 1, 3, 6, 5, 17, 21, 19, 4, 12, 9, 31, 8, 34, 13, 36, 16, \
             18, 33, 35, 14, 26, 38, 23, 15, 25, 37, 11, 30, 27, 20, 10, 32, 28, 29, 22, 24, 7]
temp = [x**(1/1.02) for x in temp]
for index in range(1, 39):
    A[index][index] = temp[index - 1]

A[0][0] = 0
print(A.shape)
Lambada = 100
feature_value = (X_train.T * X_train + Lambada * A).I * X_train.T * Y
print(feature_value)
print(feature_value.shape)
test_X = np.loadtxt('D:/zhengqi_test.txt', skiprows=1)
test_X = np.insert(test_X, 0, [1] * 1925, axis=1)
print(test_X)
print(test_X.shape)
result = test_X * feature_value
print(result)
# np.savetxt('D:\蒸汽结果\权重.txt', result, fmt='%.10f')
