import numpy as np

X = np.loadtxt('D:/zhengqi_train.txt', skiprows=1)
X = np.insert(X, 0, [1] * 2888, axis=1)
X = np.mat(X)
X = np.delete(X, 39, axis=1)  # 删除第40列
print(X)
print(X.shape)
Y = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=38)
Y = np.mat(Y)
Y = Y.T
print(Y)
print(Y.shape)
A = np.eye(39)

list_temp = [2, 1, 3, 6, 5, 17, 21, 19, 4, 12, 9, 31, 8, 34, 13, 36, 16, \
             18, 33, 35, 14, 26, 38, 23, 15, 25, 37, 11, 30, 27, 20, 10, 32, 28, 29, 22, 24, 7]
list_temp = [x**(1/1.02) for x in list_temp]
for index in range(1, 39):
    A[index][index] = list_temp[index - 1]

A[0][0] = 0

print(A)
Lambada = 100
feature_value = (X.T * X + Lambada * A).I * X.T * Y
print(feature_value)
print(feature_value.shape)
test_X = np.loadtxt('D:/zhengqi_test.txt', skiprows=1)
test_X = np.insert(test_X, 0, [1] * 1925, axis=1)
print(test_X)
print(test_X.shape)
result = test_X * feature_value
print(result)
# np.savetxt('D:\蒸汽结果\权重.txt', result, fmt='%.10f')

'''
因为训练集和测试集的分布不一样，剔除5,9,11,17,22,28
'''
