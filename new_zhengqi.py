import numpy as np

X = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=(0, 1, 2, 3, 4, 6, 7, 8, 10, 12,
                                                            13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30,
                                                            31, 32, 33, 34, 35, 36, 37))  # skiprows=1跳过第一行
X = np.insert(X, 0, [1] * 2888, axis=1)
X = np.mat(X)
print(X)
print(X.shape)
Y = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=38)
Y = np.mat(Y)
Y = Y.T
print(Y)
print(Y.shape)
A = np.eye(33)

list_temp = [2, 1, 3, 6, 5, 18, 16, 4, 9, 8, 29, 12, 31, 15, \
             28, 30, 13, 23, 20, 14, 22, 32, 11, 24, 17, 10, 27, 25, 26, 19, 21, 7]
list_temp = [x for x in list_temp]
for index in range(1, 33):
    A[index][index] = list_temp[index - 1]

A[0][0] = 0

print(A)
Lambada = 200
feature_value = (X.T * X + Lambada * A).I * X.T * Y

print(feature_value)
print(feature_value.shape)
test_X = np.loadtxt('D:/zhengqi_test.txt', skiprows=1, usecols=(
    0, 1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36,
    37))
test_X = np.insert(test_X, 0, [1] * 1925, axis=1)
print(test_X)
print(test_X.shape)
result = test_X * feature_value
print(result)

# np.savetxt('D:\蒸汽结果\哦.txt', result, fmt='%.10f')
