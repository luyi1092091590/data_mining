from sklearn import linear_model
import numpy as np
X = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=(0, 1, 2, 4, 8, 12, 27, 31, 37))
# X = np.insert(X, 0, [1]*2888, axis=1)
Y = np.loadtxt('D:/zhengqi_train.txt', skiprows=1, usecols=38)
Y = np.mat(Y)
Y = Y.T
clf = linear_model.LinearRegression()   # fit_intercep 默认True，表示有截距
clf.fit(X, Y)
feature_value = clf.coef_   # 回归系数
feature_value = np.mat(feature_value)
print(feature_value)
print(feature_value.shape)
print(type(feature_value))
test_X = np.loadtxt('D:/zhengqi_test.txt', skiprows=1, usecols=(0, 1, 2, 4, 8, 12, 27, 31, 37))
# test_X = np.insert(test_X, 0, [1]*1925, axis=1)
'''
print(test_X)
print(test_X.shape)
feature_value = feature_value.T
result = test_X * feature_value
'''
result = clf.predict(test_X)
print(result)
# print(clf.score(test_X, result))      # 越大越好，小于1的得分，测试集和真实值比较
print('哦', clf.intercept_)   # 截距
# np.savetxt('D:/ccc.txt', result, fmt='%.10f')
