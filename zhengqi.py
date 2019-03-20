import numpy as np

# 2888个样本，38个特征
num_list = []
temp = 0
for line in open('D:/zhengqi_train.txt').readlines():
    if temp != 0:
        num = list(map(float, line.split()))
        num_list.append(num)
    temp += 1
X = np.mat(num_list)
X = np.insert(X, 0, [1]*2888, axis=1)   # 在X矩阵的第0列插入一列1
Y = X[:, 39]    # 真实值y,2888*1的矩阵
X = np.delete(X, 39, axis=1)    # 删除最后一列
feature_value = (X.T * X).I * X.T * Y   # 正规方程
print(X)
print(X.shape)
print(Y)
print(Y.shape)
print(feature_value)
print(feature_value.shape)

num_list1 = []
temp = 0
for line in open('D:/zhengqi_test.txt').readlines():
    if temp != 0:
        num = list(map(float, line.split()))
        num_list1.append(num)
    temp += 1
test_X = np.mat(num_list1)
print(test_X)
print(test_X.shape)
test_X = np.insert(test_X, 0, [1]*1925, axis=1)
print(test_X)
print(test_X.shape)
result = X * feature_value
print(result)
# np.savetxt('D:/aaa.txt', result, fmt='%.10f')  # 存到txt文件，小数点后保留10位

'''
a = np.loadtxt('test1.txt', skiprows=1, dtype=int)
这里的skiprows是指跳过前1行, 如果设置skiprows=2, 就会跳过前两行,

numpy中数组和矩阵的区别：
matrix是array的分支，matrix和array在很多时候都是通用的，你用哪一个都一样。但这时候，官方建议大家如果两个可以通用，那就选择array，因为array更灵活，速度更快，很多人把二维的array也翻译成矩阵。
但是matrix的优势就是相对简单的运算符号，比如两个矩阵相乘，就是用符号*，但是array相乘不能这么用，得用方法.dot()
array的优势就是不仅仅表示二维，还能表示3、4、5...维，而且在大部分Python程序里，array也是更常用的。
'''

