import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

train = pd.read_csv(r'F:\gongyezhengqi\zhengqi_train.txt', sep='\t')
test = pd.read_csv(r'F:\gongyezhengqi\zhengqi_test.txt', sep='\t')
# for index in range(38, 77):
#     index = 'V' + str(index)
#     train[index] = 0
# train.loc[0, 'V38'] = 2
# print(train.loc[0, 'V38'])
# pop_list = [5, 11, 17, 20, 22, 27, 28]  # 表现最好
pop_list = [5, 6, 9, 11, 13, 14, 17, 19, 20, 21, 22, 23, 28, 35]
# pop_list = [5, 11, 17, 20, 22, 27, 28, 8, 13, 16, 21, 31, 32]
# pop_list = []
# 测试集V21有异常数据
for index in pop_list:
    train.pop('V' + str(index))
    test.pop('V' + str(index))
post = 2
index_list = [index for index in range(0, 38) if index not in pop_list]
for num in range(len(train) - post):
    for index in index_list:
        train.loc[num + 1, 'V' + str(index + 38)] = train.loc[num, 'V' + str(index)]
        train.loc[num + 2, 'V' + str(index + 76)] = train.loc[num, 'V' + str(index)]
        # train.loc[num + 3, 'V' + str(index + 76)] = train.loc[num, 'V' + str(index - 38)]
train = train.iloc[range(post, len(train) - post + 1)]
# print(test)
# print(test.shape)
for num in range(len(test) - post):
    for index in index_list:
        test.loc[num + 1, 'V' + str(index + 38)] = test.loc[num, 'V' + str(index)]
        test.loc[num + 2, 'V' + str(index + 76)] = test.loc[num, 'V' + str(index)]
        # test.loc[num + 3, 'V' + str(index + 76)] = test.loc[num, 'V' + str(index - 38)]
test = test.iloc[range(post, len(test) - post + 1)]
# print(test)
# print(test.shape)

# print(train.loc[[0]])
# print(train.loc[[1]])
# print(train.loc[[2]])
# print(train)
# print(train.shape)

train_Y = train['target']
train.pop('target')
train_X = train

X_train, X_test, Y_train, Y_test = \
    train_test_split(train_X, train_Y, test_size=0, random_state=40, shuffle=True)
# X_train = train_X.iloc[range(500, 2500)]
# X_test = train_X.iloc[range(0, 500)]
# Y_train = train_Y.iloc[range(500, 2500)]
# Y_test = train_Y.iloc[range(0, 500)]

# X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0, random_state=20, shuffle=True)

# model = LinearRegression(copy_X=True)
model = Ridge(alpha=100)
# rfe = RFE(estimator=model, n_features_to_select=23) 000 盼
# rfe = RFE(estimator=model, n_features_to_select=45) 001 luyi
# rfe = RFE(estimator=model, n_features_to_select=12)
rfe = RFE(estimator=model, n_features_to_select=50)
rfe.fit(X_train, Y_train)
rank = list(rfe.ranking_)
print(rank)
for index in range(0, len(rank), 38 - len(pop_list)):
    for a in range(38 - len(pop_list)):
        print(rank[index + a], end=', ')
    print('\n')
result = rfe.predict(test)
# print(result.shape)
# print(Y_test - result)
# print(rfe.n_features_, mean_squared_error(Y_test, rfe.predict(X_test)))


# print(train)
# print(train.shape)

# print(result)
# print(result[2])
# print(type(result))
# print(result.shape)

result[1277] = 0.1770498520296
result[1278] = -0.3204281379733
result[1279] = -0.0542595419233
result[1917] = -0.5763419316004
result[1918] = -2.1853724597220
result[1919] = -3.4511810864974
result[1920] = -3.3722799804122
result[1921] = -3.3003353156363
print(result)
print(result.shape)


def record(result):
    id = "梦中的木头人"
    description = "添加前两行作为特征扩充 pop_list = [5, 6, 9, 11, 13, 14, 17, 19, 20, 21, 22, 23, 28, 35]"
    preprocessing = "train_test_split(train_X, train_Y, test_size=0, random_state=40, shuffle=True)"
    shuffle = "True"
    algorithm = "ridge 特征扩充 特征选择"
    parameter = "Ridge(alpha=100) rfe = RFE(estimator=model, n_features_to_select=50)"
    testscore = "无"
    filepath = r"F:\gongyezhengqi\gongyezhengqi_2019-3-13_002.txt"

    log = "\n\nTime: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + \
          "\nFilePath: " + filepath + \
          "\nID: " + id + \
          "\nDescription: " + description + \
          "\nPreprocessing: " + preprocessing + \
          "\nShuffle: " + shuffle + \
          "\nAlgorithm: " + algorithm + \
          "\nParameter: " + parameter + \
          "\nTestScore: " + testscore + \
          "\nOnLineScore: "
    np.savetxt(filepath, result, fmt='%0.13f')
    with open(r"F:\gongyezhengqi\提交日志.txt", "a+") as fp:
        fp.write(log)


record(result)
