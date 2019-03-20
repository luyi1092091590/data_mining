import numpy as np
import math
import random
import time

e = math.exp(1)

train_file_path = 'D:/SMSSpamCollection.txt'
test_file_path = 'D:/test_set_new.txt'


def probability_i(w, x):
    probability = (1 + (e ** int(-(w.T * x)))) ** (-1)
    # print('probability=', probability)
    return probability


def partial_derivative_of_negative_log_likelihood_of_w(start, end, w, X, Y):
    temp_list = []
    for item in range(len(X[0])):
        temp_list.append([0])
    partial_derivative_sum = np.mat(temp_list)
    for index in range(start, end):
        # print('X[index]=', X[index])
        # print('w=', w)
        temp_x_t = X[index].T
        p = probability_i(w, temp_x_t)

        # partial_derivative = (float(Y[index]) - p) * temp_x_t
        # print('partial_derivative=', partial_derivative)

        # partial_derivative_sum = partial_derivative_sum + partial_derivative
        partial_derivative_sum = partial_derivative_sum + 0.01 * (float(Y[index]) - p) * temp_x_t
        # print('partial_derivative_sum=', partial_derivative_sum)

        # print()
        # print()
    # partial_derivative_sum = partial_derivative_sum * (-1)
    return partial_derivative_sum


start = time.time()
# 数据读取
all_word_list = []
list_Y = []
list_X = []

for line in open(train_file_path, encoding='ISO-8859-1').readlines():
    temp_list = line.split()
    all_word_list.extend(temp_list[1:])
all_word_list = list(set(all_word_list))
all_word_list.sort()


for line in open(train_file_path, encoding='ISO-8859-1').readlines():
    temp_list = line.split()
    # 生成单词表征
    word_vec = [1]
    for word in all_word_list:
        if word in temp_list[1:]:
            word_vec.append(1)
        else:
            word_vec.append(0)
    list_X.append(word_vec)

    # 生成单词标签
    if temp_list[0] == 'spam':
        list_Y.append([1])
    else:
        list_Y.append([0])

# 初始化w
list_w = []
for item in range(len(list_X[0])):
    list_w.append(0)

# 矩阵化
mat_X = np.mat(list_X)
mat_Y = np.mat(list_Y)
mat_w = np.mat(list_w)
mat_w = mat_w.T

# 迭代过程
'''
num = 10000
span = 1
for item in range(num):
    index = random.randint(0, len(list_Y) - span - 1)
    mat_w = mat_w + partial_derivative_of_negative_log_likelihood_of_w(index, index + span, mat_w, mat_X, mat_Y)
'''
# 生成测试数据
test_list_X = []
test_list_Y = []
for line in open(test_file_path, encoding='ISO-8859-1').readlines():
    temp_list = line.split()
    # print('列表:', temp_list)
    # 生成单词表征
    word_vec = [1]
    for word in all_word_list:
        if word in temp_list[1:]:
            word_vec.append(1)
        else:
            word_vec.append(0)
    test_list_X.append(word_vec)
    # 生成单词标签
    if temp_list[0] == 'spam':
        test_list_Y.append([1])
    else:
        test_list_Y.append([0])

test_mat_X = np.mat(test_list_X)
test_mat_Y = np.mat(test_list_Y)

'''
for item in range(100):
    index = random.randint(0, len(test_list_Y) - 1)
    print('垃圾率：', probability_i(mat_w, test_mat_X[index].T))
    print('标签：', test_list_Y[index][0])
    print()
'''

'''
true_num = 0
false_num = 0
i = 0.5

for index in range(len(test_list_Y)):
    if probability_i(mat_w, test_mat_X[index].T) >= 0.5:
        if int(test_list_Y[index][0]) == 1:
            true_num += 1
        else:
            false_num += 1
    else:
        if int(test_list_Y[index][0]) == 0:
            true_num += 1
        else:
            false_num += 1
print('true:', true_num)
print('false:', false_num)
print('正确率为：', true_num / (true_num + false_num))
'''
test_num = 100
sum = 0
for i in range(test_num):
    mat_w = np.mat(list_w)
    mat_w = mat_w.T
    num = 1000
    span = 1
    for item in range(num):
        index = random.randint(0, len(list_Y) - span - 1)
        mat_w = mat_w + partial_derivative_of_negative_log_likelihood_of_w(index, index + span, mat_w, mat_X, mat_Y)
    true_num = 0
    false_num = 0
    i = 0.5

    for index in range(len(test_list_Y)):
        if probability_i(mat_w, test_mat_X[index].T) >= 0.5:
            if int(test_list_Y[index][0]) == 1:
                true_num += 1
            else:
                false_num += 1
        else:
            if int(test_list_Y[index][0]) == 0:
                true_num += 1
            else:
                false_num += 1
    print('true:', true_num)
    print('false:', false_num)
    print('正确率为：', true_num / (true_num + false_num))
    sum += true_num / (true_num + false_num)
print()
print('平均正确率为：', sum / test_num)
end = time.time()
print('消耗时间为：', end - start, '秒')
