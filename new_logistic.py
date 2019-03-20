import math
import numpy
import random
import time


def sigmoid(z):
    p = 1/(1+math.exp(-z))
    return p


train_path = 'D:/SMSSpamCollection.txt'
test_path = 'D:/test_set_new.txt'
all_words = []
start = time.time()
for line in open(train_path, encoding='ISO-8859-1').readlines():
    temp_list = line.split()
    all_words.extend(temp_list[1:])      # 去除ham,spam的标记
all_words_list = list(set(all_words))       # 去重
list_X = []
lable_list = []
for line in open(train_path, encoding='ISO-8859-1').readlines():
    temp_list = line.split()
    word_vec = [1]
    for word in all_words_list:     # one-hot
        if word in temp_list[1:]:
            word_vec.append(1)
        else:
            word_vec.append(0)
    list_X.append(word_vec)
    if temp_list[0] == 'spam':      # 垃圾标1，正常标0
        lable_list.append(1)
    else:
        lable_list.append(0)
mat_X = numpy.mat(list_X)       # 矩阵化
print('标签数', len(lable_list))

test_list_X = []
test_lable_list = []
for line in open(test_path, encoding='ISO-8859-1').readlines():
    temp_list = line.split()
    test_word_vec = [1]
    for word in all_words_list:  # one-hot
        if word in temp_list[1:]:
            test_word_vec.append(1)
        else:
            test_word_vec.append(0)
    test_list_X.append(test_word_vec)
    if temp_list[0] == 'spam':  # 垃圾标1，正常标0
        test_lable_list.append(1)
    else:
        test_lable_list.append(0)
test_mat_X = numpy.mat(test_list_X)
print('长度', len(test_lable_list))

s = 0       # 正确率总和
for n_th in range(100):         # 取样100个
    m, n = numpy.shape(mat_X)   # 获取矩阵m行n列
    weights = numpy.ones((n, 1))      # 初始化权重，生成n行1列全是1的矩阵
    for i in range(10000):           # 迭代次数
        temp = random.randint(0, len(lable_list)-1)     # 此函数是前闭后闭的，必须-1
        h = sigmoid(mat_X[temp]*weights)
        weights = weights + 1 * mat_X[temp].transpose() * (lable_list[temp] - h)
    true_num = 0
    false_num = 0
    for j in range(len(test_lable_list)):
        if sigmoid(test_list_X[j]*weights) >= 0.5:
            if test_lable_list[j] == 1:
                true_num += 1
            else:
                false_num += 1
        else:
            if test_lable_list[j] == 0:
                true_num += 1
            else:
                false_num += 1
    print('true_num:', true_num)
    print('false_num:', false_num)
    print('正确率:', true_num / (true_num+false_num))
    s += true_num / (true_num+false_num)
print('平均正确率:', s/100)
end = time.time()
print('时间:', end-start)
