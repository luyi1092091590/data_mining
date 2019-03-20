import numpy as np
import matplotlib.pyplot as plt

num_list = []
temp = 0
for line in open('D:/zhengqi_train.txt').readlines():
    if temp != 0:
        num = list(map(float, line.split()))
        num_list.append(num)
    temp += 1
# print(num_list)
num_mat = np.mat(num_list)
# print(list(num_mat[:, 0]))

# V0,V1,V4(还行),V8,V27,V31(还行),V37(还行)，
temp_list = []
for a in range(len(num_list)):
    temp_list.append(1)
# print(len(num_list[0]))   # 39
for i in range(38):
    plt.scatter(list(num_mat[:, i]), temp_list, c='b', s=0.05)  # 一维图
    # plt.savefig('V'+str(i))
    plt.show()
    '''
    plt.scatter(list(num_mat[:, i]), list(num_mat[:, 38]), c='b')   #  二维图
    # path = 'V'+str(i)+'-'+'target'
    plt.title(str('V'+str(i)+'-'+'target'))
    plt.savefig(str('V'+str(i)+'-'+'target'))
    plt.show()
    '''
