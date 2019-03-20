import copy
import time
import numpy as np
import matplotlib.pyplot as plt
start_1=time.time()
start = time.time()

# 读取数据
data_set = []
with open("D:/outlier_show_data.txt") as fp:
    for line in fp.readlines():
        temp_list_1 = line.split()
        temp_list_2 = [int(item) for item in temp_list_1]
        data_set.append(temp_list_2)
data_set_copy = copy.deepcopy(data_set)
data_set = np.mat(data_set)
print("读取数据消耗：", time.time() - start, "s")

# 获取点的数量
point_num = data_set.shape[0]
# 根据经验自定义k(数据量的1/20左右)
#k = int(point_num/20)
k=15


# 计算两个向量点的欧式距离
def dist(vector_1, vector_2):
    '''
    distance=0
    for index in range(len(vector_1)):
        distance+=(vector_1[:,index]-vector_2[:,index])**2
    return int(distance)**(0.5)
    '''
    return np.linalg.norm(vector_1 - vector_2)

start = time.time()

# 计算点与点距离
dist_list = [[0 for i in range(point_num)] for y in range(point_num)]
for index_1 in range(0, point_num):
    temp_data=data_set[index_1]
    for index_2 in range(index_1 + 1, point_num):
        distance=dist(temp_data, data_set[index_2])
        dist_list[index_1][index_2] = distance
        dist_list[index_2][index_1] = distance
print("计算距离消耗：", time.time() - start, "s")

start = time.time()
# 计算dist_k
temp_dist_list = copy.deepcopy(dist_list)
dist_k = []
for index in range(point_num):
    temp_dist_list[index].sort()
for index in range(point_num):
    dist_k.append(temp_dist_list[index][k])
    
print("计算dist_k消耗：", time.time() - start, "s")
start = time.time()
# 计算N_k
N_k = [[] for item in range(point_num)]
for index_1 in range(point_num):
    for index_2 in range(point_num):
        if dist_list[index_1][index_2] <= dist_k[index_1] and index_1 != index_2:
            N_k[index_1].append(index_2)
print("计算N_k消耗：", time.time() - start, "s")

# 定义reachdist_k(index_1,index_2) index_1到index_2
def reachdist_k(index_1, index_2):
    temp_dist_k = dist_k[index_2]
    # temp_dist = dist(data_set[index_1], data_set[index_2])
    temp_dist=dist_list[index_1][index_2]
    if temp_dist_k > temp_dist:
        return temp_dist_k
    else:
        return temp_dist


# 定义sum_reachdist_k(index)
def sum_reachdist_k(index):
    sum = 0
    for i in N_k[index]:
        sum += reachdist_k(index, i)
    return sum

start = time.time()
sum_reachdist_k_list = []
for index in range(point_num):
    sum_reachdist_k_list.append(sum_reachdist_k(index))
print("计算sum_reachdist_k_list消耗：", time.time() - start, "s")

# 定义lrd_k(index)
def lrd_k(index):
    return len(N_k[index]) / sum_reachdist_k(index)
    # return len(N_k[index]) / sum_reachdist_k_list[index]

# 定义lof_k(index)
def lof_k(index):
    temp_list = [lrd_k(item) for item in N_k[index]]
    # temp_list = [lrd_k_list[index] for item in N_k[index]]
    return sum(temp_list) / (lrd_k(index) * len(N_k[index]))
    # return sum(temp_list) / (lrd_k_list[index] * len(N_k[index]))

start = time.time()
lof_k_list = []
for index in range(point_num):
    lof_k_list.append(lof_k(index))
print("计算lof_k_list消耗：", time.time() - start, "s")

start = time.time()
plt.figure()
for index in range(point_num):
    plt.scatter(data_set_copy[index][0], data_set_copy[index][1], \
                marker='o', color='black', s=(lof_k_list[index] ) * 10)
plt.grid(False)
print("图像显示消耗：", time.time() - start, "s")
print("总消耗：", time.time() - start_1, "s")
plt.show()
