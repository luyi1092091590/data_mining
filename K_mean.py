from numpy import *
from random import *
import matplotlib.pyplot as plt

data_list = []
for line in open('D:/testSet.txt').readlines():
    m, n = line.split()
    temp = float(m), float(n)  # 将数据全转换为float，方便求最大最小值
    data_list.append(list(temp))


def distance(list_a, list_b):  # 欧式距离
    return sqrt((list_a[0] - list_b[0]) ** 2 + (list_a[1] - list_b[1]) ** 2)


def rand_point(k, mat_data):  # 随机生成k个质点
    k_point = []
    column0_max = max(mat_data[:, 0])
    column0_min = min(mat_data[:, 0])
    column1_max = max(mat_data[:, 1])
    column1_min = min(mat_data[:, 1])
    for i in range(k):
        tem = uniform(float(column0_min), float(column0_max)), uniform(float(column1_min), float(column1_max))
        k_point.append(list(tem))
    return k_point


def family(all_data, k_data):  # 将所有点分给相应簇
    k_dic = {}
    for point_data in all_data:
        dis = inf
        k_point = []
        for point in k_data:
            if distance(point, point_data) < dis:
                dis = distance(point, point_data)  # 上面的欧式距离函数
                k_point = point  # 上面不写k_point = []这里就报错
        k_key = tuple(k_point)
        if k_dic.get(k_key, 0) == 0:  # if k_dic.get(k_key) == None:  报错！
            k_dic[k_key] = []
        k_dic[k_key].append(point_data)
    return k_dic


def replace_k_point(k_point_dic):  # 更新质点
    new_k_point = []
    for ky, v in k_point_dic.items():
        s1 = 0
        s2 = 0
        t = list(v)
        for l in t:
            s1 += l[0]
            s2 += l[1]
        new_a = s1 / len(k_point_dic[ky])
        new_b = s2 / len(k_point_dic[ky])
        new_point = new_a, new_b
        new_k_point.append(list(new_point))
    return new_k_point


num = 4  # 质点个数
k_list = rand_point(num, mat(data_list))
mark = True
d = family(data_list, k_list)
while mark:
    # print('旧字典', d)
    new_k = replace_k_point(d)
    # print('新质点列表', new_k)
    temp_list = []
    for k in d:
        temp_list.append(list(k))
    d = family(data_list, new_k)
    # print('新字典', d)
    # print('排序旧', sorted(temp_list))
    # print('排序新', sorted(new_k))
    if sorted(temp_list) == sorted(new_k):
        mark = False
        print('最终结果', d)


# 数据最大x=4.838138,最小x=-5.379713,最大y=5.1904，最小y=-4.232586
# 数据可视化
color = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
index = 0
a = []
b = []
for k, v in d.items():
    x = []
    y = []
    index += 1
    p, q = k
    a.append(p)
    b.append(q)
    plt.scatter(a, b, c='r', marker='+')        # 质心
    for i in range(len(v)):
        m, n = v[i]
        x.append(m)
        y.append(n)
        plt.scatter(x, y, c=color[index % 7])       # 对每个簇分不同颜色
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-6, 6, -6, 6])        # 横纵坐标范围
plt.show()
