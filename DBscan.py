import numpy as np
import math as m
from sklearn import metrics  #进行性能评估
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import queue

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False   #以上用来正常显示中文
createVar=locals()
NOISE = 0
UNASSIGNED = -1
labels_true=[-1]*600
data_path='synthetic_control.data'


def LoadTxt():
    """
    数据导入
    """
    data=np.loadtxt(data_path)
    return data


def DataS():
    """
    导入数据预处理之后的数据
    """
    return LoadTxt()


def label_init():
    """
    进行标签的初始化
    """
    del labels_true[:]
    for i in range(6):
        label=[i]*100
        labels_true.extend(label)
    return labels_true


def dist(a, b):
    """
    计算两个向量的距离
    """
    return m.sqrt(np.power(a-b, 2).sum())


def neighbor_points(data, pointId, radius):
    """
    得到邻域内所有样本点的Id
    """
    points = []
    for i in range(len(data)):
        if dist(data[i], data[pointId]) < radius:
            points.append(i)
    return np.asarray(points)


def to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
    """
    判断一个点是否是核心点，若是则将它和它邻域内的所用未分配的样本点分配给一个新类
    若邻域内有其他核心点，重复上一个步骤，但只处理邻域内未分配的点，并且仍然是上一个步骤的类
    """
    points = neighbor_points(data, pointId, radius)
    points = points.tolist()
    q = queue.Queue()
    if len(points) < minPts:
        clusterRes[pointId] = NOISE
        return False
    else:

        clusterRes[pointId] = clusterId    #对该点进行赋值
    for point in points:
        if clusterRes[point] == UNASSIGNED:
            q.put(point)
            clusterRes[point] = clusterId   #对该点的密度直达进行赋类
    print(clusterRes)
    while not q.empty():  #寻找该点的的密度相连 对队列的使用，这个人真的秀
        neighborRes = neighbor_points(data, q.get(), radius)
        if len(neighborRes) >= minPts:    # 核心点
            for i in range(len(neighborRes)):
                resultPoint = neighborRes[i]
                if clusterRes[resultPoint] == UNASSIGNED:
                    q.put(resultPoint)
                    clusterRes[resultPoint] = clusterId
                elif clusterRes[clusterId] == NOISE:
                    clusterRes[resultPoint] = clusterId
    return True


def dbscan(data, radius, minPts):
    """
    扫描整个数据集，为每个数据集打上核心点，边界点和噪声点标签的同时为
    样本集聚类
    """
    clusterId = 1
    nPoints = len(data)
    clusterRes = [UNASSIGNED] * 600
    print(clusterRes)
    for pointId in range(nPoints):
        if clusterRes[pointId] == UNASSIGNED:
            if to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
                clusterId = clusterId + 1
    return np.asarray(clusterRes), clusterId


if __name__ == '__main__':
    c=['标准','周期','递增','递减','递增向上','递减向下']
    labels_true=label_init()
    data = DataS()
    cluster = np.asarray(data)
    pca = PCA(n_components=2)  # 进行PCA降维
    newdata = pca.fit_transform(data)
    plt.scatter(newdata[:, 0], newdata[:, 1], c=labels_true)
    plt.show()
    print(newdata)
    clusterRes, clusterNum = dbscan(data, 47, 50)                                    #领域参数设置
    plt.scatter(newdata[:, 0], newdata[:, 1], c=clusterRes)
    plt.show()
    print(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=clusterRes))  # 进行ARI性能评估，取值[-1,1]越接近1，性能越好
    num = []
    for i in range(clusterNum):
        print(np.sum(clusterRes == i))
        num.append(np.sum(clusterRes == i))
    plt.bar(range(clusterNum), num)
    plt.show()