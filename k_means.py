from numpy import *


# 计算欧几里得距离
def distEclud(vecA, vecB):
    # 求两个向量之间的距离
    return sqrt(sum(power(vecA - vecB, 2)))


# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    # 每个质心有n个坐标值，总共要k个质心
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


# k-means聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    # 用来判断聚类是否已经收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 把每一个数据点划分到离它最近的中心点
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i, 0] != minIndex:
                # 如果分配发生变化，则需要继续迭代
                clusterChanged = True
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        # 重新计算中心点
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent, :] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment