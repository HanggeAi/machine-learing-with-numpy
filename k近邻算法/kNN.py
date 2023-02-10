import numpy as np


def classify0(X, dataSet, labels, k):
    """
    X:待测样本,待分类样本
    dataSet:已有样本数据
    labels:已有样本标签
    k:近邻个数
    """
    # 计算距离
    diffMat = dataSet-X    # 利用广播，计算差
    diffSquare = diffMat**2    # 各项平方
    distances = np.sqrt(diffSquare.sum(axis=1))    # 求和 然后开方

    # 按照距离，从小到大排序
    disIdx = distances.argsort()

    classCount = {}    # 用于保存每一类的样本个数（前k个之中）
    for i in range(k):
        label = labels[disIdx[i]]    # 第I个样本的标签
        classCount[label] = classCount.get(label, 0)+1    # 字典获取label对应的值，自增1

    keys = np.array([key for key in classCount.keys()])
    value = np.array([val for val in classCount.values()])

    val_argsort = value.argmax()

    return keys[val_argsort]
