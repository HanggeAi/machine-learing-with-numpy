import numpy as np


def loadDataSet(filename):
    """数据加载函数"""
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fline=list(map(float, curLine))
        dataMat.append(fline)
    return dataMat


def createCent(dataSet:np.matrix,k):
    n=dataSet.shape[1]    # 特征维度
    centerInit=np.mat(np.zeros((k,n)))
    dataMin=dataSet.min(axis=0)
    dataMax=dataSet.max(axis=0)
    dataRange=dataMax-dataMin
    
    for i in range(k):
        centerInit[k,:]=dataMin+dataRange*np.random.randn(1,n)
    return centerInit


def distCal(vecA,vecB):
    return np.sqrt(np.power(vecA-vecB,2))


def kMeans1(dataSet:np.matrix, k=3):
    m=dataSet.shape[0]
    clusterAss=np.mat(np.zeros((m,2)))
    clusterChanged=True
    clusterCenter=createCent(dataSet,k)    # 随机初始化各个簇的质心
    
    while clusterChanged:
        clusterChanged=False
        for i in range(m):    # 对每个样本
            lowetDist=np.inf
            belongClust=-1
            for j in range(len(clusterCenter)):    # 对每个聚类中心
                dist=distCal(dataSet[i,:],clusterCenter[j,:])
                if dist < lowetDist:
                    lowetDist=dist
                    belongClust=j
            if clusterAss[i,:]!=belongClust:
                clusterChanged=True
            clusterAss[i,:]=belongClust,lowetDist
            
        for c in range(k):
            ptsInClust=dataSet[np.nonzero(clusterAss[:,0]==c)[0],:]
            clusterCenter[c,:]=np.mean(ptsInClust,axis=0)    # 更新聚类中心
            
    return clusterCenter,clusterAss


dataPath = 'D:\\机器学习实战代码\\machinelearninginaction\\Ch06\\testSet.txt'
datMat = np.mat(loadDataSet(dataPath))
print(kMeans1(datMat,k=4))