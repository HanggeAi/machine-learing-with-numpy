import numpy as np


def loadDataSet(filename):
    """数据加载函数,但只返回整个,无单独的label"""
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        currentLine=line.strip().split("\t")
        fltLine=list(map(float,currentLine))
        dataMat.append(fltLine)
        
    return dataMat


def binSplitDataSet(dataSet,feature,value):
    """选择dataset的某个特征feature,将整个数据集按照feature与value的大小关系进行二分"""
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1
    
    
def regLeaf(dataSet):
    """负责生成叶结点的函数
    当函数chooseBestSplit确定不再对数据进行切分时,就会调用该函数,
    来得到叶结点的模型
    回归树中,该模型其实就是目标变量的均值
    """
    return np.mean(dataSet[:,-1])


def regErr(dataSet):
    """返回给定数据上的目标变量的平方误差,可理解为计算分类问题的基尼系数(实际上这是基于最小二乘偏差)"""
    return np.var(dataSet[:,-1])*dataSet.shape[0]


def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    """找到数据集的最佳二分切分方式
    如果找不到,则返回None并且同时调用creatTree来产生叶结点,叶结点也返回None
    ops中,第一个元素为容许下降的误差值,第二个元素是切分的最小样本数
    注意该函数有三个提前结束条件"""
    m,n=dataSet.shape
    S=errType(dataSet)    # 总体混乱程度
    tolS=ops[0]    # 能够容忍的最小方差变化量
    tolN=ops[1]    # 能够容忍的最小矩阵样本数量
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:    # 如果数据集中的目标值全部相等，则创建叶结点并退出
        return None,leafType(dataSet)
    
    # 初始化最优方差，最优特征，最优特征值
    bestS=np.inf
    bestIndex=0    # 最优特征的索引
    bestValue=0
    for featIndex in range(n-1):    # 遍历每个特征
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if mat0.shape[0]<tolN or mat1.shape[0]<tolN:    # 划分得到的子数据集的尺寸过小 则重新划分
                continue
            # 子数据集的方差和
            newS=errType(mat0)+errType(mat1)    # 新的混乱程度
            
            # 更新
            if newS<bestS:
                bestS=newS
                bestIndex=featIndex
                bestValue=splitVal
    
    if S-bestS<tolS:    # 如果误差减少不大，则退出
        return None,leafType(dataSet)
    
    # 使用最优特征和特征值切分
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if mat0.shape[0]<tolN or mat1.shape[0]<tolN:
        return None,leafType(dataSet)
    
    return bestIndex,bestValue



def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    常见决策树的递归函数
    leafType:给出建立叶结点的函数
    errType:给出误差计算函数
    """
    feat, val = chooseBestSplit(
        dataSet, leafType, errType, ops)    # 选择使得信息增益最大的特征及其分界值val
    
    if feat==None:    # 递归到没有特征可以划分了，则返回该特征对应的最佳划分值(这里不可以if not None 代替该语句！)
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 根据最优特征及其划分值,划分子集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(
        lSet, leafType, errType, ops)     # 利用划分后的左子集,递归创建左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree


def linearSolve(dataSet):
    m, n = dataSet.shape
    x = np.mat(np.ones((m, n)))
    y = np.mat(np.ones((m, 1)))
    x[:, 1:n] = dataSet[:, :n-1]
    y = dataSet[:, -1]

    xTx = x.T*x
    if np.linalg.det(xTx) == 0:
        print('the det of xTx is 0,None has been returned.')
        return np.array([0,0,0])
    ws = xTx.I*(x.T*y)
    return ws, x, y


def modelLeaf(dataSet):
    ws, X, y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """误差计算函数,基于线性模型的预测误差"""
    ws, X, y = linearSolve(dataSet)
    yHat = X*ws
    return np.sum(np.power(y-yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = inDat.shape[1]
    # 构造X
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


def treeForeCast(tree, inDat, modelEval=regTreeEval):
    if not isTree(tree):    # 如果tree不是一棵树,而只是一个叶子结点
        return regTreeEval(tree, inDat)
    # 如果tree是树，则需要递归预测
    if inDat[tree['spInd']] > tree['spVal']:    # 输入数据的特征值与树的分裂界限值比较，决定进入左子树还是右子树
        if isTree(tree['left']):    # 如果tree的左子节点为树，则递归预测
            return treeForeCast(tree['left'], inDat, modelEval)
        else:
            return modelEval(tree['left'], inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inDat, modelEval)
        else:
            return modelEval(tree['right'], inDat)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, testData[i], modelEval)    # 每次预测一条数据
    return yHat


def isTree(obj):
    """判断输入数据是否为一棵树(这里为字典)"""
    return isinstance(obj,dict)