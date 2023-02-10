import sys
import numpy as np


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in input]
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0

for instance in mapperOut:
    nj = float(instance[0])    # 样本个数
    cumN += nj
    cumVal += nj*float(instance[1])    # 所有值的总和
    cumSumSq += nj*float(instance[2])    # 所有值的总的平方和
mean = cumVal/cumN    # 合并之后的总体均值
varSum = (cumSumSq-2*mean*cumVal+cumN*mean*mean)/cumN
print("%d\t%f\t%f" % (cumN, mean, varSum))
# 此语法意味着写入文件对象（在本例中为sys.stderr），而不是标准输出。
print("report: still alive.", file=sys.stderr)
