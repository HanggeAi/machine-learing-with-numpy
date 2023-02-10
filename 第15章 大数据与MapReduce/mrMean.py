from mrjob.job import MRJob


class MRMean(MRJob):
    def __init__(self, *args, **kwargs):
        # 利用super,调用父类的__init__()方法，为父类传入初始化参数
        super(MRMean, self,).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqsum = 0

    def map(self, key, val):
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqsum += inVal*inVal

    def make_runner(self):
        mn = self.inSum/self.inCount
        mnSq = self.inSqsum/self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedVals):
        cumVal = 0.0
        cumSumSq = 0.0
        cumN = 0.0
        for valArr in packedVals:
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj*float(valArr[1])
            cumSumSq += nj*float(valArr[2])
        mean = cumVal/cumN
        var = (cumSumSq-2*mean*cumVal+cumN*mean*mean)/cumN
        yield (mean, var)

    def steps(self):
        return ([self.mr(mappr=self.map, reducer=self.reduce, mapper_final=self.mapper_final)])


if __name__ == "__main__":
    MRMean.run()
