from math import log


def createDataSet():
    dataSet = [
        [0, 0, 0, 0, "no"],
        [0, 0, 0, 1, "no"],
        [0, 1, 0, 1, "yes"],
        [0, 1, 1, 0, "yes"],
        [0, 0, 0, 0, "no"],
        [1, 0, 0, 0, "no"],
        [1, 0, 0, 1, "no"],
        [1, 1, 1, 1, "yes"],
        [1, 0, 1, 2, "yes"],
        [1, 0, 1, 2, "yes"],
        [2, 0, 1, 2, "yes"],
        [2, 0, 1, 1, "yes"],
        [2, 1, 0, 1, "yes"],
        [2, 1, 0, 2, "yes"],
        [2, 0, 0, 0, "no"],
    ]
    labels = ["F1-AGE", "F2-WORK", "F3-HOME", "F4-LOAN"]
    return dataSet, labels


class ChooseTree:
    def __init__(self, dataSet, labels):
        self.dataset = dataSet
        self.labels = labels
        self.featLabels = []
        self.tree = {}

    def createTree(self, dataset=None, labels=None):
        """
        创建树模型
        """
        if dataset is None:
            dataset = self.dataset
        if labels is None:
            labels = self.labels

        classList = [example[-1] for example in dataset]
        # 判断当前数据是否全为同一个结果。
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 判断当前的标签是否只剩下一个
        if len(dataset[0]) == 1:
            return self.majority(dataset)
        # 找到最好的feature
        bestFeat = self.chooseBestFeatureToSplit(dataset)
        bestFeatLabel = labels[bestFeat]
        self.featLabels.append(bestFeatLabel)

        tree = {bestFeatLabel: {}}

        # 复制一份labels，避免修改原始labels
        subLabels = labels[:]
        # 删除当前标签
        del subLabels[bestFeat]
        # 先组合最好的一列
        featuresVal = [exm[bestFeat] for exm in dataset]
        # 拿到这一列的唯一值
        featuresVal = set(featuresVal)
        for val in featuresVal:
            subDataSet = self.splitData(dataset, bestFeat, val)
            tree[bestFeatLabel][val] = self.createTree(subDataSet, subLabels)
        return tree

    def chooseBestFeatureToSplit(self, dataset):
        numFeatures = len(dataset[0]) - 1
        baseEntropy = self.calcShannonEnt(dataset)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataset]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitData(dataset, i, value)
                prob = len(subDataSet) / float(len(dataset))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def calcShannonEnt(self, dataset):
        # 计算信息熵
        numEntries = len(dataset)
        labelCount = {}
        for featVec in dataset:
            currentLabel = featVec[-1]
            if currentLabel not in labelCount.keys():
                labelCount[currentLabel] = 0
            labelCount[currentLabel] += 1
        shannonEnt = 0
        for key in labelCount:
            prob = float(labelCount[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def splitData(self, dataset, bestFeat, val):
        retDataSet = []
        for featVec in dataset:
            if featVec[bestFeat] == val:
                reducedFeatVec = featVec[:bestFeat]
                reducedFeatVec.extend(featVec[bestFeat + 1 :])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    @staticmethod
    def majority(dataset):
        classCount = {}
        for vote in dataset:
            currentLabel = vote[-1]  # 取最后一个元素作为类别标签
            if currentLabel not in classCount.keys():
                classCount[currentLabel] = 0
            classCount[currentLabel] += 1
        sortedClassCount = sorted(
            classCount.items(), key=lambda item: item[1], reverse=True
        )
        return sortedClassCount[0][0]


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    ct = ChooseTree(dataSet, labels)
    model = ct.createTree()
    print(model)
