
def createDataSet():
	dataSet = [[0, 0, 0, 0, 'no'],
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
	return dataSet, labels

class ChooseTree:
    def __int__(self,dataSet,labels):
        self.dataSet = dataSet
        self.labels = labels
		self.featLabels = []
		self.tree = {}


    def createTree(self):
        """
        创建数模型
        """
        classList = [example[-1] for example in self.dataset]
		# 判断当前数据是否全为同一个结果。
        if classList.count(classList[0]) == len(classList):
            return classList[0]
		# 判断当前的标签是否只剩下一个
		if len(self.dataset[0]) == 1:
			return majority(self.dataset)

		# 找到最好的feature
		bestFeat = chooseBestFeatureToSplit(dataset)
		bestFeatLabel = labels[bestFeat]
		self.featLabels.append(bestFeatLabel)
		self.tree = {bestFeatLabel:{}}
		# 删除当前标签
		del self.labels[bestFeatLabel]

