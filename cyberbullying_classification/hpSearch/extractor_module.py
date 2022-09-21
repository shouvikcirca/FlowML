import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def extractor(filePath):
	files = [i for i in os.listdir(filePath) if '.txt' in i] # Only .txt files contain data
	comments = []
	targets = []
	for f in files:
		with open(filePath+f, 'r') as r:
			s = r.read().split('\n')
			s = [i.split('\t') for i in s]
			comments.extend(s)	

	targets = [int(item[0]) for item in comments if len(item) == 2]	
	comments = [item[1] for item in comments if len(item) == 2]
	return comments, targets


if __name__ == "__main__":
	filePath = './'
	comments, targets = extractor('./')
	comments = np.array(comments)
	targets = [i-1 for i in targets]
	targets = np.array(targets)

	splitter = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 0)

	for train_index, test_index in splitter.split(comments, targets):
		trainX = comments[train_index]
		trainY = targets[train_index]
		testX = comments[test_index]
		testY = targets[test_index]

		print('trainX: {}'.format(trainX.shape))
		print('trainY: {}'.format(trainY.shape))
		print('testX: {}'.format(testX.shape))
		print('testY: {}'.format(testY.shape))
