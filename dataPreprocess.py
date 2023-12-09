import pandas as pd
import numpy as np
import json

def typeHandler(it):
    if isinstance(it, np.integer):
        return int(it)
    elif isinstance(it, np.floating):
        return float(it)
    elif isinstance(it, np.ndarray):
        return it.tolist()
    else:
        return it

# read in 60000 train data to json for convenience
js = {}
trainData = pd.read_csv('data\mnist_train.csv')

trainVectors = []
trainLabels = []
sortByLabel = [[],[],[],[],[],[],[],[],[],[]] #sort vectors by label

for index in range(trainData.shape[0]):
    tmpVec = []
    trainLabels.append(trainData.iloc[index, 0])
    sortByLabel[trainData.iloc[index, 0]].append(index)
    for i in range(1, 785):
        tmpVec.append(trainData.iloc[index, i])
    trainVectors.append(tmpVec)

js['trainVectors'] = trainVectors
js['trainLabels'] = trainLabels
js['sortByLabel'] = sortByLabel

with open('BaseJson.json', 'w') as fp:
    json.dump(js, fp, default=typeHandler)
