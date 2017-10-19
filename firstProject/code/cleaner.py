import numpy as np

def cleanFeatures(x):
    toRemove = [4,6,12,24,25,27,28]
    xBis = np.delete(x,toRemove,axis=1)
    return xBis

def fillMissingValuesWithY(tx,y):
    x = np.copy(tx)
    for i in range(x.shape[1]):
        means = np.zeros(2)
        counts = np.zeros(2)
        for j in range(x.shape[0]):
            if x[j][i] != -999:
                means[int(y[j])] += x[j][i]
                counts[int(y[j])] += 1
        means = means/counts
        for j in range(x.shape[0]):
            if x[j][i] == -999:
                x[j][i] = means[int(y[j])]
    return x

def fillMissingValuesWithY(tx,y):
    x = np.copy(tx)
    

def fillMissingValuesWOY(tx):
    x = np.copy(tx)
    for i in range(x.shape[1]):
        mean = 0
        count = 0
        for j in range(x.shape[0]):
            if x[j][i] != -999:
                mean += x[j][i]
                count += 1
        mean = mean/count
        for j in range(x.shape[0]):
            if x[j][i] == -999:
                x[j][i] = mean
    return x

def addConstant(x):
    np.c_[np.ones((x.shape[0], 1)), x]
    return x
