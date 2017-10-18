import numpy as np

def cleanFeatures(x):
    toRemove = [4,6,12,24,25,27,28]
    xBis = np.delete(x,toRemove,axis=1)
    return xBis

#TODO : améliorer ça
def fillMissingValues(x):
    x[np.where(x == -999)] = 0
    return x

def addConstant(x):
    np.c_[np.ones((x.shape[0], 1)), x]
    return x


def normalize_input(x):
    means = np.mean(x, 0)
    stds = np.std(x, 0)

    return (x - means)/stds
