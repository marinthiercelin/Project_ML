import numpy as np

def cleanFeatures(x):
    toRemove = [4,6,12,24,25,27,28]
    xBis = np.delete(x,toRemove,axis=1)
    return xBis
