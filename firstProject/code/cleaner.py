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
                means[(1 if(y[j] == 1) else 0)] += x[j][i]
                counts[(1 if(y[j] == 1) else 0)] += 1
        means = means/counts
        for j in range(x.shape[0]):
            if x[j][i] == -999:
                x[j][i] = means[(1 if(y[j] == 1) else 0)]
    return x

def fillMissingValuesMedianWithY(tx,y):
    x = np.copy(tx)
    x0 = x[np.where(y == 0)]
    x1 = x[np.where(y == 1)]
    for i in range(x.shape[1]):
        med = [np.median(x0[np.where(x0[:,i] != -999)][:,i]), np.median(x1[np.where(x1[:,i] != -999)][:,i])]
        for j in range(x.shape[0]):
            if y[j] == 0 and x[j,i] == -999:
                x[j,i] = med[0]
            if y[j] == 1 and x[j,i] == -999:
                x[j,i] = med[1]
    return x

def fillMissingValuesMedianWOY(tx):
    x = np.copy(tx)
    for i in range(x.shape[1]):
        m = x[:,i][np.where(x[:,i] != -999)]
        med = np.median(m)
        x[:,i][np.where(x[:,i] == -999)] = med
    return x

def fillMissingValuesMedianWOY(tx):
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
    x = np.c_[np.ones((x.shape[0], 1)), x]
    return x


def normalize_input(x):
    means = np.mean(x, 0)
    stds = np.std(x, 0)

    return (x - means)/stds


def filter_bad_samples(x, freedom=0.2):
    """removes all samples from x that have a percentage
    of unknown features that is greater or equal to freedom%"""

    D = x.shape[1]
    max_bad = int(D * freedom)

    return xCl[(xCl < -900).sum(axis=1) < max_bad]
