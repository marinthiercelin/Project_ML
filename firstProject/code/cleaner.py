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

def outliersToMedian(x):
    for col in range(x.shape[1]):
        x_col = x[:,col]
        column = x_col[np.where(x_col != -999)]
        median = np.median(column)
        quInf = np.percentile(column,25)
        quSup = np.percentile(column,75)
        iqr = quSup - quInf
        loBound = quInf - 1.5*iqr
        upBound = quSup + 1.5*iqr
        x_col[np.where(x_col < loBound)] = median
        x_col[np.where(x_col > upBound)] = median
    return x

#useless, they are all outliers for something
def deleteOutliers(x):
    toDel = np.array([])
    for col in range(x.shape[1]):
        x_col = x[:,col]
        column = x_col[np.where(x_col != -999)]
        median = np.median(column)
        quInf = np.percentile(column,25)
        quSup = np.percentile(column,75)
        woUnknown = np.copy(x_col)
        woUnknown[np.where(woUnknown == -999)] = median
        iqr = quSup - quInf
        loBound = quInf - 1.5*iqr
        upBound = quSup + 1.5*iqr
        toDel = np.union1d(toDel, np.where(woUnknown < loBound)[0])
        toDel = np.union1d(toDel, np.where(woUnknown > upBound)[0])
    print(toDel.shape[0],x.shape[0])
    x = np.delete(x,toDel,axis=0)
    return x
