import numpy as np
import Helpers.helpers as helper
def treatData(x,withY = True,y = []):
    xCl = x
    YCl = y
    if(withY):
        xCl = fillMissingValuesMediansWithY(x,y)
        yCl = helper.changeYtoBinary(y)
    else:
        xCl = fillMissingValuesMedianWOY(x)
    xCl = outliersToMedian(x)
    xCl = normalize_input(x)
    if(withY):
        return yCl, xCl
    else:
        return xCl

def fillMissingValuesMeansWithY(tx,y):
    """ Fill the missing values in tx with the mean of all real values of his category for this feature.

    Compute the means of all features in tx depending of their category.
    Change all missing value (-999) by the right mean.
    Return the new tx cleaned.
    """
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

def fillMissingValuesMeansWOY(tx):
    """ Fill the missing values in tx with the mean of all real values for this feature.

    Compute the means of all features in tx.
    Change all missing value (-999) by the right mean.
    Return the new tx cleaned.
    """
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

def fillMissingValuesMediansWithY(tx,y):
    """ Fill the missing values in tx with the median of all real values of his category for this feature.

    Compute the median of all features in tx depending of their category.
    Change all missing value (-999) by the right median.
    Return the new tx cleaned.
    """
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

def fillMissingValuesMediansWOY(tx):
    """ Fill the missing values in tx with the median of all real values for this feature.

    Compute the median of all features in tx.
    Change all missing value (-999) by the right median.
    Return the new tx cleaned.
    """
    x = np.copy(tx)
    for i in range(x.shape[1]):
        m = x[:,i][np.where(x[:,i] != -999)]
        med = np.median(m)
        x[:,i][np.where(x[:,i] == -999)] = med
    return x

def addConstant(x):
    """Add the constant 1 in front of all sample"""
    x = np.c_[np.ones((x.shape[0], 1)), x]
    return x


def normalize_input(x):
    """Normalize the data x by substracting the mean and deviding by his variance."""
    means = np.mean(x, 0)
    stds = np.std(x, 0)
    return (x - means)/stds




def select_bad_samples(x, freedom=0.2):
    """finds the indices of the samples that have a percentage
    of unknown features that is greater or equal to freedom"""
    D = x.shape[1]
    max_bad = int(D * freedom)

    return (x < -900).sum(axis=1) < max_bad

def filter_bad_samples(x, y, freedom=0.2):
    """removes all samples from x that have a percentage
    of unknown features that is greater or equal to freedom"""

    D = x.shape[1]
    max_bad = int(D * freedom)

    indices = select_bad_samples(x, freedom=0.2)
    return x[indices], y[indices]


def filter_bad_features(x, freedom=0.6):
    """removes all features from x that have a percentage
    of unknown samples that is greater or equal to freedom%"""

    N, D = x.shape
    max_bad = int(N * freedom)

    return x[:,(x < -900).sum(axis=0) < max_bad]

def outliersToMedian(x):
    """Put all the statisctical outliers to the median """
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

#compute and print correlation
def computeCorrelation(x):
    covariance = np.cov(x.T)
    correl = np.ones((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        x_i = x[:,i]
        std_i = np.std(x_i)
        for j in range(i+1,x.shape[1]):
            x_j = x[:,j]
            std_j = np.std(x_j)
            cor_ij = covariance[i,j]/(std_i*std_j)
            correl[i,j] = cor_ij
            correl[j,i] = cor_ij

    return np.tril(correl)
