import numpy as np
import Helpers.helpers as helper
# import Helpers.cleaner as cleaner
from Helpers.cleaner import *

def main():
    y,x, ids = helper.load_csv_data('../data/train.csv', False)

    x_0 = x[np.where(x[:,22] == 0)]
    y_0 = helper.changeYtoBinary(y[np.where(x[:,22] == 0)])
    x_1 = x[np.where(x[:,22] == 1)]
    y_1 = helper.changeYtoBinary(y[np.where(x[:,22] == 1)])
    x_2 = x[np.where(x[:,22] == 2)]
    y_2 = helper.changeYtoBinary(y[np.where(x[:,22] == 2)])
    x_3 = x[np.where(x[:,22] == 3)]
    y_3 = helper.changeYtoBinary(y[np.where(x[:,22] == 3)])

    xCl_0, yCl_0, gc_0, fl_0, std_0, mean_0 = cleanTrain(x_0,y_0)
    xCl_1, yCl_1, gc_1, fl_1, std_1, mean_1 = cleanTrain(x_1,y_1)
    xCl_2, yCl_2, gc_2, fl_2, std_2, mean_2 = cleanTrain(x_2,y_2)
    xCl_3, yCl_3, gc_3, fl_3, std_3, mean_3 = cleanTrain(x_3,y_3)

    xCl_0 = remove_unused_features(xCl_0,[3])
    xCl_1 = remove_unused_features(xCl_1,[18])


    helper.save_clean_data(xCl_0, yCl_0, '../data/clean/x_train_0.npy', '../data/clean/y_train_0.npy')
    helper.save_clean_data(xCl_1, yCl_1, '../data/clean/x_train_1.npy', '../data/clean/y_train_1.npy')
    helper.save_clean_data(xCl_2, yCl_2, '../data/clean/x_train_2.npy', '../data/clean/y_train_2.npy')
    helper.save_clean_data(xCl_3, yCl_3, '../data/clean/x_train_3.npy', '../data/clean/y_train_3.npy')


    y_te, x_te, ids_te = helper.load_csv_data('../data/test.csv', False)

    ids_te_0 = ids_te[np.where(x_te[:,22] == 0)]
    ids_te_1 = ids_te[np.where(x_te[:,22] == 1)]
    ids_te_2 = ids_te[np.where(x_te[:,22] == 2)]
    ids_te_3 = ids_te[np.where(x_te[:,22] == 3)]

    x_te_0 = x_te[np.where(x_te[:,22] == 0)]
    x_te_1 = x_te[np.where(x_te[:,22] == 1)]
    x_te_2 = x_te[np.where(x_te[:,22] == 2)]
    x_te_3 = x_te[np.where(x_te[:,22] == 3)]

    xCl_te_0 = cleanTest(x_te_0, gc_0, fl_0, std_0, mean_0)
    xCl_te_1 = cleanTest(x_te_1, gc_1, fl_1, std_1, mean_1)
    xCl_te_2 = cleanTest(x_te_2, gc_2, fl_2, std_2, mean_2)
    xCl_te_3 = cleanTest(x_te_3, gc_3, fl_3, std_3, mean_3)

    xCl_te_0 = remove_unused_features(xCl_te_0,[3])
    xCl_te_1 = remove_unused_features(xCl_te_1,[18])

    np.save('../data/clean/ids_test_0.npy',ids_te_0)
    np.save('../data/clean/ids_test_1.npy',ids_te_1)
    np.save('../data/clean/ids_test_2.npy',ids_te_2)
    np.save('../data/clean/ids_test_3.npy',ids_te_3)

    helper.save_clean_data(xCl_te_0, np.array([]), '../data/clean/x_test_0.npy', '../data/clean/y_test_0.npy')
    helper.save_clean_data(xCl_te_1, np.array([]), '../data/clean/x_test_1.npy', '../data/clean/y_test_1.npy')
    helper.save_clean_data(xCl_te_2, np.array([]), '../data/clean/x_test_2.npy', '../data/clean/y_test_2.npy')
    helper.save_clean_data(xCl_te_3, np.array([]), '../data/clean/x_test_3.npy', '../data/clean/y_test_3.npy')


def cleanTrain(x,y):
    yCl = helper.changeYtoBinary(y)
    good_columns = select_bad_features(x)
    xCl = x[:,good_columns]
    xCl, yCl = filter_bad_samples(xCl, yCl)
    xCl = fillMissingValuesMediansWithY(xCl,yCl)
    xCl = outliersToMedian(xCl)
    xCl, filtered, std, mean = filterAndNormalize(xCl)
    return xCl,yCl,good_columns,filtered, std, mean

def cleanTest(x,gc,filt,std,mean):
    xCl = x[:,gc]
    xCl = fillMissingValuesMediansWOY(xCl)
    xCl = remove_unused_features(xCl, filt)
    xCl = (xCl - mean)/std
    return xCl

def filterAndNormalize(x):
    std = np.std(x,0)
    filtered = np.where(std == 0)[0]
    xCl = remove_unused_features(x,filtered)
    std = std[np.where(std != 0)]
    mean = np.mean(xCl,0)
    xCl = (xCl - std)/mean
    return xCl,filtered,std,mean


if __name__ == '__main__':
    main()
