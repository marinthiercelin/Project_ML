import numpy as np
import Helpers.helpers as helper
# import Helpers.cleaner as cleaner
from Helpers.cleaner import *

def main():
    y,x, ids = helper.load_csv_data('../data/train.csv', False)

    good_columns = select_bad_features(x)
    x = x[:,good_columns]
    x, y = filter_bad_samples(x, y)
    x = remove_unused_features(x, [2,22,17,18])
    x = fillMissingValuesMediansWithY(x,y)
    x = outliersToMedian(x)
    y = helper.changeYtoBinary(y)
    std_x = np.std(x)
    mean_x = np.mean(x)
    x = (x - mean_x)/std_x

    helper.save_clean_data(x, y, '../data/x_train.npy', '../data/y_train.npy')

    y_te, x_te, ids_te = helper.load_csv_data('../data/test.csv', False)

    x_te = x_te[:,good_columns]
    x_te = remove_unused_features(x_te, [2,22,17,18])
    x_te = fillMissingValuesMediansWOY(x_te)
    y_te = helper.changeYtoBinary(y_te)
    x_te = (x_te - mean_x)/std_x

    helper.save_clean_data(x_te, y_te, '../data/x_test.npy', '../data/y_test.npy')


if __name__ == '__main__':
    main()
