import numpy as np
import Helpers.helpers as helper
# import Helpers.cleaner as cleaner
from Helpers.cleaner import *

def main():
    y,x, ids = helper.load_csv_data('../data/train.csv', False)

    x = filter_bad_features(x)
    x, y = bad_samples = filter_bad_samples(x, y)

    x = fillMissingValuesMediansWithY(x,y)
    y = helper.changeYtoBinary(y)
    x = normalize_input(x)


    helper.save_clean_data(x, y, '../data/')


if __name__ == '__main__':
    main()
