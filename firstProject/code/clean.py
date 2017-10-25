import numpy as np
import Helpers.helpers as helper
from Helpers.cleaner import *

def main():
    y,x, ids = helper.load_csv_data('../data/train.csv', False)

    x = filter_bad_features(x)
    x, y = bad_samples = filter_bad_samples(x, y)



if __name__ == '__main__':
    main()
