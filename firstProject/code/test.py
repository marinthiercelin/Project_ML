import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import cleaner
import numpy as np


y,x, ids = helper.load_csv_data('../data/train.csv', True)

path = '../data/Figures/1DwithDeg/'
xCl = cleaner.cleanFeatures(x)
xCl = cleaner.fillMissingValuesWithY(xCl,y)

xCl = cleaner.filter_bad_samples(xCl)
