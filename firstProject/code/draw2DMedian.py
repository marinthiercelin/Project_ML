"""Draw a plot for all the data cleaned.

Load a subset of the data from the train.csv file.
Clean the data with the function fillMissingValuesMedianWithY.
Draw all plots of the clean data (1 in function of all other).
Save those drawing in ../data/Figures/2DMedian.
"""

import Helpers.helpers as helper
import Visualization.visu as visu
import cleaner
import numpy as np

y,x, ids = helper.load_csv_data('../data/train.csv', True)

path = '../data/Figures/2DMedian/'
xCl = cleaner.fillMissingValuesMediansWithY(x,y)
xCl = cleaner.outliersToMedian(xCl)
xCl = cleaner.normalize_input(xCl)

visu.produce2DFigures(xCl,y,path)
