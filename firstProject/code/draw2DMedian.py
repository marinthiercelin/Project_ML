import Helpers.helpers as helper
import Visualization.visu as visu
import cleaner
import numpy as np

y,x, ids = helper.load_csv_data('../data/train.csv', True)

path = '../data/Figures/2DMedian/'
xCl = cleaner.fillMissingValuesMedianWithY(xCl,y)

visu.produce2DFigures(xCl,y,path, True)
