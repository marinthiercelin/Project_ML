import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import Helpers.cleaner as cleaner
import numpy as np


y,x, ids = helper.load_csv_data('../data/train.csv', True)

ratio = 0.8
lambda_ = 0.5
gamma = 0.08
max_iters = 500

x_tr,y_tr,x_te,y_te = tool.split_data(xCl, y, ratio, seed=3)
x_tr = cleaner.normalize_input(x_tr)
x_te = cleaner.normalize_input(x_te)


xCl = cleaner.fillMissingValuesMediansWithY(x,y)
xCl,yCl = cleaner.filter_bad_samples(xCl,y)
print(xCl)
tool.full_cross_validation(xCl,yCl)
