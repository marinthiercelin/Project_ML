import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import cleaner
import numpy as np

y,x, ids = helper.load_csv_data('../data/train.csv', False)

path = '../data/Figures/2D/'
xCl = cleaner.cleanFeatures(x)
#xCl = cleaner.fillMissingValues(xCl)
xCl = cleaner.addConstant(xCl)
#visu.produce2DFigures(xCl,y,folder_path = path, save = True)

#parameters
ratio = 0.8
#lambda_ = 0.1
gamma = 0.1
max_iters = 50

x_tr,y_tr,x_te,y_te = tool.split_data(xCl, y, ratio, seed=3)

initial_w = np.array(x_tr.shape[1]*[0])
#loss_tr_1, w = tool.reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, 50, gamma)
#loss_tr_1, w = tool.logistic_regression(y_tr, x_tr,initial_w, max_iters, gamma)
loss_tr_1, w = tool.least_squares_GD(y_tr,x_tr,initial_w,max_iters,gamma)
y_pred = helper.predict_labels(w,x_te)

res = [(1 if(y_p == y_te[i]) else 0) for i,y_p in enumerate(y_pred)]

print("w is ", w)
print(np.array(res).sum()/(1.0*len(y_pred)))
print("loss ", loss_tr_1)
