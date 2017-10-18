import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import cleaner
import numpy as np

y,x, ids = helper.load_csv_data('../data/train.csv', False)

path = '../data/Figures/2D/'
xCl = cleaner.cleanFeatures(x)
print(xCl)
print(y)
xCl = cleaner.fillMissingValuesWithY(xCl,y)
xCl = cleaner.addConstant(xCl)
#visu.produce2DFigures(xCl,y,folder_path = path, save = True)



#parameters
ratio = 0.8
lambda_ = 0.8
gamma = 0.8

x_tr,y_tr,x_te,y_te = tool.split_data(xCl, y, ratio, seed=3)

initial_w = np.array(x_tr.shape[1]*[0])
loss_tr_1, w = tool.reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, 50, gamma)

y_pred = helper.predict_labels(w,x_te)

res = [(1 if(y_p == y_te[i]) else 0) for i,y_p in enumerate(y_pred)]

print(np.array(res).sum()/(1.0*len(y_pred)))
"""

t = [[1.,2.,-999.],[2.,-999.,1.],[3.,1.,-999.],[2.,-999.,1.],[-999.,3.,2.]]
y = [1.,1.,0.,0.,0.]
y = np.array(y)
t = np.array(t)

print(t)
print(y)
tBis = cleaner.fillMissingValuesWithY(t,y)
print(t)
print(tBis)
tTer = cleaner.fillMissingValuesWOY(t)
print(t)
print(tTer)
"""
