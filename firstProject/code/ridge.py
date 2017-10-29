import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import Helpers.cleaner as cleaner
import clean
import numpy as np

y,x, ids = helper.load_csv_data('../data/train.csv', False)
#a,b, ids_s = helper.load_csv_data('../data/test.csv', False)

xtr,ytr = helper.load_clean_data('../data/x_train.npy', '../data/y_train.npy')
#xte,yte = helper.load_clean_data('../data/x_test.npy', '../data/y_test.npy')

x1, y1, xte, yte = tool.split_data(xtr,ytr,0.8)

_lambda = 0.0001

loss, w = tool.ridge_regression(ytr, xtr, _lambda)

y_pred_s = helper.predict_labels(w,xte)
y_pred = helper.changeYfromBinary(y_pred_s)

#helper.create_csv_submission(ids_s, y_pred, name="../data/submRomain.csv")

res = np.array([(1 if(y_pred_s[i] == yte[i]) else 0) for i in range(yte.shape[0])])
print(res.sum()/len(res)) # 0.735215293907
print(loss)               # 0.090209687261
