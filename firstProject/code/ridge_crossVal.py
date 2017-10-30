import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import Helpers.cleaner as cleaner
import clean
import numpy as np

y,x, ids = helper.load_csv_data('../data/train.csv', False)
a,b, ids_s = helper.load_csv_data('../data/test.csv', False)

xtr,ytr = helper.load_clean_data('../data/x_train.npy', '../data/y_train.npy')
xte,yte = helper.load_clean_data('../data/x_test.npy', '../data/y_test.npy')
#x1, y1, xte, yte = tool.split_data(xtr,ytr,0.8)

x_s = helper.build_poly(xte,5)

w = tool.full_cross_validation(xtr,ytr)

y_pred_s = helper.predict_labels(w,x_s)
y_pred = helper.changeYfromBinary(y_pred_s)

helper.create_csv_submission(ids_s, y_pred, name="../data/submRomain.csv")

#res = np.array([(1 if(y_pred_s[i] == yte[i]) else 0) for i in range(yte.shape[0])])
#print(res.sum()/len(res))