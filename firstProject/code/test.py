import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import Helpers.cleaner as cleaner
import clean
import numpy as np

y,x, ids_s = helper.load_csv_data('../data/train.csv', False)

xtr,ytr = helper.load_clean_data('../data/x_train.npy', '../data/y_train.npy')
xte,yte = helper.load_clean_data('../data/x_test.npy', '../data/y_test.npy')

w = tool.full_cross_validation(xtr,ytr)
print('W = ',w)
print('xTest = ',xte)
print('shape w:', w.shape)
print('shape x:', x.shape)
y_pred_s = helper.predict_labels(w,xte)
helper.create_csv_submission(ids_s, y_pred_s, name="../data/subm1.csv")
