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

batch_size = 1
gamma = 0.085
lambda_ = 0.0001
max_iters = 500
initial_w = np.zeros(xtr.shape[1])
loss_1, w_1 = tool.least_squares_GD(ytr, xtr, initial_w, max_iters, gamma)
loss_2, w_2 = tool.least_squares_SGD(ytr, xtr, initial_w, batch_size, max_iters, gamma)
loss_3, w_3 = tool.least_squares(ytr, xtr)
loss_4, w_4 = tool.ridge_regression(ytr, xtr, lambda_)
loss_5, w_5 = tool.logistic_regression(ytr, xtr, initial_w, max_iters, gamma)
loss_6, w_6 = tool.reg_logistic_regression_sgd(ytr, xtr, lambda_, initial_w, batch_size, max_iters, gamma)
print(loss_1)
print(loss_2)
print(loss_3)
print(loss_4)
print(loss_5)
print(loss_6)
sum_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

y_pred_1 = helper.predict_labels(w_1,xte) # 0.731482618843
y_succ_1 = 0.731482618843
y_pred_2 = helper.predict_labels(w_2,xte) # 0.62458264006
y_succ_2 = 0.62458264006
y_pred_3 = helper.predict_labels(w_4,xte) # 0.732543466851
y_succ_3 = 0.732543466851
y_pred_4 = helper.predict_labels(w_4,xte) # 0.732543466851
y_succ_4 = 0.732543466851
y_pred_5 = helper.predict_labels(w_5,xte) # 0.661008810622
y_succ_5 = 0.661008810622
y_pred_6 = helper.predict_labels(w_6,xte) # 0.62458264006
y_succ_6 = 0.62458264006
sum_succ = y_succ_1 + y_succ_2 + y_succ_3 + y_succ_4 + y_succ_5 + y_succ_6

y_pred_avg = np.zeros(y_pred_1.shape[0])
for i in range(y_pred_avg.shape[0]):
#    y_pred_avg[i] = 0 if (y_pred_1[i] + y_pred_2[i] + y_pred_3[i] + y_pred_4[i] + y_pred_5[i] + y_pred_6[i]) < 3  else 1
    y_pred_avg[i] = 0 if (y_pred_1[i]*y_succ_1/sum_succ + y_pred_2[i]*y_succ_2/sum_succ + y_pred_3[i]*y_succ_3/sum_succ + y_pred_4[i]*y_succ_4/sum_succ + y_pred_5[i]*y_succ_5/sum_succ + y_pred_6[i]*y_succ_6/sum_succ) < 0.5  else 1

# y_pred = helper.changeYfromBinary(y_pred_avg)

#helper.create_csv_submission(ids_s, y_pred, name="../data/submRomain.csv")

res = np.array([(1 if(y_pred_avg[i] == yte[i]) else 0) for i in range(yte.shape[0])])
print(res.sum()/len(res)) # 0.734344291585
