import matplotlib.pyplot as plt

import Helpers.helpers as helper
import Helpers.cleaner as cleaner
import numpy as np
import Helpers.implementations as imp

def toDeg(x,degree):
    res = x
    for i in range(2,degree +1):
        power = np.power(x,i)
        res = np.c_[res,power]
    return res

def addConstant(x):
    return np.c_[np.ones((x.shape[0],1)),x]

def main():
    x,y = helper.load_clean_data('../data/x_train.npy', '../data/y_train.npy')
    xpow = addConstant(toDeg(x, 3))
    x_tr,y_tr,x_te,y_te = imp.split_data(xpow,y,0.5,seed = 9)
    max_iters = 3
    gamma = .0004
    lambda_ = 0.1
    loss,w = imp.reg_logistic_regression(y=y_tr, tx=x_tr, lambda_=lambda_, initial_w=np.zeros((x_tr.shape[1], 1)), max_iters=max_iters, gamma=gamma)

    y_pred = helper.predict_labels(w,x_te)
    res = np.array([(1 if(y_pred[i] == y_te[i]) else 0) for i in range(y_te.shape[0])])
    print(res.sum()/len(res))

if __name__ == '__main__':
    main()
