import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import cleaner
import numpy as np

def learn(y, x):
    #parameters
    lambda_ = 0.1
    gamma = 0.1
    max_iters = 5

    initial_w = np.array(x.shape[1]*[0])
    #tool.reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    return tool.logistic_regression(y, x, initial_w, max_iters, gamma)

def makeSubmission(w, ids, x , submissionPath):
    y_pred = helper.predict_labels(w, x)
    y_pred = helper.changeYfromBinary(y_pred)
    helper.create_csv_submission(ids, y_pred, name=submissionPath)

def treatData(x,withY = True,y = []):
    xCl = x
    if(withY):
        xCl = cleaner.fillMissingValuesMedianWithY(x,y)
    else:
        xCl = cleaner.fillMissingValuesMedianWOY(x)
        yCl = helper.changeYtoBinary(y)
    xCl = cleaner.outliersToMedian(x)
    xCl = cleaner.normalize_input(x)
    xCl = cleaner.addConstant(x)
    if(withY):
        return y, xCl
    else:
        return xCl

def main1(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv', submissionPath = '../data/subm1.csv'):
    #parameters
    ratio = 0.8

    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl,xCl = treatData(x, True, y)
    loss,w = learn(yCl, xCl)
    y_s, x_s, ids_s = helper.load_csv_data(testPath, False)
    xCl_s = treatData(x_s, False)
    makeSubmission(w, ids_s, xCl_s, submissionPath)

def main2(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv', submissionPath = '../data/subm1.csv'):
    #parameters
    ratio = 0.2

    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl, xCl = treatData(x, True, y)
    x_tr, y_tr, x_te, y_te = tool.split_data(xCl, yCl, ratio, seed=3)
    loss, w = learn(yCl, xCl)
    y_pred = helper.predict_labels(w,x_te)
    res = [(1 if(y_p == y_te[i]) else 0) for i,y_p in enumerate(y_pred)]
    print("w is ", w)
    print(np.array(res).sum()/(1.0*len(y_pred)))
    print("loss ", loss)

main2(subSample=True)
