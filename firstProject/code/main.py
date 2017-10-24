import Helpers.helpers as helper
import Helpers.implementations as tool
import Visualization.visu as visu
import cleaner
import numpy as np

def learn(y, x):
    #parameters
    lambda_ = 0.7
    gamma = 0.0000000001
    max_iters = 1000

    initial_w = np.array(x.shape[1]*[0])
    #tool.reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    return tool.reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma)

def learnParameters(y, x, initial_w, max_iters, lambda_, gamma):
    #tool.reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    return tool.reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma)

def makeSubmission(w, ids, x , submissionPath):
    y_pred = helper.predict_labels(w, x)
    y_pred = helper.changeYfromBinary(y_pred)
    helper.create_csv_submission(ids, y_pred, name=submissionPath)

def treatData(x,withY = True,y = []):
    xCl = x
    YCl = y
    if(withY):
        xCl = cleaner.fillMissingValuesMedianWithY(x,y)
        yCl = helper.changeYtoBinary(y)
    else:
        xCl = cleaner.fillMissingValuesMedianWOY(x)
    xCl = cleaner.outliersToMedian(x)
    xCl = cleaner.normalize_input(x)
    xCl = cleaner.addConstant(x)
    if(withY):
        return yCl, xCl
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

def main2(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv'):
    #parameters
    ratio = 0.8

    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl, xCl = treatData(x, True, y)
    x_tr, y_tr, x_te, y_te = tool.split_data(xCl, yCl, ratio, seed=3)
    loss, w = learn(yCl, xCl)
    y_pred = helper.predict_labels(w,x_te)
    res = [(1 if(y_p == y_te[i]) else 0) for i,y_p in enumerate(y_pred)]
    print("w is ", w)
    print(np.array(res).sum()/(1.0*len(y_pred)))
    print("loss ", loss)
    print(y_pred.sum())

def main3(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv', submissionPath = '../data/subm1.csv'):
    path = '../data/Figures/2Dtreated/'
    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl,xCl = treatData(x, True, y)
    visu.produce2DFiguresWOMissing(x, y, path)

def main4(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv', submissionPath = '../data/subm1.csv'):
    #parameters
    ratio = 0.8

    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl, xCl = treatData(x, True, y)
    visu.produce2DWithBoundary(yCl,xCl,"../data/Figures/2Dij/")

def main5(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv', submissionPath = '../data/subm1.csv'):
    #parameters
    ratio = 0.8

    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl, xCl = treatData(x, True, y)
    i = 18
    j = 11
    tx = np.c_[xCl[:,i], xCl[:, j]]
    visu.logistic_regression_gradient_descent_demo(yCl,tx,i,j, "../data/Figures/2Dij2/")

def main6(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv'):
    #parameters
    ratio = 0.8
    max_i = [10000]
    lambdas= [0.8,0.85]
    gammas = [0.01]
    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl, xCl = treatData(x, True, y)
    x_tr, y_tr, x_te, y_te = tool.split_data(xCl, yCl, ratio, seed=3)
    for max_iters in max_i:
        for lambda_ in lambdas:
            for gamma in gammas:
                loss, w = learnParameters(yCl, xCl, max_iters, lambda_, gamma)
                y_pred = helper.predict_labels(w,x_te)
                res = [(1 if(y_p == y_te[i]) else 0) for i,y_p in enumerate(y_pred)]
                print("max_iters = ", max_iters,"lamda = ", lambda_, "gamma = ", gamma)
                print(np.array(res).sum()/(1.0*len(y_pred)))

def main7(subSample = False, trainPath = '../data/train.csv', testPath = '../data/test.csv', submissionPath = '../data/subm1.csv'):
    #parameters
    ratio = 0.8
    lambda_ = 0.7
    gamma = 0.0000000001
    max_iters = 1000
    loops = 100
    print("l = ", lambda_, " g = ", gamma)
    y, x, ids = helper.load_csv_data(trainPath, subSample)
    yCl, xCl = treatData(x, True, y)
    x_tr, y_tr, x_te, y_te = tool.split_data(xCl, yCl, ratio, seed=3)
    w =  np.array(x_tr.shape[1]*[0])
    for m in range(loops):
        print("loop ",m)
        loss, w = learnParameters(yCl, xCl, w, max_iters, lambda_, gamma)
        y_pred = helper.predict_labels(w,x_te)
        res = [(1 if(y_p == y_te[i]) else 0) for i,y_p in enumerate(y_pred)]
        print("w is ", w)
        print(np.array(res).sum()/(1.0*len(y_pred)))
        print("loss ", loss)
        print(y_pred.sum())
    y_s, x_s, ids_s = helper.load_csv_data(testPath, False)
    xCl_s = treatData(x_s, False)
    makeSubmission(w, ids_s, xCl_s, submissionPath)


main7(subSample=False)
