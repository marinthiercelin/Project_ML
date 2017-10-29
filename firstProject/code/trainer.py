import Helpers.helpers as helper
import Helpers.cleaner as cleaner
import numpy as np
import Helpers.implementations as imp

def train_model_ridge_cross(x, y, lambdas=[-0.00001], ratio=.8, seed=1):
    """Trains a model using ridge_regression and cross_validation.

    Selects the weights which have the best accuracy locally"""
    accuracies = []
    ws = []
    rmse_trs = []
    rmse_tes = []

    x_tr,y_tr,x_te,y_te = imp.split_data(x,y,ratio,seed)
    w_all, rmse_tr, rmse_te = imp.full_cross_validation(x_tr,y_tr, lambdas=lambdas)

    for w in w_all:
        y_pred = helper.predict_labels(w,x_te)
        res = np.array([(1 if(y_pred[i] == y_te[i]) else 0) for i in range(y_te.shape[0])])
        accuracies.append(res.sum()/len(res))


    i = np.argmax(accuracies)
    return w_all[i], accuracies[i], rmse_tr[i], rmse_te[i]

def train_all_models(xs, ys, lambdas=[0.001, .01, -0.01], parts=[0,1,2,3]):
    """Trains all the models.

    Takes as input a list of the x and y
    and Returns 4 lists: the weight for each model, their corresponding accuracy,
    and the rmses (training and testing)
    """
    xpow = [helper.addConstant(helper.toDeg(xs[p], 2)) for p in parts]
    ws = []
    accuracies = []
    rmse_trs = []
    rmse_tes = []
    for p in parts:
        w, accuracy, rmse_tr, rmse_te = train_model_ridge_cross(xpow[p], ys[p], lambdas=lambdas)
        ws.append(w)
        accuracies.append(accuracy)
        rmse_trs.append(rmse_tr)
        rmse_tes.append(rmse_te)

    return ws, accuracies, rmse_trs, rmse_tes
