import numpy as np
import helpers as hlp
from helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_mse(y,tx,w)
        w = w - gamma*gradient
    loss = compute_mse(y,tx,w)
    return loss, w


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, seed=1):
    """Stochastic gradient descent algorithm using mse."""
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,batch_size, seed):
            gradient = compute_gradient(batch_y,batch_tx,w)
            loss = compute_mse(batch_y,batch_tx,w)
            w = w - gamma*gradient
    loss = compute_mse(y, tx, w)
    return loss, w


#least squares with mse
def least_squares(y, tx):
    """Calculate the least squares solution."""
    xt = tx.T
    gram = np.dot(xt,tx)
    ft = np.dot(xt,y)
    w = np.linalg.lstsq(gram, ft)[0]
    loss = compute_mse(y,tx,w)
    return loss, w


#ridge regression
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    xt = tx.T
    gram = np.dot(xt,tx)
    gram += (2.0*y.shape[0]*lambda_)*np.identity(gram.shape[0])
    ft = np.dot(xt,y)
    w = np.linalg.lstsq(gram, ft)[0]
    loss = compute_mse(y, tx, w)
    return loss, w



def logistic_regression_step(y, tx, w, gamma):
    """Execute on step of the logistic regression.

    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    grad = calculate_gradient(y,tx,w)
    print("grad",grad)
    w = w - gamma*grad
    return w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        w = logistic_regression_step(y,tx,w,gamma)
        print("w",w)
    loss = calculate_loss(y,tx,w)
    return loss, w

def logistic_regression_sgd(y, tx, initial_w, batch_size, max_iters, gamma, seed = 1):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,batch_size, seed):
            w = logistic_regression_step(batch_y,batch_tx,w,gamma)
    loss = calculate_loss(y, tx, w)
    return loss, w


def reg_logistic_regression_step(y, tx, lambda_, w, gamma):
    """Execute one step of the regularized logistic regression.

    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    grad = reg_calculate_gradient(y,tx, lambda_, w)
    w = w - gamma*grad
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        w = reg_logistic_regression_step(y,tx,lambda_,w,gamma)
        """if(n_iter % (max_iters/10) == 0):
            loss = reg_calculate_loss(y, tx, lambda_, w)
            print(n_iter," loss ",loss)
            print(n_iter," w ", w)"""

    loss = reg_calculate_loss(y, tx, lambda_, w)
    return loss, w

def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, seed = 1):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,batch_size, seed):
            w = reg_logistic_regression_step(batch_y,batch_tx,lambda_,w,gamma)
    loss = reg_calculate_loss(y, tx, lambda_, w)
    return loss, w

def one_cross_validation(y, x, k_indices, k, lambda_):
    """Return the loss of ridge regression."""
    # ***************************************************
    # get k'th subgroup in test, others in train
    # ***************************************************
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    all_ind = np.array(list(range(x.shape[0])))
    rest_ind = np.setdiff1d(all_ind,k_indices[k])
    x_train = x[rest_ind]
    y_train = y[rest_ind]
    # ***************************************************
    # ridge regression
    # ***************************************************
    loss,w = ridge_regression(y_train,x_train,lambda_)
    # ***************************************************
    # calculate the loss for train and test data
    # ***************************************************
    loss_tr = compute_mse(y_train,x_train,w)
    loss_te = compute_mse(y_test,x_test,w)
    return w,loss_tr, loss_te

def full_cross_validation(x,y):
    """Tests the cross validation on different values of k_folds."""
    degree = 5
    k_fold = 4
    seed = 7
    lambda_ = 0 # -0.000000000001
    # split data in k fold
    x_pow = hlp.build_poly(x,degree)
    k_indices = hlp.build_k_indices(y, k_fold, seed)
    sum_te = 0
    sum_tr = 0
    sum_w = np.array(x_pow.shape[1]*[0.0])
    for k in range(k_fold):
        w,mse_tr,mse_te = one_cross_validation(y,x_pow,k_indices,k, lambda_)
        sum_w += w
        sum_tr += np.sqrt(2*mse_tr)
        sum_te += np.sqrt(2*mse_te)
    mean_w = sum_w/(1.0*k_fold)
    mean_error_tr = sum_tr/(1.0*k_fold)
    mean_error_te = sum_te/(1.0*k_fold)
    return mean_w
