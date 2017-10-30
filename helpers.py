# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np



def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)."""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1 #modified this to fit the course formula

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def changeYtoBinary(y):
    """Map all -1 in Y to 0 and keep the other at 1."""
    res = np.array(y)
    res[np.where(y == -1)] = 0
    return res

def changeYfromBinary(y):
    """Map all 0 in Y to -1 and keep the other at 1."""
    res = np.array(y)
    res[np.where(y == 0)] = -1
    return res

def sigmoid(t):
    """Apply sigmoid function on t."""
    exp_inv = np.exp(-1.0*t)
    return np.divide(1,1+exp_inv)

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix."""
    y_pred = np.dot(data, weights)
    #y_pred = sigmoid(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = 0 #modified this to fit the formula
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

#for cross validation
def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def toDeg(x,degree):
    """Raise x to the given degree"""
    res = x
    for i in range(2,degree +1):
        power = np.power(x,i)
        res = np.c_[res,power]
    return res

def addConstant(x):
    """Add a constant term in the features"""
    return np.c_[np.ones((x.shape[0],1)),x]

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    x2 = toDeg(x,degree)
    x3 = addConstant(x2)
    return x3

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx,w)
    return (-1.0*(np.dot(tx.T,e)))/N

def batch_iter(y, tx, batch_size, seed, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

#loss functions
def compute_mse(y, tx, w):
    """Compute the loss with the mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_mae(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.dot(tx,w)
    e = np.absolute(e)
    N = y.shape[0]
    return e.sum()/(N*1.0)

#used to have train and test set
def split_data(x, y, ratio, seed=1):
    """Split the data x and y with a given ratio.

    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    np.random.seed(seed)
    ind = list(range(y.shape[0]))
    np.random.shuffle(ind)
    x_shuffled = x[ind]
    y_shuffled = y[ind]
    limit = int(np.floor(y.shape[0]*ratio))
    return x_shuffled[:limit],y_shuffled[:limit],x_shuffled[limit:],y_shuffled[limit:]

# regular regression
def calculate_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    fx_list = np.dot(tx,w)
    fx_trans_1 = np.logaddexp(0,fx_list)
    fx_trans_2 = np.multiply(y,fx_list)
    res = fx_trans_1 - fx_trans_2
    return res.sum()


def calculate_gradient(y, tx, w):
    """Compute the gradient of loss."""
    fx_sigma = sigmoid(np.dot(tx,w))
    mul = fx_sigma - y
    return np.dot(tx.T,mul)


#regularized logistic regression
def reg_calculate_loss(y, tx, lambda_, w):
    """Compute the cost by negative log likelihood."""
    return calculate_loss(y,tx,w) + (lambda_/2.0)*np.dot(w.T,w)

def reg_calculate_gradient(y, tx, lambda_, w):
    """Compute the gradient of loss."""
    return calculate_gradient(y,tx,w) + (lambda_)*w
