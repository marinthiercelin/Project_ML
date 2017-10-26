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

def save_clean_data(clean_x, clean_y, path_of_x, path_of_y):
    """saves the clean data to the file system, pass the last '/' as argument"""
    np.save(path_of_x, clean_x)
    np.save(path_of_y, clean_y)

def load_clean_data(path_of_x, path_of_y):
    """loads the clean data as x, y, pass the last '/' as argument"""
    clean_x = np.load(path_of_x)
    clean_y = np.load(path_of_y)

    return clean_x, clean_y

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
    """apply sigmoid function on t."""
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
