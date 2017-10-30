import helpers as hlp
import implementations as imp
import cleaner as cln
import numpy as np


def main():
    """Use this code to test the different implementations in implementations.py"""

    x, y, x_te, y_te, ids_te = cln.final_clean_data()

    xtr, ytr, xte, yte = imp.split_data(x, y,0.8)

    """Change parameters here:"""
    initial_w = np.ones(x.shape[1])
    max_iters = 500
    gamma = .0008
    lambda_ = 0


    """Change name of function here:"""
    loss, w = imp.least_squares(ytr, xtr)

    y_pred_s = hlp.predict_labels(w,xte)
    y_pred = hlp.changeYfromBinary(y_pred_s)

    #helper.create_csv_submission(ids_s, y_pred, name="../data/submRomain.csv")

    res = np.array([(1 if(y_pred_s[i] == yte[i]) else 0) for i in range(yte.shape[0])])
    print("Accuracy is: ", res.sum()/len(res))
    print("Loss is: ", loss)

if __name__ == '__main__':
    main()
