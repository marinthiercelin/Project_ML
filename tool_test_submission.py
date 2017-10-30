import helpers as hlp
import implementations as imp
import cleaner as cln
import numpy as np


def main():
    """Use this code to test the different implementations in implementations.py"""

    x, y, x_te, y_te, ids_te = cln.final_clean_data()

    # Part 1: Select and choose the parameters that you need

    # initial_w = np.ones(x.shape[1])
    # max_iters = 500
    # gamma = .0008
    # lambda_ = 0

    # Part 2: Select the implementation that you want to test
    #loss, w = imp.least_squares(y, x)

    y_pred_s = hlp.predict_labels(w,x_te)
    y_pred = hlp.changeYfromBinary(y_pred_s)

    hlp.create_csv_submission(ids_te, y_pred, name="test_submission.csv")

if __name__ == '__main__':
    main()
