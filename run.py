import helpers as hlp
import cleaner as cl
import implementations as imp
import numpy as np

def main():
    """Makes the predictions on the test data.

    First, we load the train and test data, clean them.
    Second, we train the model using cross validation.
    Third, we prepare the test sample matrix.

    Lastly, we predict the labels for the test data and create a csv submission"""
    x,y, x_test, y_test, ids_te = cl.final_clean_data()

    loss, w = imp.full_cross_validation(x,y)

    x_te = hlp.build_poly(x_test, 5)
    y_pred = hlp.predict_labels(w,x_te)

    y_pred = hlp.changeYfromBinary(y_pred)
    hlp.create_csv_submission(ids_te,y_pred,'submission.csv')

if __name__ == '__main__':
    main()
