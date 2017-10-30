import Helpers.helpers as hlp
import Helpers.cleaner as cl
import Helpers.implementations as imp
import numpy as np

def main():
    x,y, x_test, y_test, ids_te = cl.final_clean_data()

    w = imp.full_cross_validation(x,y)

    x_te = hlp.build_poly(x_test, 5)
    y_pred = hlp.predict_labels(w,x_te)

    y_pred = hlp.changeYfromBinary(y_pred)
    hlp.create_csv_submission(ids_te,y_pred,'../data/submRomain.csv')

if __name__ == '__main__':
    main()
