# Machine Learning Project on Higgs boson

Using machine learning techniques viewed in class to classify particules

## Recreate the final submission results:

- In a shell go in the directory Project_ML
- Run `python3 run.py`
- The file `submission.csv` will be created in the Project_ML directory

## `cleaner.py`:
Contains all methods that we used or tried to use to clean the data
- `fillMissingValuesMeansWithY`
- `fillMissingValuesMeansWOY`
- `fillMissingValuesMediansWithY`
- `fillMissingValuesMediansWOY`

## `helpers.py`:
Contains the sub methods that we used in `cleaner.py` and `implementations.py` (including the given ones),
- `load_csv_data`
- `changeYtoBinary and changeYfromBinary`
- `sigmoid`
- `predict_labels`
- `create_csv_submission`
- `build_k_indices`
- `toDeg` and `addConstant`
- `build_poly`

## `implementations.py`:
Contains the 6 required implementations and some others that we tried:
- `least_squares_GD`
- `least_squares_SGD`
- `least_squares`
- `ridge_regression`
- `logistic_regression` + step function
- `reg_logistic_regression` + step function
- `logistic_regression_sgd` ?
- `reg_logistic_regression_sgd` ?
- `full_cross_validation` + validation for one fold

## Use one of the 6 implementations required:

- Duplicate the file `tool_test.py`
- Open the new file in your editor (for example under 'my_new_file.py')
- Replace the commented part 1 by the parameters that you will need
- Replace the commented part 2 by the function from implementation that you want to test
- Save this new file
- run `python3 my_new_file.py`
This will create the needed submission file (test_submission.csv)
