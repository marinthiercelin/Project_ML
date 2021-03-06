# Machine Learning Project on Higgs boson

Using machine learning techniques viewed in class to classify particules.
The documentation to any method can be found in the corresponding file accordingly to PEP 257.
You can also access it though a python2/3 shell after importing it using `help(file.function)`.

## Functionalities

### Recreate the final submission results:

- In a shell go in the directory Project_ML
- Run `python3 run.py`
- The file `submission.csv` will be created in the Project_ML directory

### Change the data sets
You have the choice between 2 alternatives to use another set of data to test our methods:
- Save them under train.csv and test.csv in the root directory Project_ML
- Open cleaner.py with your editor, change 'train.csv' and 'test.csv' to the names of your 2 data sets

### Use one of the 6 implementations required:
This will create the needed submission file (test_submission.csv) given by a specified implementation
- Duplicate the file `tool_test_submission.py`
- Open the new file in your editor (for example under 'my_new_file.py')
- Replace the commented part 1 by the parameters that you will need
- Replace the commented part 2 by the function from implementation that you want to test
- Save this new file
- run `python3 my_new_file.py`

### Test an implementation locally:
This will give you a estimated accuracy and loss of a specified implementation by splitting the training data
- Duplicate the file `tool_test_local.py`
- Open the new file in your editor (for example under 'my_new_file.py')
- Replace the commented part 1 by the parameters that you will need
- Replace the commented part 2 by the function from implementation that you want to test
- Save this new file
- run `python3 my_new_file.py`

## Methods :

### `implementations.py`:

Contains the 6 required implementations and some others that we tried:
- `least_squares_GD`
- `least_squares_SGD`
- `least_squares`
- `ridge_regression`
- `logistic_regression` and his step function
- `reg_logistic_regression` and his step function
- `logistic_regression_sgd`
- `reg_logistic_regression_sgd` ?
- `full_cross_validation` and the validation for one fold

### `cleaner.py`:

Contains all methods that we used or tried to use to clean the data:
- `fillMissingValuesMeansWithY`
- `fillMissingValuesMeansWOY`
- `fillMissingValuesMediansWithY`
- `fillMissingValuesMediansWOY`
- `normalize_input`
- `select_bad_samples`
- `filter_bad_samples`
- `select_bad_features`
- `outliersToMedian`
- `remove_unused_features`
- `computeCorrelation`

### `helpers.py`:

Contains the sub methods that we used in `cleaner.py` and `implementations.py` (including the given ones):
- `load_csv_data`
- `changeYtoBinary and changeYfromBinary`
- `sigmoid`
- `predict_labels`
- `create_csv_submission`
- `build_k_indices`
- `toDeg` and `addConstant`
- `build_poly`
- `compute_gradient`
- `batch_iter`
- `compute_mse`
- `compute_mae`
- `split_data`
- `calculate_loss`
- `calculate_gradient`
- `reg_calculate_loss`
- `reg_calculate_gradient`
