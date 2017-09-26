# Multiple Linear-Regression
Implementation of a simple linear regression using least square method on multiple predictors and single response variable. 
Our implementation is compared with the standard implementation of sklearn's linear regression based on the computation of RMSE.

***Write up/ Presentation slides: Please find the link to our presentation slides below:***

https://docs.google.com/presentation/d/1YRGwG3gX3PY3yZn0_JbwysoVO5zntbnlNGD8oB5XIFE/edit?usp=sharing

**Implementation steps:**

1) Dataset: The dataset for our implementation is chosen from the http://archive.ics.uci.edu/ml/machine-learning-databases/. Thus, chosen dataset
is particularly suitable for our implementation (Multiple Linear Regression). The link is provided below:

data: http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data

description: http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names

Note: The first column in the dataset is ignored as it does not contribute to the prediction. In the dataset, we consider that the last  column, 'Rings' is our response variable (Y) and the other columns are the predictor values (Say, x1, x2, x3..). Hence,
we are predicting the Rings (Y) based on the various parameters (x1,x2,x3...).

CSV files: For the sake of easy processing, we divide the response variable and predictor varibles into two separate files, response.csv
and predictors.csv respectively. 

2) Our Implementation: Our implementation can be found in the files, 'Multiple_linear_regression.py' and 'Multiple_linear_regression_best_model.py'

Multiple_linear_regression.py : This implementation uses cross_validatiion.KFold (folds=5) method on the entire dataset to calculate average RMSE value. It also contains an independent implementation using test_train_split() method. The performance metrics, RMSE is compared for both.

Multiple_linear_regression_best_model.py: The dataset is divided into three parts, Train, validation and test sets. Our model is implemented on the train set and validation set and the best model is compared with the test_set and finally, RMSE is calculated.

3) Sklearn implementation: Now that we already have our implementation handy, we use the sklearn library's linear regression model with cross validation (using Kflold, no of folds is 5) to compute the RMSE. Thus obtained model is compared with our implementation.

Technical Details:

Language used: Python
Operating system: Linux (Ubuntu).

Commands of Ubuntu:
To run a python code, "python <<file_name.py>>"


