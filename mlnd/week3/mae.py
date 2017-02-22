import numpy as np
import pandas as pd
# Load the dataset
from sklearn import cross_validation
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
# from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target


# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)


reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)

reg2 = LinearRegression()
reg2.fit(X_train, y_train)

lr_mae = mae(y_test, reg2.predict(X_test))
tree_mae = mae(y_test, reg1.predict(X_test))

print "Linear regression mean absolute error: {:.2f}".format(lr_mae)
print "Decision Tree mean absolute error: {:.2f}".format(tree_mae)

results = {
 "Linear Regression": lr_mae,
 "Decision Tree": tree_mae
}
