# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd
# Load the dataset
from sklearn import datasets, cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

X = pd.read_csv('../titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

# The decision tree classifier
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)

# The naive Bayes classifier
clf2 = GaussianNB()
clf2.fit(X_train, y_train)

cf_tree = confusion_matrix(y_test, clf1.predict(X_test))
cf_gaussian_nb = confusion_matrix(y_test, clf2.predict(X_test))

print "Confusion matrix for this Decision Tree:\n", cf_tree
print "GaussianNB confusion matrix:\n", cf_gaussian_nb

confusions = {"Naive Bayes": cf_gaussian_nb, "Decision Tree": cf_tree}
