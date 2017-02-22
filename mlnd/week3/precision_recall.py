# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
X = pd.read_csv('../titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)

clf2 = GaussianNB()
clf2.fit(X_train, y_train)

tree_recall, tree_precision = recall(y_test, clf1.predict(X_test)), precision(
    y_test, clf1.predict(X_test))
nb_recall, nb_precision = recall(y_test, clf2.predict(X_test)), precision(
    y_test, clf2.predict(X_test))

print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(
    tree_recall, tree_precision)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(nb_recall,
                                                               nb_precision)

results = {
    "Naive Bayes Recall": nb_recall,
    "Naive Bayes Precision": nb_precision,
    "Decision Tree Recall": tree_recall,
    "Decision Tree Precision": tree_precision
}
