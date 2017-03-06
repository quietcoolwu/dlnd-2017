#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import tree
from sklearn.metrics import accuracy_score

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
acc_min_samples_split_2 = accuracy_score(labels_test,
                                         clf.predict(features_test))

clf_2 = tree.DecisionTreeClassifier(min_samples_split=50)
clf_2.fit(features_train, labels_train)
acc_min_samples_split_50 = accuracy_score(labels_test,
                                          clf_2.predict(features_test))


def submitAccuracies():
    return {
        "acc_min_samples_split_2": round(acc_min_samples_split_2, 3),
        "acc_min_samples_split_50": round(acc_min_samples_split_50, 3)
    }


if __name__ == '__main__':
    print(submitAccuracies())
