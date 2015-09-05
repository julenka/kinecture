#!/usr/bin/env python
# coding=utf-8
""" Helper functions for Kinecture experiment.
"""
__author__ = 'julenka'

import math
import numpy as np
import sys
import sklearn.metrics

def remove_columns(data, columns):
    for column_name in columns:
        if column_name in data:
            del data[column_name]
        else:
            print("column {} not found (may already be removed)".format(column_name))

def clean_data(data):
    result = data.copy()
    remove_columns(result, ['id', 'order', 'SISP', 'SNS', 'audTime', 'heuristic', 'timestampUNIX'])
    return result

def gen_features(data):
    """ Given pandas dataframe with raw data, remove unused data and add features

    :param data:
    :return: features array
    """
    result = data.copy()

    # save the labels, then remove them from the dataframe (we'll add it back at the end)
    labels = result["Truth"]

    result['speakerXYNorm'] = (result['speakerX'] ** 2 + result['speakerY'] ** 2) ** 0.5

    remove_columns(result, ['Truth'])

    # add new features
    result['logLoudnessLeft'] = result['loudnessLeft'].map(lambda x: math.log(x + 1e-5))
    result['logLoudnessRight'] = result['loudnessRight'].map(lambda x: math.log(x + 1e-5))
    result['averageLogLoudness'] = (result['logLoudnessLeft'] + result['logLoudnessRight']) / 2
    result['silenceLeftOrRight'] = result['silenceLeft'] | result['silenceRight']

    result['Truth'] = labels
    return result


def remap_labels(label_map, labels):
    result = []
    for i, label in enumerate(labels):
        if label in label_map:
            result.append(label_map[label])
        else:
            result.append(label)
    return result

def categorical_to_index(categorical_column):
    categorical_values = set(categorical_column)
    result = {cv: i for i, cv in enumerate(categorical_values)}
    return result

class LabeledDataForSklearn:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def append(self, other):
        X = np.append(self.X, other.X, axis=0)
        y = np.append(self.y, other.y, axis=0)
        return LabeledDataForSklearn(X, y)

def convert_features_for_sklearn(data):
    """ Splits data into features and classlabel so sklearn can use it

    Assumes no categorical variables
    Assumes class label is the last value in the class
    :param data:
    :return:
    """

    labels_to_index = categorical_to_index(data.iloc[:, -1])
    X = data.iloc[:, :-1].astype(float)
    X = sklearn.preprocessing.normalize(X, axis=0)
    y = [labels_to_index[label] for label in data.iloc[:, -1]]

    return LabeledDataForSklearn(np.array(X), np.array(y))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrices"""
    columnwidth = max([len(x) for x in labels]+[5]) # 7 is value length
    empty_cell = " " * columnwidth
    # Print header
    sys.stdout.write("    " + empty_cell)
    for label in labels:
        sys.stdout.write("%{0}s".format(columnwidth) % label)
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        sys.stdout.write("    %{0}s".format(columnwidth) % label1)
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            sys.stdout.write(cell)
        print()
    print()


def report_accuracy(predictions, actual, header=None):
    if header:
        print(header)
    cm = sklearn.metrics.confusion_matrix(actual, predictions)
    class_names = ["S", "SI", "TA"]
    print_cm(cm, class_names)
    print(sklearn.metrics.classification_report(actual, predictions, target_names=class_names))
