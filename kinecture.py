#!/usr/bin/env python
# coding=utf-8
""" Helper functions for Kinecture experiment.
"""
__author__ = 'julenka'

import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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

    result = gen_my_speaker_features(result)
    result['speakerXYNorm'] = (result['mySpeakerX'] ** 2 + result['mySpeakerY'] ** 2) ** 0.5

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

class ClassroomParams:
    def __init__(self, classroom, roomX, roomY, Ltheta, Rtheta, dxL, dyL, dxR, dyR):
        self.classroom = classroom
        self.roomX = roomX
        self.roomY = roomY
        self.Ltheta = Ltheta
        self.Rtheta = Rtheta
        self.dxL = dxL
        self.dyL = dyL
        self.dxR = dxR
        self.dyR = dyR

        self.Rx = roomX - dxR
        self.Ry = roomY - dyR

        self.Lx = dxL
        self.Ly = roomY - dyL

# classroom_param_map = { 9: ClassroomParams(9, 226, 356, 317, 220, 7, 10, 22, 12),
#                        10: ClassroomParams(10, 456, 317, 313, 221, 12, 6, 13, 8),
#                        11: ClassroomParams(11, 390, 334, 312, 222, 9, 11, 6, 48)}
# I had to modify the classroom params to make the y position of the kinects (dyR, dyL) match up, otherwise
# speakerX and speakerY positions could be too large because the lines wouldn't intersect.
classroom_param_map = { 9: ClassroomParams(9, 226, 356, 317, 220,  7, 10, 22, 10),
                       10: ClassroomParams(10, 456, 317, 313, 221, 12, 6, 13, 6),
                       11: ClassroomParams(11, 390, 334, 312, 222, 9, 11, 6, 11)}


def gen_my_speaker_features(data):
    result = data.copy()
    for row_num, row in result.iterrows():
        my_speaker_x, my_speaker_y = get_speakerxy_from_angles(row.session,
                                                               row['angleLeft'],
                                                               row['angleRight'])
        result.loc[row_num, 'mySpeakerX'] = my_speaker_x
        result.loc[row_num, 'mySpeakerY'] = my_speaker_y

    return result

def get_speakerxy_from_angles(classroom_number, left_angle, right_angle):
    classroom_params = classroom_param_map[classroom_number]

    left_angle_2 = classroom_params.Ltheta + left_angle
    right_angle_2 = classroom_params.Rtheta + right_angle

    # Clamp these bad boys
    left_angle_2 = max(271, min(359, left_angle_2))
    right_angle_2 = max(181, min(269, right_angle_2))

    # y = mx + b
    left_m = math.tan(math.radians(left_angle_2))
    left_b = classroom_params.Ly - left_m * classroom_params.Lx

    right_m = math.tan(math.radians(right_angle_2))
    right_b = classroom_params.Ry - right_m * classroom_params.Rx

    a = left_m
    b = left_b
    c = right_m
    d = right_b

    x = (b - d) / (c - a)
    y = a * (b - d) / (c - a) + b

    x = max(0, min(classroom_params.roomX, x)) / classroom_params.roomX
    y = max(0, min(classroom_params.roomY, y)) / classroom_params.roomY

    return x, y

def plot_distributions(data, columns, num_subplot_cols, figsize=(20, 50)):
    num_subplot_rows = math.ceil(len(columns) / num_subplot_cols)
    fig, axes = plt.subplots(num_subplot_rows, num_subplot_cols, squeeze=False)
    for i, column in enumerate(columns):
        # Tutorial at http://stanford.edu/~mwaskom/software/seaborn/tutorial/distributions.html#distribution-tutorial
        # Plot a histogram
        sns.distplot(data[column], ax=axes[i/num_subplot_cols][i%num_subplot_cols], kde=False)

    # Make the figure big and look pretty
    fig.set_size_inches(figsize, forward=True)
    fig.tight_layout()

def plot_timeseries(data, columns, num_subplot_cols, title=None, figsize=(20, 50)):
    num_subplot_rows = math.ceil(len(columns) / num_subplot_cols)
    fig, axes = plt.subplots(num_subplot_rows, num_subplot_cols, squeeze=False)
    if title:
        fig.suptitle(title, fontsize=18, y=1.08)
    for i, column in enumerate(columns):
        # Tutorial at http://stanford.edu/~mwaskom/software/seaborn/tutorial/distributions.html#distribution-tutorial
        # Plot a histogram
        ax = axes[i/num_subplot_cols][i % num_subplot_cols]
        ax.plot(data[column])
        ax.set_xlabel(column)

    # Make the figure big and look pretty
    fig.set_size_inches(figsize, forward=True)
    fig.tight_layout()

