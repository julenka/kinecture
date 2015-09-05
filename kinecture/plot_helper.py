#!/usr/bin/env python
# coding=utf-8
""" Helper functions for plotting
"""
__author__ = 'julenka'
import seaborn as sns
import math
import matplotlib.pyplot as plt


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

