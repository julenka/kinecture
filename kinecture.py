#!/usr/bin/env python
# coding=utf-8
""" 
"""
__author__ = 'julenka'

import pandas as pd

def remove_columns(data, columns):
    for column_name in columns:
            del data[column_name]