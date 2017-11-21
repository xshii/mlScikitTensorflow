#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  DataFrameSelector.py
#       Author @  xshi
#  Change date @  11/21/2017 8:33 PM
#        Email @  xshi@kth.se
#  Description @  machine learning with scikit-learn and tensorflow
# ********************************************************************
"""
pipe for select data frame columns
@:param attribte_names = [names of the columns]
@:return data frame
"""

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]