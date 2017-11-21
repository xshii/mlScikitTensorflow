#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  LabelBinarizerPipe.py
#       Author @  xshi
#  Change date @  11/21/2017 8:29 PM
#        Email @  xshi@kth.se
#  Description @  machine learning with scikit-learn and tensorflow
# ********************************************************************
"""
pipe Version of LabelBinarizer text->cat vectors
@:param sparse_output
@:result array of list[{0,1}]
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

class LabelBinarizerPipe(BaseEstimator,TransformerMixin):
    def __init__(self, sparse_output=True):
        self.sparse_output = sparse_output


    def fit(self, X, y=None):
        self.encoder = LabelBinarizer(sparse_output=self.sparse_output)
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        result = self.encoder.transform(X)
        return result