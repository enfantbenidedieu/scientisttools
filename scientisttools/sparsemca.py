# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin


class SparseMCA(BaseEstimator,TransformerMixin):
    """
    Sparse Multiple Correspondence Analysis (SparseMCA)
    ---------------------------------------------------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass