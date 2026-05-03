# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, linalg, zeros, diff, insert, cumsum, c_,nan, diag, sum,sqrt
from pandas import DataFrame, Series, concat, CategoricalDtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from ..onetable._pca import PCA
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd, wcorr, func_groupby
from ..functions.func_eta2 import func_eta2
from ..functions.cov2corr import cov2corr
from ..functions.utils import check_is_bool, is_all_numeric_dtype, is_all_object_or_category_dtype
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix

class DGCANCOR(BaseEstimator,TransformerMixin):
    """
    Dual Generalized Canonical Analysis (DGCANCOR)
    
    
    
    
    
    """
    def __init__(
            self, scale_unit = True, ncp = 5,  group = None, row_w = None, col_w = None, ind_sup = None, sup_var = None, tol = 1e-7
    ):  
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.group = group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
        self.tol = tol

    def fit(self,X,y=None):
        pass