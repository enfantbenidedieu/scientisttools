# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin

from .function_eta2 import function_eta2
from .weightedcorrcoef import weightedcorrcoef
from .revaluate_cat_variable import revaluate_cat_variable
from .svd_triplet import svd_triplet

# https://github.com/erichson/spca
# https://bioconductor.org/packages/release/bioc/vignettes/scPCA/inst/doc/scpca_intro.html
# https://github.com/rk-terence/Sparse-PCA-with-Elastic-Net/blob/master/sparsePCA_elastic_net.py
# https://github.com/rk-terence/SJSPCA
# https://github.com/cran/elasticnet/blob/master/R/enet_funcs.R
# https://github.com/idnavid/sparse_PCA/tree/master
# https://github.com/yongchunli-13/Sparse-PCA/tree/master

class SparsePCA(BaseEstimator,TransformerMixin):
    """
    Sparse Principal Components Analysis (SparsePCA)
    ------------------------------------------------
    
    
    
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass