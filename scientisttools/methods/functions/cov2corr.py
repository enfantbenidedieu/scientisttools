# -*- coding: utf-8 -*-
from numpy import sqrt, diag, outer
from pandas import DataFrame

def cov2corr(X):
    """
    Covariance to Correlation
    
    Parameters
    ----------
    X : 2d array-like of shape (n_columns, n_columns)
        Covariance matrix.

    Returns
    -------
    Y : 2d array-like of shape (n_columns, n_columns)
        Correlation matrix.
    """
    if X.shape[0] != X.shape[1]:
        raise TypeError("`X` must be a squared matrix")

    v = diag(X)
    Y = X / sqrt(outer(v, v))
    Y[X == 0] = 0 

    #convert to DataFrame
    if isinstance(X,DataFrame):
        Y = DataFrame(Y,index=X.columns,columns=X.columns)
    return Y