# -*- coding: utf-8 -*-
from numpy import sqrt, diag, outer
from pandas import DataFrame

def cov_to_corr(X):
    """
    Covariance to Correlation
    -------------------------

    Parameters
    ----------
    `X`: a numpy 2-D array or a pandas DataFrame with (n_samples, n_columns)
    
    Reference
    ---------
    see # https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """
    if X.shape[0] != X.shape[1]:
        raise TypeError("`X` must be a squared matrix")

    v = sqrt(diag(X))
    outer_v = outer(v, v)
    Y = X / outer_v
    Y[X == 0] = 0

    #convert to DataFrame
    if isinstance(X,DataFrame):
        Y = DataFrame(Y,index=X.columns,columns=X.columns)

    return Y