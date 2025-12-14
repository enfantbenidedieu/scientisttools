# -*- coding: utf-8 -*-
from numpy import cov
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

#intern function
from .cov_to_corr import cov_to_corr

def wcorrcoef(X:DataFrame,weights=None):
    """
    Weighted pearson correlation coefficient
    ----------------------------------------

    Description
    -----------
    Performs weighted pearson coerrelation coefficient matrix

    Usage
    -----
    ```python
    >>> wcorrcoef(X, weights)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_columns).
        X contains numerics variables

    `weights`: an optional individuals weights

    Return(s)
    ---------
    `wcorr`: a pandas DataFrame of shape (n_columns, n_columns) containing the weighted correlation matrix of the variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
        raise TypeError("All columns must be numeric")
    #weighted pearson correlation matrix
    wcorr = DataFrame(cov_to_corr(cov(m=X,rowvar=False,aweights=weights,ddof=0)),index=X.columns,columns=X.columns)
    return wcorr