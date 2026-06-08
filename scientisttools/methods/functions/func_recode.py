# -*- coding: utf-8 -*-
from numpy import number
from pandas import DataFrame, Series
from collections import namedtuple
from typing import NamedTuple

#intern functions
from .func_fillna import func_fillna
from .revalue import revalue

def func_recode(
        X
) -> NamedTuple:
    """
    Recode Data

    Parameters
    ----------
    X : array-like of shape (n_samples, n_columns) or (n_samples,)
        Input data.

    Returns
    -------
    result : recodeResult
        An object with the following attributes:

        quanti: DataFrame of shape (n_samples, k1) default = None
            Continuous variables.

        quali: DataFrame of shape (n_samples, k2), default = None
            Categorical variables.

        n: int, default = n_samples
            Number of rows.

        k1: int, default = 0
            Number of continuous variables.

        k2: int, default = 0
            Number of categorical variables.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert series to dataframe
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.DataFrame")

    #initialisation
    X_quanti, X_quali, n, k1, k2 = None, None, X.shape[0], 0, 0

    #select all numerics columns
    is_quanti = X.select_dtypes(include=number)
    if not is_quanti.empty:
        X_quanti = is_quanti.to_frame() if isinstance(is_quanti,Series) else is_quanti
        #fill NA by mean
        X_quanti = func_fillna(X=X_quanti, method="mean")
        k1 = X_quanti.shape[1]

    #select all categorics columns
    is_quali = X.select_dtypes(include=["object","category"])
    if not is_quali.empty:
        X_quali = is_quali.to_frame() if isinstance(is_quali,Series) else is_quali
        #fill NA by most_frequency & revalue
        X_quali = revalue(X=func_fillna(X=X_quali, method="most_frequent"))
        k2 = X_quali.shape[1]    
    return namedtuple("recodeResult",["quanti","quali","n","k1","k2"])(X_quanti,X_quali,n,k1,k2)