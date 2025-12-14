# -*- coding: utf-8 -*-
from numpy import number
from pandas import DataFrame, Series, Categorical, concat
from collections import namedtuple
from typing import NamedTuple

def splitmix(X) -> NamedTuple:
    """
    Split Mixed Data
    ----------------

    Description
    -----------
    Splits a mixed data matrix in two data sets: one with the quantitative variables and one with the qualitative variables.

    Usage
    -----
    ```python
    >>> splitmix(X)
    ```
    
    Parameters
    ----------
    `X`: a pandas dataframe of shape (n_row, n_columns)

    Return
    ------
    nametuple containing: 

    `quanti`: None or a pandas DataFrame containing only the quantitative variables

    `quali`: None or a pandas DataFrame containing only the qualitative variables

    `n`: a numeric value indicating the number of rows.

    `k1`: a numeric value indicating the number of quantitative variables

    `k2`: a numeric value indicating the number of qualitative variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import splitmix
    >>> split_x = splitmix(wine)
    >>> X_quanti, X_quali, n_quanti, n_quali = split_x.quanti, split_x.quali, split_x.k1, split_x.k2
    ```
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #initialisation
    X_quali, X_quanti, n_quali, n_quanti = None, None, 0, 0

    #select object or category
    is_quali = X.select_dtypes(include=["object","category"])
    if not is_quali.empty:
        X_quali = concat((Series(Categorical(is_quali[q],categories=sorted(is_quali[q].dropna().unique().tolist()),ordered=True),index=is_quali.index,name=q) for q in is_quali.columns),axis=1)
        if isinstance(X_quali, Series):
            X_quali = X_quali.to_frame()
        n_quali = X_quali.shape[1]
    
    #select all numerics columns
    is_quanti = X.select_dtypes(include=number)
    if not is_quanti.empty:
        X_quanti = concat((is_quanti[k].astype(float) for k in is_quanti.columns),axis=1)
        if isinstance(X_quanti, Series):
            X_quanti = X_quanti.to_frame()
        n_quanti = X_quanti.shape[1]

    #convert to namedtuple
    return namedtuple("SplitmixResult",["quanti","quali","n","k1","k2"])(X_quanti,X_quali,X.shape[0],n_quanti, n_quali)