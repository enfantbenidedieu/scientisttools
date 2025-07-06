# -*- coding: utf-8 -*-
from pandas import DataFrame,concat
from typing import NamedTuple
from collections import namedtuple

def recodecont(X) -> NamedTuple:
    """
    Recoding of the continuous data matrix
    ----------------------------------------

    Description
    -----------
    Recoding of the continuous data matrix

    Usage
    -----
    ```python
    >>> from scientisttools import recodecont
    >>> recodcont = recodecont(X)
    ```

    Parameters
    ----------
    `X`: pandas dataframe of continuous variables

    Return
    ------
    namedtuple containing:
        - `X`: the continuous DataFrame X with missing values replaced with the column mean values
        - `Z`: the standardizd continuous dataframe
        - `center`: the mean value for each columns in X
        - `scale`: the standard deviation for each columns of X
 
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if pandas dataframe
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # exclude object of category
    X = X.select_dtypes(exclude=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be numeric")
    else:
        X = concat((X[j].astype("float") for j in X.columns),axis=1)

    # Fill NA by mean
    if X.shape[0] > 1:
        for j in X.columns:
            if X.loc[:,j].isnull().any():
                X.loc[:,j] = X.loc[:,j].fillna(X.loc[:,j].mean())
    
    if X.shape[0] == 1:
        Z, center, scale = None, X, None
    else:
        center, scale =  X.mean(axis=0), X.std(axis=0,ddof=0) 
        Z = X.apply(lambda x : (x - center.values)/scale.values,axis=1)
    
    return namedtuple("recodecont",["X","Z","center","scale"])(X,Z,center,scale)