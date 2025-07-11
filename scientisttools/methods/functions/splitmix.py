# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import NamedTuple

def splitmix(X) -> NamedTuple:
    """
    Split mixed data
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
    `X` : pandas dataframe of mixed data

    Return
    ------
    nametuple of two dataframe containing : 

    `quanti`: pandas dataframe containing only the quantitative variables or None

    `quali` : pandas dataframe containing only the qualitative variables or None

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_gironde, splitmix
    >>> gironde = load_gironde()
    >>> X_quanti = splitmix(X=gironde).quanti
    >>> X_quali = splitmix(X=girdone).quali
    ```
    """
    # Check if pandas dataframe
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # select object of category
    quali = X.select_dtypes(include=["object","category"])
    if quali.shape[1]==0:
        X_quali = None
    else:
        for col in quali.columns:
            quali[col] = pd.Categorical(quali[col],categories=sorted(np.unique(quali[col])),ordered=True)
        X_quali = quali
    
    # exclude object of category
    quanti = X.select_dtypes(exclude=["object","category"])
    if quanti.shape[1]==0:
        X_quanti = None
    else:
        for col in quanti.columns:
            quanti[col] = quanti[col].astype("float")
        X_quanti = quanti
    return namedtuple("SplitmixResult",["quanti","quali"])(X_quanti,X_quali)