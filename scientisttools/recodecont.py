# -*- coding: utf-8 -*-
import pandas as pd
from collections import namedtuple

def recodecont(X):
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
    `X` : pandas dataframe of continuous variables

    Return
    ------
    A NamedTuple containing:
        - `Xcod` : the continuous DataFrame X with missing values replaced with the column mean values
        - `Z` : the standardizd continuous dataframe
        - `center` : the mean value for each columns in X
        - `scale` : the standard deviation for each columns of X
 
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if pandas dataframe
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # exclude object of category
    X = X.select_dtypes(exclude=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be numeric")
    else:
        X = pd.concat((X[col].astype("float") for col in X.columns),axis=1)

    # Fill NA by mean
    if X.shape[0] > 1:
        for col in X.columns:
            if X.loc[:,col].isnull().any():
                X.loc[:,col] = X.loc[:,col].fillna(X.loc[:,col].mean())
    
    if X.shape[0] == 1:
        Xcod = X
        Z = None
        center = X
        scale = None
    else:
        Xcod = X
        center = X.mean(axis=0)
        scale = X.std(axis=0,ddof=0)
        Z = Xcod.apply(lambda x : (x - center.values)/scale.values,axis=1)
    
    return namedtuple("recodecont",["Xcod","Z","center","scale"])(Xcod,Z,center,scale)