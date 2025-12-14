# -*- coding: utf-8 -*-
from numpy import array, ndarray, ones
from pandas import DataFrame,Series,concat
from statsmodels.stats.weightstats import DescrStatsW
from typing import NamedTuple
from collections import namedtuple

def recodecont(X,weights=None) -> NamedTuple:
    """
    Recoding of the continuous data
    -------------------------------

    Description
    -----------
    Recoding of the continuous data

    Usage
    -----
    ```python
    >>> from scientisttools import recodecont
    >>> recodcont = recodecont(X)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame/Series of continuous variables

    `weights`: an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals

    Return
    ------
    namedtuple of pandas DataFrame/Series containing:

    `X`: the continuous DataFrame X with missing values replaced with the column mean values,
    
    `Z`: the standardizd continuous DataFrame,
    
    `center`: the mean value for each columns in X,
    
    `scale`: the standard deviation for each columns of X.
 
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if isinstance(X,Series): #convert to pandas DataFrame if pandas Series
        X = X.to_frame()

    if not isinstance(X,DataFrame): #check if pandas dataframe
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/1darray/Series of individuals weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/1darray/Series with length {X.shape[0]}.")
    else:
        weights = array([x/sum(weights) for x in weights])

    #convert to Series
    weights = Series(weights,index=X.index,name="weight")

    #exclude object of category
    X = X.select_dtypes(exclude=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be numeric")
    else:
        X = concat((X[j].astype("float") for j in X.columns),axis=1)

    #fill NA by mean
    for j in X.columns:
        if X.loc[:,j].isnull().any():
            X.loc[:,j] = X.loc[:,j].fillna(X.loc[:,j].mean())
    
    if X.shape[0] == 1:
        Z, center, scale = None, X, None
    else:
        d_x = DescrStatsW(X,weights=weights,ddof=0) #compute weighted average and standard deviation
        center, scale = Series(d_x.mean,index=X.columns,name="center"), Series(d_x.std,index=X.columns,name="scale")
        Z = X.apply(lambda x : (x - center)/scale,axis=1)
    
    return namedtuple("recodecont",["X","Z","center","scale"])(X,Z,center,scale)