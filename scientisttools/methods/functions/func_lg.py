# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, sum, average, sqrt, repeat
from pandas import Series, DataFrame

def func_Lg(X,Y,row_w=None, xcol_w=None,ycol_w=None):
    """
    Calulate the Lg coefficients
    
    Calculate the Lg coefficients between two groups X and Y

    Parameters
    ----------
    X : Dataframe of shape (n_samples, n_xcolumns)
        First group.

    Y : Dataframe of shape (n_samples, n_ycolumns)
        Second group.
        
    xcol_w : 1d array-like of shape (n_xcolumns,), default = None
        An optional variables weights for X.

    ycol_w : 1d array-like of shape (n_ycolumns,), default = None
        An optional variables weights for Y.

    row_w : 1d array-like of shape (n_samples,), default = None
        An optional individuals weights.
        
    Returns
    -------
    lg : float
        lg coefficient.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()
    if isinstance(Y,Series):
        Y = Y.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X and Y are an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    if not isinstance(Y,DataFrame):
        raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #check if len are equal
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of samples in X must be equal to the number of samples in Y")

    #set dimenstion
    n_rows, n_xcols = X.shape
    n_ycols = Y.shape[1]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #rows weights
    if row_w is None:
        row_w = ones(n_rows)/n_rows
    elif isinstance(row_w,(int,float)):
        w = repeat(w,n_rows)
        row_w = array(w)/sum(w)
    elif not isinstance(row_w,(list,tuple,ndarray,Series)):
        raise TypeError(f"{type(row_w)} is not supported.")
    else:
        if len(row_w) != n_rows:
            raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
        row_w = array(row_w)/sum(row_w)
    
    #X columns weights
    if xcol_w is None:
        xcol_w = ones(n_xcols)
    elif isinstance(xcol_w,(int,float)):
        xcol_w = repeat(xcol_w,n_xcols)
    elif not isinstance(xcol_w,(list,tuple,ndarray,Series)):
        raise TypeError(f"{type(xcol_w)} is not supported.")
    else:
        if len(xcol_w) != n_xcols:
            raise ValueError(f"xcol_w must be a 1d array-like of shape ({n_xcols},).")
        xcol_w = array(xcol_w)
    
    #set Y columns weights
    if ycol_w is None:
        ycol_w = ones(n_ycols)
    elif isinstance(ycol_w,(int,float)):
        ycol_w = repeat(ycol_w,n_ycols)
    elif not isinstance(ycol_w,(list,tuple,ndarray,Series)):
        raise TypeError(f"{type(ycol_w)} is not supported.")
    else:
        if len(ycol_w) != n_ycols:
            raise ValueError(f"ycol_w must be a 1d array-like of shape ({n_ycols},).")
        ycol_w = array(ycol_w)

    #update X and Y
    X = (((X - average(X,axis=0,weights=row_w)) * sqrt(xcol_w)).T * sqrt(row_w)).T
    Y = (((Y - average(Y,axis=0,weights=row_w)) * sqrt(ycol_w)).T * sqrt(row_w)).T
    lg = sum([sum(X.iloc[:,i].dot(Y)**2) for i in range(n_xcols)])
    return lg