# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, sum, average, sqrt, repeat
from pandas import Series, DataFrame

def func_Lg(
        X,Y,row_w=None, xcol_w=None,ycol_w=None
):
    """
    Calulate the Lg coefficients
    
    Calculate the Lg coefficients between two groups X and Y

    Parameters
    ----------
    X : Dataframe of shape (n_samples, n_columns)
        First groups

    Y : Dataframe of shape (n_samples, n_columns)
        Second group
        
    X_weights : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for the variables in X

    Y_weights : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for the variables in Y

    ind_weights : `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.

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
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.DataFrame")
    if not isinstance(Y,DataFrame):
        raise TypeError(f"{type(Y)} is not supported. Y must be an object of class pd.DataFrame")
    
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
    X = X.sub(average(X,axis=0,weights=row_w),axis=1).mul(sqrt(xcol_w),axis=1).mul(sqrt(row_w),axis=0)
    Y = Y.sub(average(Y,axis=0,weights=row_w),axis=1).mul(sqrt(ycol_w),axis=1).mul(sqrt(row_w),axis=0)
    return sum([sum(X.iloc[:,i].dot(Y)**2) for i in range(X.shape[1])])