# -*- coding: utf-8 -*-
from collections import OrderedDict
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted

def predict_first_check(
        obj,X
):
    """
    Prediction first check

    Parameters
    ----------
    obj : class
        An object of class

    X : DataFrame of shape (n_rows, n_columns)
        Input data.

    Returns
    -------
    X : DataFrame of shape 
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.DataFrame")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index name as None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X.index.name = None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #drop level if ndim greater than 1 and reset columns name
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.columns.nlevels > 1:
        X.columns = X.columns.droplevel()
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X contains original columns
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not set(obj.call_.X.columns).issubset(X.columns): 
        raise ValueError("The names of the columns is not the same as the ones in the active columns of the {} result".format(obj.__class__.__name__))
    
    #select original columns
    return X[obj.call_.X.columns]

def func_predict(
        X,Y,w,axis=0
):
    """
    Predict supplementary elements (rows/columns)
   
    Performs the coordinates, squared cosinus and squared distance to origin for new elements (rows/columns) with general factor analysis

    Parameters
    ----------
    X : DataFrame of shape (n_rows, n_columns)
        Standardized data

    Y : 2d numpy array of shape (n_rows, n_components) or (n_columns, n_components)
        The right/left matrix of generalized singular value decomposition (GSVD).

    w : Series of shape (n_rows, ) or (n_columns,)
        weights (rows/columns)

    axis : None, str or int, defualt = 0
        indicating which axis to aggregate. Possible values are:

        - None or 0 or "index" indicates aggregating along rows
        - 1 or "columns" indicates aggregating along columns

    Returns
    -------
    result : OrderedDict
        An object with the following keys:
    
        coord : DataFrame of shape (n_rows, n_components) or (n_columns, n_components)
            coordinates of the supplementary rows/columns,

        cos2 : DataFrame of shape (n_rows, n_components) or (n_columns, n_components)
            squared cosinus of the supplementary rows/columns,

        dist2 : Series of shape (n_rows,) or (n_columns,)
            squared distance to origin of the supplementary rows/columns.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.DataFrame")

    if axis in (None, 0, "index"):
        #coordinates and dist2 of the new rows
        coord, sqdisto  = X.mul(w,axis=1).dot(Y), X.pow(2).mul(w,axis=1).sum(axis=1)
        #cos2 of the new rows
        sqcos = coord.pow(2).div(sqdisto,axis=0)
    elif axis in (1, "columns"):
        #coordinates and dist2 of the new columns
        coord, sqdisto = X.mul(w,axis=0).T.dot(Y), X.pow(2).mul(w,axis=0).sum(axis=0)
        #cos2 of the new columns
        sqcos = coord.pow(2).div(sqdisto,axis=0)
    else:
        raise ValueError("'axis' must be either index (0) or columns (1).")
    
    #set columns and names
    coord.columns, sqcos.columns, sqdisto.name  = [f"Dim{x+1}" for x in range(coord.shape[1])], [f"Dim{x+1}" for x in range(sqcos.shape[1])], "Sq. Dist."
    return OrderedDict(coord=coord,cos2=sqcos,dist2=sqdisto)