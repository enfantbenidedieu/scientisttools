# -*- coding: utf-8 -*-
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted

def predict_first_check(obj,X):
    """
    Prediction first check

    Parameters
    ----------
    obj : class
        An object of class

    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    Returns
    -------
    X : DataFrame of shape (n_samples, n_columns)

    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

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

def func_predict(X,Y,w,axis=0):
    """
    Predict supplementary elements (rows/columns)
   
    Performs the coordinates, squared cosinus and squared distance to origin for new elements (rows/columns) with general factor analysis

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Standardized data.

    Y : 2d numpy array of shape (n_samples, ncp) or (n_columns, ncp)
        The right/left matrix of generalized singular value decomposition (GSVD).

    w : Series of shape (n_samples, ) or (n_columns,)
        weights (rows/columns)

    axis : None, str or int, defualt = 0
        indicating which axis to aggregate. Possible values are:

        * None or 0 or "index" indicates aggregating along rows
        * 1 or "columns" indicates aggregating along columns

    Returns
    -------
    result : dict
        An object with the following keys:
    
        coord : DataFrame of shape (n_samples, ncp) or (n_columns, ncp)
            coordinates of the supplementary rows/columns,

        cos2 : DataFrame of shape (n_samples, ncp) or (n_columns, ncp)
            squared cosinus of the supplementary rows/columns,

        dist2 : Series of shape (n_samples,) or (n_columns,)
            squared distance to origin of the supplementary rows/columns.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #coordinates and dist2 of the new rows/columns
    if axis in (None, 0, "index"):
        coord, sqdisto = (X * w).dot(Y), ((X ** 2) * w).sum(axis=1)
    elif axis in (1, "columns"):
        coord, sqdisto = (X.T * w).dot(Y), ((X ** 2).T * w).sum(axis=1)
    else:
        raise ValueError("axis must be either index (0) or columns (1).")
    
    # set columns and names
    sqdisto.name, coord.columns = "Sq. Dist.", [f"Dim{x+1}" for x in range(coord.shape[1])]
    # cos2 of the new rows/columns
    sqcos = ((coord ** 2).T/sqdisto).T
    
    # return as dictionary
    return {"coord":coord, "cos2":sqcos, "dist2":sqdisto}