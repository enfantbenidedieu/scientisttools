# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def function_lg(X,Y,X_weights=None,Y_weights=None,ind_weights=None):
    """
    Calulate the Lg coefficients between two groups
    -----------------------------------------------

    Description
    -----------
    Calculate the Lg coefficients between two groups X and Y

    Usage
    -----
    ```
    >>> function_lg(X,Y,X_weights=None,Y_weights=None,ind_weights=None)
    ```

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_samples, n_columns)

    `Y` : pandas dataframe of shape (n_samples, n_columns)

    'X_weights' : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for the variables in X

    `Y_weights` : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for the variables in Y

    `ind_weights` : `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.

    Returns
    -------
    lg : a numeric value
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if X is an instance of pandas series
    if isinstance(X,pd.Series):
        X = X.to_frame()
    
    # Check if Y is an instance of pandas series
    if isinstance(Y,pd.Series):
        Y = Y.to_frame()

    # Check if X is an instance of pandas DataFrame
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if not isinstance(Y,pd.DataFrame):
        raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Set individuals weights
    if ind_weights is None:
        ind_weights = np.ones(X.shape[1])/X.shape[1]
    else:
        ind_weights = ind_weights/np.sum(ind_weights)
    
    if X_weights is None:
        X_weights = np.ones(X.shape[1])
    
    if Y_weights is None:
        Y_weights = np.ones(Y.shape[1])
    
    # columns average of X
    X_center = X.apply(lambda x : x*ind_weights,axis=0).sum(axis=0)
    # Update X with weights
    X = (X - X_center.values.reshape(1,-1)).apply(lambda x : x*np.sqrt(X_weights),axis=1).apply(lambda x : x*np.sqrt(ind_weights),axis=0)
    # Columns average of Y
    Y_center = Y.apply(lambda x : x*ind_weights,axis=0).sum(axis=0)
    # Update Y with weights
    Y = (Y - Y_center.values.reshape(1,-1)).apply(lambda x : x*np.sqrt(Y_weights),axis=1).apply(lambda x : x*np.sqrt(ind_weights),axis=0)

    lg = 0.0
    for i in range(X.shape[1]):
        lg = lg + (X.iloc[:,i].dot(Y)**2).sum()
    return lg