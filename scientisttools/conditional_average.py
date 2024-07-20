# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def conditional_average(X,Y,weights=None):
    """
    Conditional mean
    ----------------

    Description
    -----------
    Compute the weighted average

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_samples, n_columns). X contains quantitative variables

    `Y` : pandas dataframe/series of shape (n_samples, n_columns) or (n_samples,). Y contains qualitative variables

    `weights` :  numpy array of length n_samples contains weights onf rows.

    Returns
    -------
    pandas dataframe
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    
    
    # Set weights
    if weights is None:
        weights = np.ones(X.shape[0])/X.shape[0]
    
    barycentre = pd.DataFrame().astype("float")
    for col in Y.columns:
        vsQual = Y[col]
        modalite = np.unique(vsQual)
        bary = pd.DataFrame(index=modalite,columns=X.columns)
        for mod in modalite:
            idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
            bary.loc[mod,:] = np.average(X.iloc[idx,:],axis=0,weights=weights[idx])
        barycentre = pd.concat((barycentre,bary),axis=0)
    return barycentre