# -*- coding: utf-8 -*-
from numpy import ones,array,ndarray,unique,average
from pandas import concat, DataFrame

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
    # Check if X is an instance of DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    
    
    # Set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray)):
        raise TypeError("'weights' must be a list/tuple/array of weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/array with length {X.shape[0]}.")
    else:
        weights = array([x/sum(weights) for x in weights])
    
    barycentre = DataFrame().astype("float")
    for col in Y.columns:
        vsQual = Y[col]
        modalite = unique(vsQual)
        bary = DataFrame(index=modalite,columns=X.columns)
        for mod in modalite:
            idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
            bary.loc[mod,:] = average(X.iloc[idx,:],axis=0,weights=weights[idx])
        barycentre = concat((barycentre,bary),axis=0)
    return barycentre