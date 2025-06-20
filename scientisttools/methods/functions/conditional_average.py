# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, unique, average
from pandas import concat, DataFrame, Series

#intern functions
from .get_indices import get_indices

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

    `weights` :  a list/tuple/ndarray/Series of length n_samples contains weights onf rows.

    Returns
    -------
    pandas dataframe
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if X is an instance of DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    
    
    # Set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/array/Series of weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/array/Series with length {X.shape[0]}.")
    else:
        weights = array(list(map(lambda x : x/sum(weights),weights)))
        
        #array([x/sum(weights) for x in weights])
    
    def wmean(j):
        vsQual = Y[j]
        modalite = unique(vsQual)
        def mean_k(k):
            idx = get_indices(vsQual,k)
            return DataFrame(average(X.iloc[idx,:],axis=0,weights=weights[idx]).reshape(1,-1),index=[k],columns=X.columns)
        return concat(map(lambda k : mean_k(k),modalite),axis=0)
    return concat(map(lambda j : wmean(j),Y.columns),axis=0)