# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones
from pandas import DataFrame, Series, get_dummies, concat

#interns function
from .utils import convert_series_to_dataframe, check_is_dataframe

def func_eta2(X, by, w=None, excl=None):
    """
    Squared correlation ratio - Eta2
    
    Perform squared correlation ratio - eta square

    Parameters
    ----------
    Y : DataFrame of shape (n_samples, n_xcolumns) or Series of shape (n_samples,)
        Y input data, which contains continuous variables.

    by : DataFrame of shape (n_samples, n_bycolumns) or Series of shape (n_samples,)
        by input data, which contains categorical variables.

    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in X.

    excl : list, default = None
        Excluded categories.

    Returns
    -------
    eta2 : DataFrame of shape (n_bycolumns, n_xcolumns)
        Eta-squared.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert to an object of class pd.DataFrame if an object of class pd.Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X, by = convert_series_to_dataframe(X), convert_series_to_dataframe(by)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if input data are object of class pandas.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X), check_is_dataframe(by)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X and by have same number of rows
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.shape[0] != by.shape[0]:
        raise TypeError("Not convenient dimension between X and by")
    
    #set number of rows
    n_rows = X.shape[0]

    #set weights
    if w is None: 
        w = ones(n_rows)/n_rows
    elif not isinstance(w,(list,tuple,ndarray,Series)): 
        raise TypeError("w must be a 1d array-like of rows weights.")
    elif len(w) != n_rows: 
        raise ValueError(f"w must be a 1d array-like with length {n_rows}.")
    else: 
        w = array(w)/sum(w)
    
    def weta2_q(q,w=None,excl=None):
        def weta2_kq(kq):
            #dummies tables
            dummies = get_dummies(by[q],dtype=int)
            #remove exclude
            if excl is not None:
                intersect = list(set(excl) & set(dummies.columns))
                if len(intersect) >= 1: 
                    dummies = dummies.drop(columns=intersect)
            n_k  = (dummies.T * w).sum(axis=1)
            num_k = ((dummies.T * X.iloc[:,kq]) * w).sum(axis=1) ** 2
            return (num_k/n_k).sum()/((X.iloc[:,kq]**2) * w).sum()
        return DataFrame([[weta2_kq(k) for k in range(X.shape[1])]],index=[q],columns=X.columns)
    #application
    return concat((weta2_q(q=q,w=w,excl=excl) for q in by.columns),axis=0)