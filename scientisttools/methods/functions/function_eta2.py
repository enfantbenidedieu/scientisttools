# -*- coding: utf-8 -*-
from numpy import array, ones
from pandas import DataFrame, Series, get_dummies, concat

def function_eta2(X,Y,weights=None,excl=None):
    """
    Squared correlation ratio - Eta2
    --------------------------------

    Description
    -----------
    Perform squared correlation ratio - eta square

    Parameters
    ----------
    `X`: pandas DataFrame/Series of shape (n_rows, n_columns) or (n_rows, ), dataframe containing categoricals variables

    `Y`: pandas DataFrame/Series of shape (n_rows, n_columns) or (n_rows, ), DataFrame containing quantitative variables

    `weights`: array with the weights of each row

    `excl`: a list indicating excluded categories

    Return
    ------
    `value`: pandas DataFrame of shape (n_categories, n_columns)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    """
    #convert to DataFrame if Series
    if isinstance(X,Series):
        X = X.to_frame()

    #convert to DataFrame if Series
    if isinstance(Y,Series):
        Y =  Y.to_frame()
    
    #check if pandas DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #check if pandas DataFrame
    if not isinstance(Y, DataFrame):
        raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #set weight if None
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    else:
        weights = array([x/sum(weights) for x in weights])
   
    def func_eta2(X,q,Y,weights=None,excl=None):
        def eta2(k):
            #dummies tables
            dummies = get_dummies(X[q],dtype=int)
            #remove exclude
            if excl is not None:
                intersect = list(set(excl) & set(dummies.columns))
                if len(intersect)>=1:
                    dummies = dummies.drop(columns=intersect)
            num_k, n_k = dummies.mul(Y.iloc[:,k],axis=0).mul(weights,axis=0).sum(axis=0).pow(2), dummies.mul(weights,axis=0).sum(axis=0)
            return num_k.div(n_k).sum()/Y.iloc[:,k].pow(2).mul(weights).sum()
        return DataFrame([[eta2(k) for k in range(Y.shape[1])]],index=[q],columns=Y.columns)
    #application
    return concat((func_eta2(X=X,q=q,Y=Y,weights=weights,excl=excl) for q in X.columns),axis=0)