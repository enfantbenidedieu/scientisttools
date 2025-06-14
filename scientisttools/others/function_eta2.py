# -*- coding: utf-8 -*-
from numpy import array, ones
from pandas import DataFrame, get_dummies, concat
from mapply.mapply import mapply

def function_eta2(X,Y,weights=None,excl=None,n_workers=1):
    """
    Square correlation ratio - Eta2
    -------------------------------

    Description
    -----------
    Perform square correlation ratio - eta square

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_rows, n_columns), dataframe containing categoricals variables

    `Y` : pandas dataframe of individuals coordinates of shape (n_rows, n_components)

    `weights` : array with the weights of each row

    `excl` : a list indicating excluded categories

    `n_workers` : Maximum amount of workers (processes) to spawn.

    Return
    ------
    `value` : pandas dataframe of shape (n_categories, n_components)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """

    # Set weight if None
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    else:
        weights = array([x/sum(weights) for x in weights])
   
    def func_eta2(X,j,Y,weights=None,excl=None,n_workers=1):
        def fct_eta2(i):
            #dummies tables
            dummies = get_dummies(X[j],dtype=int)
            #remove exclude
            if excl is not None:
                intersect = list(set(excl) & set(dummies.columns.tolist()))
                if len(intersect)>=1:
                    dummies = dummies.drop(columns=intersect)

            n_k  = mapply(dummies, lambda x : x*weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            num_k = mapply(mapply(dummies,lambda x : x*Y.iloc[:,i],axis=0,progressbar=False,n_workers=n_workers),lambda x : x*weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0).pow(2)
            num, denom = sum([num_k[k]/n_k[k] for k in num_k.index]), sum(Y.iloc[:,i]*Y.iloc[:,i]*weights)
            return num/denom
        return DataFrame([list(map(lambda i : fct_eta2(i),range(Y.shape[1])))],index=[j],columns=["Dim."+str(x+1) for x in range(Y.shape[1])])
    #application
    value = concat((func_eta2(X=X,j=j,Y=Y,weights=weights,excl=excl,n_workers=n_workers) for j in X.columns),axis=0)
    return value