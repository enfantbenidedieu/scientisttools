# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from mapply.mapply import mapply

def function_eta2(X,lab,x,weights,n_workers):
    """
    Correlation ratio - Eta2
    ------------------------

    Description
    -----------
    Perform correlation ratio eta square

    Parameters
    ----------
    X : pandas dataframe of shape (n_rows, n_columns), dataframe containing categoricals variables

    lab : name of a columns in X

    x : pandas dataframe of individuals coordinates of shape (n_rows, n_components)

    weights : array with the weights of each row

    n_workers : Maximum amount of workers (processes) to spawn.

    Return
    ------
    value : pandas dataframe of shape (n_columns, n_components)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    def fct_eta2(idx):
        tt = pd.get_dummies(X[lab])
        ni  = mapply(tt, lambda k : k*weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        num = mapply(mapply(tt,lambda k : k*x[:,idx],axis=0,progressbar=False,n_workers=n_workers),
                        lambda k : k*weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)**2
        num = sum([num[k]/ni[k] for k in num.index])
        denom = sum(x[:,idx]*x[:,idx]*weights)
        return num/denom
    res = pd.DataFrame(np.array(list(map(lambda i : fct_eta2(i),range(x.shape[1])))).reshape(1,-1),
                        index=[lab],columns=["Dim."+str(x+1) for x in range(x.shape[1])])
    return res