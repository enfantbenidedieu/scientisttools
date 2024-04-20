# -*- coding: utf-8 -*-
import numpy as np

def revaluate_cat_variable(X):
    """
    Revaluate Categorical variable
    ------------------------------

    Parameters
    ----------
    X : pandas DataFrame of shape (n_rows, n_columns)

    Return
    ------
    X : pandas DataFrame of shape (n_rows, n_columns)
    """
    # check if shape greater than 1:
    if X.shape[1]>1:
        for i in range(X.shape[1]-1):
            for j in range(i+1,X.shape[1]):
                if X.iloc[:,i].dtype in ["object","category"] and X.iloc[:,j].dtype in ["object","category"]:
                    intersect = list(set(np.unique(X.iloc[:,i]).tolist()) & set(np.unique(X.iloc[:,j]).tolist()))
                    if len(intersect)>=1:
                        valuei = {x : X.columns.tolist()[i]+"_"+str(x) for x in np.unique(X.iloc[:,i]).tolist()}
                        valuej = {x : X.columns.tolist()[j]+"_"+str(x) for x in np.unique(X.iloc[:,j]).tolist()}
                        X.iloc[:,i],X.iloc[:,j] = X.iloc[:,i].map(valuei), X.iloc[:,j].map(valuej)
    return X