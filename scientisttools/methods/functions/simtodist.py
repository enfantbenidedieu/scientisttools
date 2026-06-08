# -*- coding: utf-8 -*-
from numpy import zeros, sqrt

def simtodist(
        X, method= "standard"
):
    """
    Transforms similarities matrix to dissimilarities matrix

    Parameters
    ----------
    X : 2D array-like of shape (n_columns, n_columns)
        Similarities matrix.

    method : str, default = 'standard'

        - 'standard'
        - 'oneminus'

    Returns
    -------
    D : 2D array-like of shape (n_columns, n_columns)
        Dissimilarities matrix
    """
    if X.shape[0] != X.shape[1]:
        raise ValueError("'X' must be square matrix")
    
    if method == "standard":
        D = zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i,j] = sqrt((X[i,i] - X[j,j] + 2*X[i,j]))
    elif method == "oneminus":
        D = 1 - X
    else:
        raise ValueError("Allowed method are 'standard' or 'oneminus'.")

    return D