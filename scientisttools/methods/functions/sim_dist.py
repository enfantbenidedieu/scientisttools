# -*- coding: utf-8 -*-
import numpy as np

def sim_dist(X, method= "standard"):
    """
    Transforms similarities matrix to dissimilarities matrix
    --------------------------------------------------------

    Parameters
    ----------
    X : array of float, square matrix.
    method : {'standard','oneminus'} 

    Return
    ------
    D : Dissimilarities matrix

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if X.shape[0] != X.shape[1]:
        raise ValueError("'X' must be square matrix")
    
    # check if method
    if method not in ["standard","oneminus"]:
        raise ValueError("Allowed method are 'standard' or 'oneminus'.")

    if method == "standard":
        D = np.zeros(shape=(X.shape[0],X.shape[0]))
        for i in np.arange(0,X.shape[0]):
            for j in np.arange(0,X.shape[0]):
                D[i,j] = np.sqrt((X[i,i] - X[j,j] +2*X[i,j]))
    elif method == "oneminus":
        D = 1 - X

    return D