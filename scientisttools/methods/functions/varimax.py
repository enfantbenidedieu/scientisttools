# -*- coding: utf-8 -*-
from numpy import eye,sqrt, diag,squeeze,ones,linalg
from pandas import DataFrame
from collections import namedtuple
from typing import NamedTuple

def varimax(loadings:DataFrame,normalize=True,max_iter=1000,tol=1e-5) -> NamedTuple:
    """
    Varimax rotation
    ----------------

    Description
    -----------
    Perform varimax rotation, with optional Kaiser normalization
    
    Parameters
    ----------
    `loadings`: a pandas DataFrame of shape (n_columns, n_components)
        The loading table.

    `normalize`: logical.  Should Kaiser normalization be performed? If so the rows of loadings are re-scaled to unit length before rotation, and scaled back afterwards.

    `max_iter`: integer, optional. Defaults to 1000.
        The maximum number of iterations. Used for 'varimax' and 'oblique' rotations.

    `tol`: numeric. The tolerance for stopping: the relative change in the sum of singular values.

    Return(s)
    ---------
    a namedtuple containing:

    `loadings`: a pandas DataFrame of shape (n_columns, n_components) of the rotated loadings
        The loadings matrix.

    `rotmat`: a pandas DataFrame of the rotation matrix.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    see https://stat.ethz.ch/R-manual/R-devel/library/stats/html/varimax.html

    """
    #make a copy of loadings
    X = loadings.copy()
    #shape of loadings
    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X

    # normalize the loadings matrix using sqrt of the sum of squares (Kaiser)
    if normalize:
        sc = X.copy().apply(lambda x: sqrt(sum(x**2)),axis=1)
    else:
        sc = 1
    X = (X.T / sc).T

    #initialize the rotation matrix to N x N identity matrix
    rotmat = eye(n_cols)
    d = 0
    for _ in range(max_iter):
        old_d = d
        #take inner product of loading matrix and rotation matrix
        z = X.dot(rotmat)
        #transform data for singular value decomposition using updated formula : B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
        B = X.T.dot(z.pow(3) - z.dot(diag(squeeze(ones(n_rows).dot(z.pow(2))))) / n_rows)
        #perform SVD on the transformed matrix
        U, S, V = linalg.svd(B)
        #take inner product of U and V, and sum of S
        rotmat = U.dot(V)
        d = sum(S)
        # check convergence
        if d < old_d * (1 + tol):
            break

    #take inner product of loading matrix and rotation matrix
    X =  X.dot(rotmat)
    #de-normalize the data
    if normalize:
        X = X.T.mul(sc)
    else:
        X = X.T
    #convert to DataFrame
    loadings = X.T.copy()
    loadings.columns = ["Dim."+ str(x+1) for x in range(n_cols)]
    rotmat = DataFrame(rotmat,index=loadings.columns,columns=loadings.columns)
    #convert to namedtuple
    return namedtuple("varimax",["loadings","rotmat"])(loadings,rotmat)