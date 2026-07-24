# -*- coding: utf-8 -*-
from numpy import eye,sqrt, diag,squeeze,ones,linalg
from pandas import DataFrame
from collections import namedtuple

def varimax(loadings,
            normalize=True,
            max_iter=1000,
            tol=1e-5):
    """
    Varimax rotation
    
    Perform varimax rotation, with optional Kaiser normalization.
    
    Parameters
    ----------
    loadings : DataFrame of shape (n_columns, n_components)
        The loading table.

    normalize : bool, default = True 
        Should Kaiser normalization be performed? If so the rows of loadings are re-scaled to unit length before rotation, and scaled back afterwards.

    max_iter : int, optional, defauult = 1000.
        The maximum number of iterations. Used for 'varimax' and 'oblique' rotations.

    tol : float, default = 1e-5
        The tolerance for stopping. The relative change in the sum of singular values.

    Returns
    -------
    result : varimaxResult
        An object with the following attributes:

        loadings : DataFrame of shape (n_columns, n_components)
            The rotated loadings matrix.

        rotmat : DataFrame of shape (n_components, n_components)
            The rotation matrix.

    References
    ----------
    [1] Kaiser HF (1958). `The Varimax Criterion for Analytic Rotation in Factor Analysis. <https://www.cambridge.org/core/journals/psychometrika/article/abs/varimax-criterion-for-analytic-rotation-in-factor-analysis/88F99AA31F472BF854B01B6B92F4212B>`_ Psychometrika, 23(3), 187-200.
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
        #transform data for singular value decomposition using updated formula :
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
    loadings.columns = [f"Dim{x+1}" for x in range(n_cols)]
    rotmat = DataFrame(rotmat,index=loadings.columns,columns=loadings.columns)
    #convert to namedtuple
    return namedtuple("varimaxResult",["loadings","rotmat"])(loadings,rotmat)