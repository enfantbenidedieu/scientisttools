# -*- coding: utf-8 -*-
from numpy import ones, array,apply_along_axis,sqrt,linalg,sign, nan_to_num, diag, set_printoptions, errstate
from collections import namedtuple
set_printoptions(suppress=True)
errstate(invalid='ignore', divide='ignore')

def gSVD(
        X, ncp=5, row_w=None, col_w=None, tol = 1e-7
):
    """
    Generalized Singular Value Decomposition (GSVD) of a Matrix

    Compute the generalized singular value decomposition (gsvd) of a rectangular matrix with weights for rows and columns

    Parameters
    ----------
    X : DataFrame of shape (n_rows, n_columns)
        Input data.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_w : 1d array-like of shape (n_rows,), default = None
        The rows weights.

    col_w : 1d array-like of shape (n_columns,), default = None
        The columns weights.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.

    Returns
    -------
    result : gSVDResult
        An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
            
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_columns, maxcp)
            The left singular vectors.
        V : 2d numpy array of shape (n_rows, maxcp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set number of rows and columns weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set number of rows and columns
    n_rows, n_cols = X.shape

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set rows and columsn weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set row weights
    if row_w is None: 
        row_w = ones(n_rows)
    
    #set columns weights
    if col_w is None: 
        col_w = ones(n_cols)
    
    #multiply X with square root of rows and columns weights
    X = (X.T * sqrt(row_w)).T * sqrt(col_w)

    if n_cols < n_rows:
        svd = linalg.svd(X,full_matrices=False)
        U, V = svd[0], svd[2].T
    else:
        svd = linalg.svd(X.T,full_matrices=False)
        U, V = svd[2].T, svd[0]

    #set maximum number of components
    rank = sum(((svd[1]/svd[1][0])**2)>tol)
    
    #set number of components
    if ncp is None: 
        ncp = rank
    elif ncp < 1: 
        raise ValueError("'ncp' must be strictly positive")
    else: 
        ncp = int(min(ncp,rank))

    if ncp > 1:
        mult = array([1 if x == 0 else x for x in sign(V.sum(axis=0))])
        U, V = U*mult, V*mult

    #extract singular values decomposition
    vs = svd[1][:rank]
    #recalibrate U and V using row weight and col weight
    U, V = U.T.dot(diag(1/sqrt(row_w))).T, V.T.dot(diag(1/sqrt(col_w))).T
    #replace NAN, inf with 1e-15
    U, V = nan_to_num(U,nan=1e-15,posinf=1e-15,neginf=-1e-15), nan_to_num(V,nan=1e-15,posinf=1e-15,neginf=-1e-15)
    #select index and values which satisfied the condition
    num, vs_num = [i for i, v in enumerate(vs) if v < 1e-15], [v for i,v in enumerate(vs) if v < 1e-15]
    
    if len(num) == 1: 
        U[:,num], V[:,num] = U[:,num].reshape(-1,1)*array(vs_num), V[:,num].reshape(-1,1)*array(vs_num)
    elif len(num) > 1: 
        U[:,num], V[:,num] = apply_along_axis(func1d=lambda x : x*vs_num,axis=1,arr=U[:,num]), apply_along_axis(func1d=lambda x : x*vs_num,axis=1,arr=V[:,num])

    #convert to namedtuple
    return namedtuple("gSVDResult",["vs","U","V","rank","ncp"])(vs,U,V,rank,ncp)