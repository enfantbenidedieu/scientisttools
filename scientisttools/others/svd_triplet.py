# -*- coding: utf-8 -*-
from numpy import ones, array,apply_along_axis,sqrt,linalg,sign
from collections import namedtuple

def svd_triplet(X,row_weights=None,col_weights=None,n_components=None):
    """
    Singular Value Decomposition of a Matrix
    ----------------------------------------

    Description
    -----------
    Compute the singular value decomposition of a rectangular matrix with weights for rows and columns

    Parameters
    ----------
    `X` : pandas DataFrame of float, shape (n_rows, n_columns)

    `row_weights` : array with the weights of each row (None by default and the weights are uniform)

    `col_weights` : array with the weights of each colum (None by default and the weights are uniform)

    `n_components` : the number of components kept for the outputs

    Return
    ------
    namedtuple of array containing
    `vs` : a vector containing the singular values of 'X'

    `U` : a matrix whose columns contain the left singular vectors of 'X'

    `V` : a matrix whose columns contain the right singular vectors of 'X'.
    
    See also
    --------
    See also https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html or https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #set dimensions
    n_rows, n_cols = X.shape

    # Set row weights
    if row_weights is None:
        row_weights = ones(n_rows)/n_rows
    else:
        row_weights = array([x/sum(row_weights) for x in row_weights])
    
    # Set columns weights
    if col_weights is None:
        col_weights = ones(n_cols)
    
    # Set number of components
    if n_components is None:
        n_components = min(n_rows-1, n_cols)
    else:
        n_components = min(n_components, n_rows-1, n_cols)

    row_weights, col_weights = row_weights.astype(float), col_weights.astype(float)
    
    X = apply_along_axis(func1d=lambda x : x*sqrt(row_weights),axis=0,arr=X*sqrt(col_weights))

    if X.shape[1] < X.shape[0]:
        svd = linalg.svd(X,full_matrices=False)
        U, V = svd[0], svd[2].T
    else:
        # SVD to transpose X
        svd = linalg.svd(X.T,full_matrices=False)
        U, V = svd[2].T, svd[0]

    if n_components > 1:
        # Find sign
        mult = sign(V.sum(axis=0))

        # Replace signe 0 by 1
        mult = array(list(map(lambda x: 1 if x == 0 else x, mult)))

        #####
        U, V = apply_along_axis(func1d=lambda x : x*mult,axis=1,arr=U), apply_along_axis(func1d=lambda x : x*mult,axis=1,arr=V)

    #Recalibrate U and V using row weight and col weight
    U, V = apply_along_axis(func1d=lambda x : x/sqrt(row_weights),axis=0,arr=U), apply_along_axis(func1d=lambda x : x/sqrt(col_weights),axis=0,arr=V)

    #Set number of columns using n_cp
    U, V = U[:,:n_components], V[:,:n_components]

    #Set delta length
    vs = svd[1][:min(n_rows-1,n_cols)]

    #
    vs_filter = list(map(lambda x : True if x < 1e-15 else False, vs[:n_components]))

    # Select index which respect the criteria
    num, vs_num = [idx for idx, i in enumerate(vs[:n_components]) if vs_filter[idx] == True], [i for idx, i in enumerate(vs[:n_components]) if vs_filter[idx] == True]

    #######
    if len(num) == 1:
        U[:,num] = U[:,num].reshape(-1,1)*array(vs_num)
        V[:,num] = V[:,num].reshape(-1,1)*array(vs_num)
    elif len(num)>1:
        U[:,num] = apply_along_axis(func1d=lambda x : x*vs_num,axis=1,arr=U[:,num])
        V[:,num] = apply_along_axis(func1d=lambda x : x*vs_num,axis=1,arr=V[:,num])

    # Store Singular Value Decomposition (SVD) information
    return namedtuple("svd_tripletResult",["vs","U","V"])(vs,U,V)