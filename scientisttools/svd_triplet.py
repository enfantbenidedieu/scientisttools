
import numpy as np

def svd_triplet(X,row_weights=None,col_weights=None,n_components=None):
    """
    Singular Value Decomposition of a Matrix
    ----------------------------------------

    Description
    -----------
    Compute the singular value decomposition of a rectangular matrix with weights for rows and columns

    Parameters
    ----------
    X : pandas DataFrame of float, shape (n_rows, n_columns)

    row_weights : array with the weights of each row (None by default and the weights are uniform)

    col_weights : array with the weights of each colum (None by default and the weights are uniform)

    n_components : the number of components kept for the outputs

    Return
    ------
    a dictionary containing
    vs : a vector containing the singular values of 'X'

    U : a matrix whose columns contain the left singular vectors of 'X'

    V : a matrix whose columns contain the right singular vectors of 'X'.
    
    See also
    --------
    See also https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html or https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    # Set row weights
    if row_weights is None:
        row_weights = np.ones(X.shape[0])/X.shape[0]
    else:
        row_weights = np.array([x/sum(row_weights) for x in row_weights])
    
    # Set columns weights
    if col_weights is None:
        col_weights = np.ones(X.shape[1])
    
    # Set number of components
    if n_components is None:
        n_components = min(X.shape[0] - 1, X.shape[1])
    else:
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])

    row_weights = row_weights.astype(float)
    col_weights = col_weights.astype(float)
    ################### Compute SVD using row weights and columns weights
    X = np.apply_along_axis(func1d=lambda x : x*np.sqrt(row_weights),axis=0,arr=X*np.sqrt(col_weights))

    ####### Singular Value Decomposition (SVD) ################################
    if X.shape[1] < X.shape[0]:
        # Singular Value Decomposition
        svd = np.linalg.svd(X,full_matrices=False)
        #### Extract U and V
        U = svd[0]
        V = svd[2].T

        if n_components > 1:
            # Find sign
            mult = np.sign(V.sum(axis=0))

            # Replace signe 0 by 1
            mult = np.array(list(map(lambda x: 1 if x == 0 else x, mult)))

            #####
            U = np.apply_along_axis(func1d=lambda x : x*mult,axis=1,arr=U)
            V = np.apply_along_axis(func1d=lambda x : x*mult,axis=1,arr=V)

        ### Recalibrate U and V using row weight and col weight
        U = np.apply_along_axis(func1d=lambda x : x/np.sqrt(row_weights),axis=0,arr=U)
        V = np.apply_along_axis(func1d=lambda x : x/np.sqrt(col_weights),axis=0,arr=V)
    else:
        # SVD to transpose X
        svd = np.linalg.svd(X.T,full_matrices=False)

        ##### Extract U and V
        U = svd[2].T
        V = svd[0]

        if n_components > 1:
            # Find sign
            mult = np.sign(V.sum(axis=0))

            # Replace signe 0 by 1
            mult = np.array(list(map(lambda x: 1 if x == 0 else x, mult)))

            #####
            U = np.apply_along_axis(func1d=lambda x : x*mult,axis=1,arr=U)
            V = np.apply_along_axis(func1d=lambda x : x*mult,axis=1,arr=V)

        ### Recalibrate U and V using row weight and col weight
        U = np.apply_along_axis(func1d=lambda x : x/np.sqrt(row_weights),axis=0,arr=U)
        V = np.apply_along_axis(func1d=lambda x : x/np.sqrt(col_weights),axis=0,arr=V)

    ######## Set number of columns using n_cp
    U = U[:,:n_components]
    V = V[:,:n_components]

    #### Set delta length
    vs = svd[1][:min(X.shape[1],X.shape[0]-1)]

    ################### 
    vs_filter = list(map(lambda x : True if x < 1e-15 else False, vs[:n_components]))

    # Select index which respect the criteria
    num = [idx for idx, i in enumerate(vs[:n_components]) if vs_filter[idx] == True]
    vs_num = [i for idx, i in enumerate(vs[:n_components]) if vs_filter[idx] == True]

    #######
    if len(num) == 1:
        U[:,num] = U[:,num].reshape(-1,1)*np.array(vs_num)
        V[:,num] = V[:,num].reshape(-1,1)*np.array(vs_num)
    elif len(num)>1:
        U[:,num] = np.apply_along_axis(func1d=lambda x : x*vs_num,axis=1,arr=U[:,num])
        V[:,num] = np.apply_along_axis(func1d=lambda x : x*vs_num,axis=1,arr=V[:,num])

    # Store Singular Value Decomposition (SVD) information
    return {"vs" : vs, "U" : U, "V" : V}