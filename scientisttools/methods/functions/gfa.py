# -*- coding: utf-8 -*-
from numpy import ones, c_, insert, diff, nan, cumsum
from pandas import DataFrame, Series
from collections import namedtuple

#intern functions
from .gsvd import gSVD

def gFA(X,
        ncp = 5,
        row_w = None,
        col_w = None, 
        tol = 1e-7):
    """
    General Factor Analysis (GFA)

    Performs general factor analysis analysis of a rectangular matrix with weights for rows and columns.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_w : 1d array-like of shape (n_samples,), default = None
        The rows weights.

    col_w : 1d array-like of shape (n_columns,), default = None
        The columns weights.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Returns
    -------
    result : gFAResult
        An object with the following attributes:

        svd : svd
            An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
            
            vs : 1d numpy array of shape (rank,)
                The singular values.
            d : 1d numpy array of shape (rank,)
                The eigen values.
            U : 2d numpy array of shape (n_columns, ncp)
                The left singular vectors.
            V : 2d numpy array of shape (n_samples, ncp)
                The right singular vectors.
            rank : int
                The maximum number of components.
            ncp : int
                The number of components kepted.
        
        eig : DataFrame of shape (rank, 4)
            The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

        row : dict
            An dictionary with the following keys:

            coord : DataFrame of shape (n_samples, ncp) 
                The coordinates for the rows,
            cos2 : DataFrame of shape (n_samples, ncp)
                The squared cosinus for the rows,
            contrib : DataFrame of shape (n_samples, ncp)
                The relative contributions for the rows,
            infos : DataFrame of shape (n_samples, 4)
                Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the rows.

        col : dict
            An dictionary with the following keys:

            coord : DataFrame of shape (n_columns, ncp)
                The coordinates for the columns.
            cos2 : DataFrame of shape (n_columns, ncp)
                The squared cosinus for the columns.
            contrib : DataFrame of shape (n_columns, ncp)
                The relative contributions for the columns.
            infos : DataFrame of shape (n_columns, 4)
                Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the columns.

    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod
    
    [2] Husson, F., Le, S. and Pages, J. (2009). Analyse de donnees avec R, Presses Universitaires de Rennes.

    [3] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    [4] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    [5] Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    [6] Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0, `https://hal.science/hal-04868625v1 <https://hal.science/hal-04868625v1>`_
    """
    # set number of rows and columns
    n_rows, n_cols = X.shape
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set rows and columns weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set rows weights
    if row_w is None: 
        row_w = Series(ones(n_rows)/n_rows,index=X.index,name="Weight")
    # set columns weights
    if col_w is None: 
        col_w = Series(ones(n_cols),index=X.columns,name="Weight")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # generalized singular values decomposition (GSVD)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    svd = gSVD(X=X,ncp=ncp,row_w=row_w,col_w=col_w,tol=tol)
    # reset number of components
    ncp = svd.ncp
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #eigen values informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    eigvals = svd.d
    difference, proportion = insert(-diff(eigvals),svd.rank-1,nan), 100*eigvals/sum(eigvals)
    #convert to DataFrame
    eig = DataFrame(c_[eigvals,difference,proportion,cumsum(proportion)],
                    columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],
                    index = [f"Dim{x+1}" for x in range(svd.rank)])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #rows informations: weights, squared distance to origin, inertia, percentage of inertia, coordinates, contributions and squared cosinus
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #dist2 of the rows
    row_sqdisto = ((X**2)*col_w).sum(axis=1)
    #inertia of the rows
    row_inertia = row_sqdisto*row_w
    #percentage of inertia of the rows
    row_inertia_pct = 100*row_inertia/sum(row_inertia)
    #convert to DataFrame
    row_infos = DataFrame(c_[row_w,row_sqdisto,row_inertia,row_inertia_pct],
                          columns=["Weight","Sq. Dist.","Inertia","Inertia (%)"],
                          index=X.index)
    #coordinates of the rows
    row_coord = DataFrame(svd.U[:,:ncp]*svd.vs[:ncp],index=X.index,columns=eig.index[:ncp])
    #contributions of the rows
    row_ctr = 100*((row_coord**2).T * row_w).T/eigvals[:ncp]
    #cos2 of the rows
    row_sqcos = ((row_coord**2).T/row_sqdisto).T
    #convert to dictionary
    row = {"coord":row_coord,"cos2":row_sqcos,"contrib":row_ctr,"infos":row_infos}
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #columns informations : weights, squared distance to origin, inertia, percentage of inertia, coordinates, contributions and squared cosinus
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #dist2 of the columns
    col_sqdisto = ((X**2).T*row_w).sum(axis=1)
    #inertia of the columns
    col_inertia = col_sqdisto*col_w
    #percentage of inertia of the columns
    col_inertia_pct = 100*col_inertia/sum(col_inertia)
    #convert to DataFrame
    col_infos = DataFrame(c_[col_w,col_sqdisto,col_inertia,col_inertia_pct],
                          columns=["Weight","Sq. Dist.","Inertia","Inertia (%)"],
                          index=X.columns)
    #coordinates of the columns
    col_coord = DataFrame(svd.V[:,:ncp]*svd.vs[:ncp],index=X.columns,columns=eig.index[:ncp])
    #contributions of the columns
    col_ctr = 100*((col_coord**2).T* col_w).T/eigvals[:ncp]
    #cos2 of the columns
    col_sqcos = ((col_coord**2).T/col_sqdisto).T
    #convert to dictionary
    col = {"coord":col_coord,"cos2":col_sqcos,"contrib":col_ctr,"infos":col_infos}
    
    # convert to dictionary
    gfa_ = {"svd":svd,"eig":eig,"row":row,"col":col}
    # convert to namedtuple
    return namedtuple("gFAResult",gfa_.keys())(*gfa_.values())