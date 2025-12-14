# -*- coding: utf-8 -*-
from numpy import ones, c_, insert, diff, nan, cumsum, diag
from pandas import DataFrame, Series
from collections import namedtuple, OrderedDict
from typing import NamedTuple

#intern functions
from .gsvd import gsvd

def gfa(X:DataFrame,row_weights:Series|None,col_weights:Series|None,max_components:int,n_components:int) -> NamedTuple:
    """
    Generalized Factor Analysis (GFA)
    ---------------------------------

    Description
    -----------
    performs generalized factor analysis of a rectangular matrix with weights for rows and columns

    Usage
    -----
    ```
    >>> gfa(X,row_weights,col_weights,max_components,n_components)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame, of shape (n_rows, n_columns)

    `row_weights`: None or a pandas Series with the weights of each row (None by default and the weights are uniform)

    `col_weights`: None or a pandas Series with the weights of each colum (None by default and the weights are uniform)

    `max_components`: an integer indicating the maximum number of dimensions

    `n_components`: an integer indicating indicating the number of dimensions kept in the results

    Returns
    -------
    a namedtuple of namedtuple/pandas DataFrame, including:

    `svd`: a namedtuple of numpy array containing all the results for the generalized singular value decomposition (GSVD), including:
        * `vs`: 1D numpy array containing the singular values,
        * `U`: 2D numpy array whose columns contain the left singular vectors,
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig`: a pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `row`: a namedtuple of pandas DataFrames containing all the results for the rows, including:
        * `coord`: coordinates of the rows,
        * `cos2`: squared cosinus of the rows,
        * `contrib`: relative contributions of the rows,
        * `infos`: additionals informations (weight, margin, squared distance to origin, inertia and percentage of inertia) of the rows.
    
    `col`: a namedtuple of pandas DataFrames containing all the results for the columns, including:
        * `coord`: coordinates of the columns,
        * `cos2`: squared cosinus of the columns,
        * `contrib`: relative contributions of the columns,
        * `infos`: additionals informations (margin, squared distance to origin, inertia and percentage of inertia) of the columns.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod
    
    * Husson, F., Le, S. and Pages, J. (2009). Analyse de donnees avec R, Presses Universitaires de Rennes.

    * Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    * Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0
    """
    #shape of dataframe
    n_rows, n_cols = X.shape

    #set rows weights
    if row_weights is None:
        row_weights = Series(ones(n_rows)/n_rows,index=X.index,name="Weight")

    #set columns weights
    if col_weights is None:
        col_weights = Series(ones(n_cols),index=X.columns,name="Weight")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #generalized singular values decomposition (GSVD)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    svd = gsvd(X=X,row_weights=row_weights,col_weights=col_weights,n_components=n_components)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #eigen values informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    eigen_values = svd.vs[:max_components]**2
    difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values)
    #convert to DataFrame
    eig = DataFrame(c_[eigen_values,difference,proportion,cumsum(proportion)],columns=["Eigenvalue","Difference","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(max_components)])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #rows informations: weights, squared distance to origin, inertia, percentage of inertia, coordinates, contributions and squared cosinus
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #dist2 of the rows
    row_sqdisto = X.pow(2).mul(col_weights,axis=1).sum(axis=1)
    #inertia of the rows
    row_inertia = row_sqdisto*row_weights
    #percentage of inertia of the rows
    row_inertia_pct = 100*row_inertia/sum(row_inertia)
    #convert to DataFrame
    row_infos = DataFrame(c_[row_weights,row_sqdisto,row_inertia,row_inertia_pct],columns=["Weight","Sq. Dist.","Inertia","% Inertia"],index=X.index)
    #coordinates of the rows
    row_coord = DataFrame(svd.U.dot(diag(svd.vs[:n_components])),index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])
    #contributions of the rows
    row_ctr = row_coord.pow(2).mul(100).mul(row_weights,axis=0).div(eigen_values[:n_components],axis=1)
    #cos2 of the rows
    row_sqcos = row_coord.pow(2).div(row_sqdisto,axis=0)
    #convert to ordered dictionary
    row = OrderedDict(coord=row_coord,cos2=row_sqcos,contrib=row_ctr,infos=row_infos)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #columns informations : weights, squared distance to origin, inertia, percentage of inertia, coordinates, contributions and squared cosinus
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #dist2 of the columns
    col_sqdisto = X.pow(2).mul(row_weights,axis=0).sum(axis=0)
    #inertia of the columns
    col_inertia = col_sqdisto*col_weights
    #percentage of inertia of the columns
    col_inertia_pct = 100*col_inertia/sum(col_inertia)
    #convert to DataFrame
    col_infos = DataFrame(c_[col_weights,col_sqdisto,col_inertia,col_inertia_pct],columns=["Weight","Sq. Dist.","Inertia","% Inertia"],index=X.columns)
    #coordinates of the columns
    col_coord = DataFrame(svd.V.dot(diag(svd.vs[:n_components])),index=X.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
    #contributions of the columns
    col_ctr = col_coord.pow(2).mul(100).mul(col_weights,axis=0).div(eigen_values[:n_components],axis=1)
    #cos2 of the columns
    col_sqcos = col_coord.pow(2).div(col_sqdisto,axis=0)
    #convert to ordered dictionary
    col = OrderedDict(coord=col_coord,cos2=col_sqcos,contrib=col_ctr,infos=col_infos)

    #convert to namedtuple
    return namedtuple("gfaResult",["svd","eig","row","col"])(svd,eig,row,col)