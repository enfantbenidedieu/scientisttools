# -*- coding: utf-8 -*-
from numpy import ones, c_, insert, diff, nan, cumsum, diag
from pandas import DataFrame
from mapply.mapply import mapply
from collections import namedtuple, OrderedDict

# intern functions
from .svd_triplet import svd_triplet

def fitfa(Z,row_weights,col_weights,max_components,n_components,n_workers):

    # shape of dataframe
    n_rows, n_cols = Z.shape

    # Set rows weights
    if row_weights is None:
        row_weights = ones(n_rows)/n_rows

    # Set columns weights
    if col_weights is None:
        col_weights = ones(n_cols)

    #--------------------------------------------------------------------------------------------
    ##Rows informations: weights, squared distance to origin, inertia and percentage of inertia
    #--------------------------------------------------------------------------------------------
    #row squared distance to origin
    row_sqdisto = mapply(Z,lambda x : (x**2)*col_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    #row inertia
    row_inertia = row_sqdisto*row_weights
    #row percentage of inertia
    row_inertia_pct = 100*row_inertia/sum(row_inertia)
    #convert to DataFrame
    row_infos = DataFrame(c_[row_weights,row_sqdisto,row_inertia,row_inertia_pct],columns=["Weight","Sq. Dist.","Inertia","% Inertia"],index=Z.index)
    
    #------------------------------------------------------------------------------------------------------------------------
    ##Columns informations : weights, squared distance to origin, inertia and percentage of inertia
    #-----------------------------------------------------------------------------------------------------------------------
    #columns squared distance to origin
    col_sqdisto = mapply(Z,lambda x : (x**2)*row_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
    #columns inertia
    col_inertia = col_sqdisto*col_weights
    #columns percentage of inertia
    col_inertia_pct = 100*col_inertia/sum(col_inertia)
    #convert to DataFrame
    col_infos = DataFrame(c_[col_weights,col_sqdisto,col_inertia,col_inertia_pct],columns=["Weight","Sq. Dist.","Inertia","% Inertia"],index=Z.columns)
    
    #-------------------------------------------------------------------------------------------------------
    ## Generalized Singular Value Decomposition (GSVD)
    #-------------------------------------------------------------------------------------------------------
    svd = svd_triplet(X=Z,row_weights=row_weights,col_weights=col_weights,n_components=n_components)
    
    #----------------------------------------------------------------------------------------------------------
    ##Eigen values informations
    #-------------------------------------------------------------------------------------------------------
    eigen_values = svd.vs[:max_components]**2
    difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values)
    # store all informations
    eig = DataFrame(c_[eigen_values,difference,proportion,cumsum(proportion)],columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(max_components)])

    #-----------------------------------------------------------------------------------------------------------
    ##row informations : factor coordinates, contributions and squared Cosinus
    #-----------------------------------------------------------------------------------------------------------
    #rows factor coordinates
    row_coord = DataFrame(svd.U.dot(diag(svd.vs[:n_components])),index=Z.index,columns=["Dim."+str(x+1) for x in range(n_components)])
    #rows contributions
    row_ctr = mapply(mapply(row_coord,lambda x : 100*(x**2)*row_weights,axis=0,progressbar=False,n_workers=n_workers),lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
    #rows square cosine
    row_cos2 = mapply(row_coord,lambda x : (x**2)/row_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
    #convert to ordered dictionary
    row = OrderedDict(zip(["coord","cos2","contrib","infos"],[row_coord,row_cos2,row_ctr,row_infos]))

    #---------------------------------------------------------------------------------------------------------------
    ##columns informations : factor coordinates, contributions and squared cosinus
    #---------------------------------------------------------------------------------------------------------------
    #columns factor coordinates
    col_coord = DataFrame(svd.V.dot(diag(svd.vs[:n_components])),index=Z.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
    #columns contributions
    col_ctr = mapply(mapply(col_coord,lambda x : 100*(x**2)*col_weights,axis=0,progressbar=False,n_workers=n_workers), lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
    #columns square cosine
    col_cos2 = mapply(col_coord,  lambda x : (x**2)/col_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
    #convert to ordered dictionary
    col = OrderedDict(zip(["coord","cos2","contrib","infos"],[col_coord,col_cos2,col_ctr,col_infos]))

    return namedtuple("fitfaResult",["svd","eig","row","col"])(svd,eig,row,col)