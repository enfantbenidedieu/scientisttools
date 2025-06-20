# -*- coding: utf-8 -*-
from numpy import ones, dot
from mapply.mapply import mapply
from collections import OrderedDict

#intern functions
from .function_eta2 import function_eta2

#----------------------------------------------------------------------------------------------------------------------------------------
##predict supplementary individuals
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_ind_sup(Z,V,sqdisto,col_weights=None,n_workers=1):
    """
    Predict supplementary individuals
    ---------------------------------

    Parameters
    ----------
    `Z` :  pandas dataframe of standardize individuals data

    `V` : right matrix of generalized singular value decomposition

    `sqdisto` : pandas series of supplementary individuals square distance to origin

    `col_weights` : array of columns weights (by default None)

    `n_workers` : integer specifying maximum amount of workers (by default 1). See https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html

    Return(s)
    ---------
    ordered dictionary containing:

    `coord` : pandas dataframe of supplementary individuals factor coordinates

    `cos2` : pandas dataframe of supplementary individuals square cosinus

    `dist` : pandas dataframe of supplementary individuals square distance to origin

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com    

    """
    #set columns weights
    if col_weights is None:
        col_weights = ones(Z.shape[1])
    
    # Factor coordinates
    coord = mapply(Z,lambda x : x*col_weights,axis=1,progressbar=False,n_workers=n_workers).dot(V)
    coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]

    #Square cosine
    sqcos = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)
    
    return OrderedDict(coord=coord,cos2=sqcos,dist=sqdisto)

#----------------------------------------------------------------------------------------------------------------------------------------
##predict supplementary quantitative variables
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_quanti_sup(Z,U,row_weights=None,n_workers=1):
    """
    Predict supplementary quantitative variables
    --------------------------------------------

    Parameters
    ----------
    `Z` :  pandas dataframe of standardize quantitative variables

    `U` : left matrix of generalized singular value decomposition

    `row_weights` : array of rows weights (by default None)

    `n_workers` : integer specifying maximum amount of workers (by default 1). See https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html

    Return(s)
    ---------
    ordered dictionary containing:

    `coord` : pandas dataframe of supplementary quantitative variables factor coordinates

    `cos2` : pandas dataframe of supplementary quantitative variables square cosinus


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #set row weights
    if row_weights is None:
        row_weights = ones(Z.shape[0])/Z.shape[0]

    # Factor coordinates
    coord = mapply(Z,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(U)
    coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]

    # Square distance to origin
    sqdisto = dot(ones(Z.shape[0]),mapply(Z,lambda x : (x**2)*row_weights,axis=0,progressbar=False,n_workers=n_workers))

    # Square cosine
    sqcos = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)  
    
    return OrderedDict(coord=coord,cos2=sqcos)

#----------------------------------------------------------------------------------------------------------------------------------------
##predict supplementary qualitative variables
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_quali_sup(X,Z,Y,V,col_coef,sqdisto,row_weights=None,col_weights=None,n_workers=1):
    """
    Predict supplementary qualitative variables
    -------------------------------------------

    Parameters
    ----------
    `X` : pandas dataframe of qualitative variables

    `Z` : pandas dataframe of standardize categorical variables

    `Y` : pandas dataframe of rows factor coordinates

    `V` : right matrix of generalized singular value decomposition

    `col_coef` : pandas series of categories coefficients. Useful for categories value-test

    `sqdisto` : pandas series of categories square distance to origin

    `row_weights` : ndarray of rows weights (by default None)

    `col_weights` : ndarray of columns weights (by default None)

    `n_workers` : integer specifying maximum amount of workers (by default 1). See https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html

    Retun(s)
    --------
    ordered dictionary containing:

    `coord` : pandas dataframe of categories factor coordinates

    `cos2` : pandas dataframe of categories square cosinus.

    `vtest` : pandas dataframe of categories value-test.

    `dist` : pandas series of categories square distance to origin

    `eta2` : pandas dataframe of qualitative variables square correlation ratio

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    
    """
    #set row weights
    if row_weights is None:
        row_weights = ones(Z.shape[0])/Z.shape[0]
    
    #set columns weights
    if col_weights is None:
        col_weights = ones(Z.shape[1])

    #factor coordinates
    coord = mapply(Z, lambda x : x*col_weights,axis=1,progressbar=False,n_workers=n_workers).dot(V)
    coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]

    #value-test (vtest)
    vtest = mapply(coord,lambda x : x*col_coef,axis=0,progressbar=False,n_workers=n_workers)

    #square cosinus (cos2)
    sqcos = mapply(coord, lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)

    #square correlation ratio (eta2)
    sqeta = function_eta2(X=X,Y=Y,weights=row_weights,n_workers=n_workers)

    return OrderedDict(coord=coord,cos2=sqcos,vtest=vtest,dist=sqdisto,eta2=sqeta)