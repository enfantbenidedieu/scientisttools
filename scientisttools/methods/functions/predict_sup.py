# -*- coding: utf-8 -*-
from numpy import ones
from pandas import concat, DataFrame
from mapply.mapply import mapply
from collections import OrderedDict
from statsmodels.stats.weightstats import DescrStatsW

#intern functions
from .function_eta2 import function_eta2

#----------------------------------------------------------------------------------------------------------------------------------------
#predict supplementary rows/individuals
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_ind_sup(Z,V,sqdisto,col_weights=None,n_workers=1):
    """
    Predict supplementary individuals
    ---------------------------------

    Parameters
    ----------
    `Z`: a pandas DataFrame of standardized individuals data

    `V`: a 2D numpy array of the right matrix of generalized singular value decomposition (GSVD)

    `sqdisto`: pandas Series of supplementary individuals squared distance to origin

    `col_weights`: a pandas Series/ndarray/list of columns weights (by default None)

    `n_workers`: integer specifying maximum amount of workers (by default 1). See https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html

    Return(s)
    ---------
    an ordered dictionary of pandas DataFrames containing all the results for the supplementary individuals.
    
    `coord`: factor coordinates

    `cos2`: squared cosinus

    `dist2`: squared distance to origin

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com   
    """
    #set columns weights
    if col_weights is None:
        col_weights = ones(Z.shape[1])
    
    #factor coordinates
    coord = mapply(Z,lambda x : x*col_weights,axis=1,progressbar=False,n_workers=n_workers).dot(V)
    coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
    #squared cosine
    sqcos = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)
    return OrderedDict(coord=coord,cos2=sqcos,dist2=sqdisto)

#----------------------------------------------------------------------------------------------------------------------------------------
#predict supplementary quantitative variables
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_quanti_sup(X,row_coord,row_weights=None,n_workers=1):
    """
    Predict supplementary quantitative variables
    --------------------------------------------

    Parameters
    ----------
    `X`: a pandas DataFrame of quantitative variables

    `row_coord`: a pandas DataFrame of row factor coordinates

    `row_weights`: a pandas Series/ndarray/list of rows weights (by default None)

    `n_workers`: integer specifying maximum amount of workers (by default 1). See https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html

    Return(s)
    ---------
    an ordered dictionary of pandas DataFrames containing all the results for the supplementary quantitative variables.

    `coord`: factor coordinates

    `cos2`: squared cosinus

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #set row weights
    if row_weights is None:
        row_weights = ones(row_coord.shape[0])/row_coord.shape[0]
    
    #factor coordinates - factor correlation
    wcorr = DescrStatsW(concat((X,row_coord),axis=1),weights=row_weights,ddof=0).corrcoef[:X.shape[1],X.shape[1]:]
    coord = DataFrame(wcorr,index=X.columns,columns=["Dim."+str(x+1) for x in range(wcorr.shape[1])])
    #squared cosinus
    sqcos = mapply(coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers)
    return OrderedDict(coord=coord,cos2=sqcos)

#----------------------------------------------------------------------------------------------------------------------------------------
#predict supplementary qualitative variables
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_quali_sup(X,row_coord,coord,sqdisto,col_coef,row_weights=None,n_workers=1):
    """
    Predict supplementary qualitative variables
    -------------------------------------------

    Parameters
    ----------
    `X`: a pandas DataFrame of qualitative variables

    `row_coord`: a pandas DataFrame of rows factor coordinates

    `coord`: a pandas DataFrame of categories factor coordinates

    `sqdisto`: a pandas Series of categories squared distance to origin

    `col_coef`: a pandas Series of categories coefficients. Useful for categories value-test

    `row_weights`: a pandas Series/ndarray/list of rows weights (by default None)

    `n_workers`: integer specifying maximum amount of workers (by default 1). See https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html

    Retun(s)
    --------
    an ordered dictionary of pandas DataFrames containing all the results for the supplementary categories.

    `coord`: factor coordinates

    `cos2`: squared cosinus.

    `vtest`: value-test.

    `dist2`: squared distance to origin

    `eta2`: squared correlation ratio

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #set row weights
    if row_weights is None:
        row_weights = ones(row_coord.shape[0])/row_coord.shape[0]

    #value-test (vtest)
    vtest = mapply(coord,lambda x : x*col_coef,axis=0,progressbar=False,n_workers=n_workers)
    #squared cosinus (cos2)
    sqcos = mapply(coord, lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)
    #squared correlation ratio (eta2)
    sqeta = function_eta2(X=X,Y=row_coord,weights=row_weights,n_workers=n_workers)
    return OrderedDict(coord=coord,cos2=sqcos,vtest=vtest,dist2=sqdisto,eta2=sqeta)