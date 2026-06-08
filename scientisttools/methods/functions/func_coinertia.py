# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, sum,trace, diag, linalg
from pandas import Series

#interns functions
from .utils import convert_series_to_dataframe, check_is_dataframe

def func_coinertia(
        X,Y,xcol_w=None,ycol_w=None,row_w=None
):
    """
    Calculate the coinertia coefficients between two groups X and Y

    Parameters
    ----------
    X : Dataframe of shape (n_samples, n_xcolumns)
        First groups

    Y : Dataframe of shape (n_samples, n_ycolumns)
        Second group
        
    xcol_w : 1d array-like of shape (n_xcolumns,), default = None
        An optional variables weights for ``X``.

    ycol_w : 1d array-like of shape (n_ycolumns,), default = None
        An optional variables weights for ``Y``.

    row_w : 1d array-like of shape (n_samples,), default = None
        An optional individuals weights.

    Returns
    -------
    coin : float
        coinertia value.

    References
    ----------
    [1] A. B. Dufour, « `Coinertia Analysis <https://esb.univ-lyon1.fr/pdf/course6.pdf>`_ », Fiche TD avec le logiciel R, course6.

    [2] S. Dolédec and D. Chessel. « `Co-inertia analysis: an alternative method for studying species-environment relationships <https://www.researchgate.net/publication/228011497_Co-inertia_analysis_an_alternative_method_for_studying_species-environment_relationships>`_ ». Freshwater Biology, 31:277-294, 1994.

    [3] S. Dray, D. Chessel, and J. Thioulouse, « `Co-inertia analysis and the linking of ecological tables <https://pbil.univ-lyon1.fr/JTHome/Biblio/Drayetal2003.pdf>`_ », Ecology, 84(11):3078-3089, 2003
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X = convert_series_to_dataframe(X)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Y = convert_series_to_dataframe(Y)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if Y is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(Y)

    #check if len are equal
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of samples in X must be equal to the number of samples in Y")

    #set dimenstion
    n_samples, n_xcols, n_ycols = X.shape[0], X.shape[1], Y.shape[1]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set individuals and variables weigths
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set individuals weights
    if row_w is None:
        row_w = ones(n_samples)/n_samples
    elif not isinstance(row_w,(list,tuple,ndarray,Series)):
        raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
    elif len(row_w) != n_samples:
        raise ValueError(f"'row_w' must be a 1d array-like with length {n_samples}.")
    else:
        row_w = array(row_w)/sum(row_w)

    #set X variables weights
    if xcol_w is None:
        xcol_w = ones(n_xcols)
    elif not isinstance(xcol_w,(list,tuple,ndarray,Series)):
        raise TypeError("'xcol_w' must be a 1d array-like of Y-variables weights.")
    elif len(xcol_w) != n_xcols:
        raise ValueError(f"'xcol_w' must be a 1d array-like with length {n_xcols}.")
    else:
        xcol_w = array(xcol_w)
    
    #set Y variables weights
    if ycol_w is None:
        ycol_w = ones(n_ycols)
    elif not isinstance(ycol_w,(list,tuple,ndarray,Series)):
        raise TypeError("'ycol_w' must be a 1d array-like of Y-variables weights.")
    elif len(ycol_w) != n_ycols:
        raise ValueError(f"'ycol_w' must be a 1d array-like with length {n_ycols}.")
    else:
        ycol_w = array(ycol_w)
    return trace(linalg.multi_dot([X,diag(xcol_w),X.T,diag(row_w),Y,diag(ycol_w),Y.T,diag(row_w)]))