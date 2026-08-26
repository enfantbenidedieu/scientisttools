# -*- coding: utf-8 -*-
from numpy import array, repeat, ones, ndarray, c_, cumsum, sqrt
from pandas import CategoricalDtype, DataFrame, Series, concat
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

def statsMCA(obj ):
    """
    Statistics with Multiple Correspondence Analysis

    Performs statistics with multiple correspondence analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    Returns
    -------
    result : statMCAResult
        A object with the following attributes

        correction_ : correction
            An object containing eigenvalues correction, with the following attributes:

            benzecri : DataFrame of shape (..., 3)
                The benzecri correction.
            greenacre : DataFrame of shape (..., 3)
                The greenacre correction.

        others_ : others
            An object of others statistics, with the following attributes:

            inertia : float
                The global multiple correspondence analysis inertia.
            kaiser : DataFrame of shape (1,2)
                The kaiser threshold.

    References
    ----------
    [1] Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, statsMCA
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison.data)
    MCA(sup_var=range(4))
    >>> #statistics with multiple correspondence analysis
    >>> stats = statsMCA(clf)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class MCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MCA": 
        raise TypeError("'obj' must be an object of class MCA")

    #set number of categorical variables and number of levels
    n_cols, n_levels = obj.quali_var_.coord.shape[0], obj.levels_.coord.shape[0]

    #save eigen value grather than threshold
    kaiser_threshold = 1/n_cols
    lambd = obj.eig_.iloc[:,0][obj.eig_.iloc[:,0]>kaiser_threshold]

    #benzecri correction
    lambd_tilde = ((n_cols/(n_cols-1))*(lambd - kaiser_threshold))**2
    s_tilde = 100*(lambd_tilde/sum(lambd_tilde))
    benzecri = DataFrame(c_[lambd_tilde,s_tilde,cumsum(s_tilde)],
                         columns=["Eigenvalue","Proportion","Cumulative"],
                         index = [f"Dim{x+1}" for x in range(len(lambd))])
    #greenacre correction
    s_tilde_tilde = n_cols/(n_cols-1)*(sum(obj.eig_.iloc[:,0]**2)-(n_levels - n_cols)/(n_cols**2))
    tau = 100*(lambd_tilde/s_tilde_tilde)
    greenacre = DataFrame(c_[lambd_tilde,tau,cumsum(tau)],
                          columns=["Eigenvalue","Proportion","Cumulative"],
                          index = [f"Dim{x+1}" for x in range(len(lambd))])
    #convert to namedtuple
    correction_ = namedtuple("correction",["benzecri","greenacre"])(benzecri,greenacre)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #multiple correspondence analysis additionals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #inertia
    inertia = (n_levels/n_cols) - 1
    #eigenvalue threshold
    kaiser_proportion_threshold = 100/inertia
    #eigen value threshold
    kaiser = DataFrame([[kaiser_threshold,kaiser_proportion_threshold]],
                       columns=["threshold","proportion"],
                       index=["Kaiser critical values"])
    #convert to namedtuple
    others_ = namedtuple("others",["inertia","kaiser"])(inertia,kaiser)
    return namedtuple("statsMCAResult",["correction_","others_"])(correction_,others_)