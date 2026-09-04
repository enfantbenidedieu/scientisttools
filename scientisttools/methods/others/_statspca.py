# -*- coding: utf-8 -*-
from numpy import sqrt, linalg, log, cumsum,mean, nan
from pandas import DataFrame, Series, concat
from scipy.stats import chi2
from collections import namedtuple
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.statistics import wcorr
from ..functions.cov2corr import cov2corr
from ..others._kaisermsa import kaisermsa

def statsPCA(obj):
    """
    Statistics with Principal Component Analysis

    Performs statistics with principal component analysis (PCA).

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`.

    Returns
    -------
    result : statPCAResult
        A object with the following attributes

        corr : corr
            An object containing all the results for the correlation with the following attributes:  

            corrcoef: DataFrame of shape (n_columns, n_columns) 
                The pearson correlation coefficient matrix.
            pcorrcoef: DataFrame of shape (n_columns, n_columns) 
                The partial pearson correlation coefficient matrix
            reconst: DataFrame of shape (n_columns, n_columns) 
                The reconstitution pearson correlation coefficient matrix
            residual: DataFrame of shape (n_columns, n_columns) 
                The residual correlation matrix

        others : others
            An object with the following attributes:

            threshold : DataFrame of shape (1,2)
                Eigen values threshold: kaiser, kaiser proportion and KSS (Karlis - Saporta - Spinaki).
            bartlett: DataFrame of shape (1,4)
                The Bartlett's test of Spericity.
            broken: Series of shape (rank, 2)
                The broken's stick threshold.
            msa: Series of shape (n_columns + 1,)
                The Kaiser measure of sampling adequacy.
                
    References
    ----------
    [1] Ricco Rakotomalala. Analyse de corrélation. 2025. ⟨`hal-05066618 <https://hal.science/hal-05066618v1>`_⟩
    
    [1] Ricco Rakotomalala. Pratique des Méthodes Factorielles avec Python. 2020. `hal-04868625 <https://hal.science/hal-04868625>`_.
                
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, statsPCA
    >>> # run PCA
    >>> clf = PCA()
    >>> clf.fit(decathlon.actif)
    PCA()
    >>> # statistics with PCA
    >>> stats = statsPCA(clf)
    >>> stats._fields
    ... ("corr","others")
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class PCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCA":
        raise TypeError("'obj' must be an object of class PCA")

    #set number of rows and columns
    n_rows, n_cols = obj.call_.X.shape
    maxncp, colnames = obj.eig_.shape[0], obj.call_.Z.columns

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #correlation matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #covariance/correlation of Z and reconst
    M = wcorr(obj.call_.Z,w=obj.call_.row_w,ddof=0)
    #inverse of M
    try: 
        inv_M = DataFrame(linalg.inv(M),index=colnames,columns=colnames)
    except linalg.LinAlgError: 
        inv_M = DataFrame(linalg.pinv(M,hermitian=True),index=colnames,columns=colnames)
    #weighted partial correlation matrix and reconst covariance/correlation
    partial_M, reconst_M = -1*cov2corr(inv_M), (obj.quanti_var_.coord.T * obj.call_.col_w).T.dot(obj.quanti_var_.coord.T)
    for c in partial_M.columns:
        partial_M.loc[c,c] = 1
    #residual covariance/correlation
    resid_M = M - reconst_M.to_numpy()
    for c in resid_M.columns:
        resid_M.loc[c,c] = nan
    #convert to ordered dictionary
    corr_ = {"corrcoef": M, "pcorrcoef": partial_M, "reconst": reconst_M, "resid": resid_M}
    #convert to namedtuple
    corr_ = namedtuple("corr",corr_.keys())(*corr_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #bartlett's test
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Bartlett - statistics
    bartlett_stats, bs_dof = -(n_rows-1-(2*n_cols+5)/6)*sum(log(obj.eig_.iloc[:,0])), n_cols*(n_cols-1)/2
    bs_pvalue = chi2.sf(bartlett_stats,df=bs_dof)
    bartlett = DataFrame([[linalg.det(M),bartlett_stats,bs_dof,bs_pvalue]],columns=["|CORR.MATRIX|","CHISQ","dof","p-value"],index=["Bartlett's test"])
    bartlett["dof"] = bartlett["dof"].astype(int)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #others informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Karlis - Saporta - Spinaki threshold
    kss_th =  1 + 2*sqrt((maxncp-1)/(n_rows-1))
    #eigen value threshold
    eig_th = DataFrame([[mean(obj.eig_.iloc[:,0]),kss_th]],columns=["Kaiser-Guttman","Karlis-Saporta-Spinaki"],index=["Critical values"])
    
    #broken-stick crticial values
    broken = Series(cumsum([1/x for x in range(maxncp,0,-1)])[::-1],name="Broken-stick critical values",index=[f"Dim{x+1}" for x in range(maxncp)])
    broken = concat((obj.eig_.iloc[:,0],broken),axis=1)
    #convert to ordered dictionary
    others_ = {"threshold": eig_th, "bartlett": bartlett, "broken": broken, "msa": kaisermsa(X=obj.call_.X,w=obj.call_.ind_w)}
    #convert to namedtuple
    others_ = namedtuple("others",others_.keys())(*others_.values())
    return namedtuple("statsPCAResult",["corr","others"])(corr_,others_)