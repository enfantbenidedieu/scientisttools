# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, concat
from numpy import array, ndarray, ones, diag,linalg, insert, diff, nan, cumsum, c_, sqrt
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# interns functions
from ..functions.cov2corr import cov2corr

def statsFA(obj):
    """
    Statistics with Factor Analysis

    Performs statistics with Factor Analysis (FA).

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`.

    Returns
    -------
    result : statsFAResult
        An object with the following attributes:

        corr : corr
            An object with the following attributes:
            
            corrcoef : DataFrame of shape (n_columns, n_columns)
                Pearson correlation coefficients.
            pcorrcoef : DataFrame of shape (n_columns, n_columns)
                partial pearson correlation coefficients.
            reconst : DataFrame of shape (n_columns, n_columns)
                Correlation matrix estimated by the model.
            residual : DataFrame of shape (n_columns, n_columns)
                Residual correlations after the factor model is applied.
            
        others : others
            An object with the following attributes:

            vaccounted : DataFrame of shape (7, ncp)
                Variance acconted.
            explained_variance : DataFrame of shape (ncp, 3)
                Variance explained by each factor (weighted, unweighted) and R2, which is the multiple R-square between the factors and factor score estimates.
            communalities : float
                The communalities reflecting the total amount of common variance. They will exceed the communality (above) which is the model estimated common variance.
            inertia : float
                The total inertia.

    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, statsFA
    >>> clf = FA(ncp=2,warn_message=False)
    >>> clf.fit(beer)
    FA(ncp=2,warn_message=False)
    >>> stats = statsFA(clf)
    ... ("corr","others")    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class FA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("FA","FArot")): 
        raise TypeError("'obj' must be an object of class FA, FArot")

    #set number of columns and number of components kepted
    n_cols, ncp = obj.quanti_var_.coord.shape
    colnames = obj.quanti_var_.coord.index

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #correlation matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #inverse of M
    try: 
        inv_M = DataFrame(linalg.inv(obj.call_.M),index=colnames,columns=colnames)
    except linalg.LinAlgError: 
        inv_M = DataFrame(linalg.pinv(obj.call_.M),index=colnames,columns=colnames)
    #weighted partial correlation matrix and reconst covariance/correlation
    partial_M, reconst_M = -1*cov2corr(inv_M), obj.quanti_var_.coord.mul(obj.call_.col_w,axis=0).dot(obj.quanti_var_.coord.T)
    for c in partial_M.columns:
        partial_M.loc[c,c] = 1
    #residual covariance/correlation
    resid_M = obj.call_.M - reconst_M.values
    for c in resid_M.columns:
        resid_M.loc[c,c] = nan
    #convert to ordered dictionary
    corr_ = { "corrcoef": obj.call_.M, "pcorrcoef": partial_M, "reconst": reconst_M, "residual": resid_M }
    #convert to namedtuple
    corr_ = namedtuple("corr",corr_.keys())(*corr_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #variance accounted
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #eigen value decomposition of the original matrix
    eigvals = linalg.eigh(obj.call_.M)[0][::-1]
    #sum of squared loadings - common eigen values
    ss_loadings = ((obj.quanti_var_.coord**2).T * obj.call_.col_w).T.sum(axis=0)
    #proportion
    prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
    #convert to pd.DataFrame
    vaccounted = DataFrame(c_[eigvals[:ncp],obj.eig_.iloc[:ncp,0],ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],
                           index = [f"Dim{x+1}" for x in range(ncp)],
                           columns=["Original","Common","SS loadings","Proportion Var (%)","Cumulative Var (%)",
                                    "Proportion Explained (%)","Cumulative Proportion (%)"]).T
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##others statistics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fidélité des facteurs - variance of the scores - R2
    r2_score = (obj.quanti_var_.coord * obj.coef_.values).sum(axis=0)
    #Variance explained by each factor
    explained_variance = DataFrame(c_[obj.eig_.iloc[:ncp,0],ss_loadings,r2_score],
                                   index=[f"Dim{x+1}" for x in range(ncp)],
                                   columns=["Weighted","Unweighted","R2"])
    #total inertia and communalities
    inertia, communalities = sum(obj.quanti_var_.infos.iloc[:,1]), sum(obj.quanti_var_.infos.iloc[:,2])
    #convert to ordered dictionary
    others_ = { "vaccounted": vaccounted, "explained_variance": explained_variance, "communalities": communalities, "inertia": inertia }
    #convert to namedtuple
    others_ = namedtuple("others",others_.keys())(*others_.values())
    return namedtuple("statsFAResult",["corr_","others_"])(corr_,others_)