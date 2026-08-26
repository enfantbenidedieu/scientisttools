# -*- coding: utf-8 -*-
from numpy import linalg, cumsum,c_,nan
from collections import namedtuple
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..functions.cov2corr import cov2corr

def statsPCArot(obj):
    """
    Statistics with varimax in Principal Component Analysis

    Performs statistics with varimax in principal component analysis.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCArot`.

    Returns
    -------
    result : statPCArotResult
        A object with the following attributes

        corr_ : corr
            An object containing all the results for the correlation with the following attributes:  

            corrcoef: DataFrame of shape (n_columns, n_columns) 
                The pearson correlation coefficient matrix.
            pcorrcoef: DataFrame of shape (n_columns, n_columns) 
                The partial pearson correlation coefficient matrix
            reconst: DataFrame of shape (n_columns, n_columns) 
                The reconstitution pearson correlation coefficient matrix after rotation
            residual: DataFrame of shape (n_columns, n_columns) 
                The residual correlation matrix after rotation

        others_ : others
            An object with the following attributes:  

            vaccounted: DataFrame of shape (6, n_components)
                The variance accounted

            explained_variance: DataFrame of shape (n_components, 3)
                The explained variance.
                
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, PCArot, statsPCArot
    >>> clf = PCA()
    >>> clf.fit(decathlon.actif)
    PCA()
    >>> clfrot = PCArot()
    >>> clfrot.fit(clf)
    >>> stats = statsPCArot(clfrot)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class PCArot
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCArot": 
        raise TypeError("'obj' must be an object of class PCArot")

    #set number of rows and columns
    n_cols, ncp = obj.quanti_var_.coord.shape
    colnames = obj.quanti_var_.coord.index

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #correlation matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #covariance/correlation of Z and reconst
    M = (obj.call_.Z.T * obj.call_.ind_w).dot(obj.call_.Z)
    #inverse of M
    try: 
        inv_M = DataFrame(linalg.inv(M),index=colnames,columns=colnames)
    except linalg.LinAlgError: 
        inv_M = DataFrame(linalg.pinv(M),index=colnames,columns=colnames)
    #weighted partial correlation matrix and reconst
    partial_M = -1*cov2corr(inv_M)
    reconst_M = (obj.quanti_var_.coord.T * obj.call_.col_w).T.dot(obj.quanti_var_.coord.T)
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
    #variance accounted
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #sum of squared loadings
    ss_loadings = ((obj.quanti_var_.coord**2).T * obj.call_.col_w).sum(axis=1)
    #proportion
    prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
    #convert to DataFrame
    vaccounted = DataFrame(c_[obj.call_.obj.eig_.iloc[:ncp,0],ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],
                           index = [f"Dim{x+1}" for x in range(ncp)],
                           columns=["Eigenvalue","SS loadings","Proportion Var","Cumulative Var","Proportion Explained","Cumulative Proportion"]).T
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##others informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fidélité des facteurs - variance of the scores - R2
    r2_score = (obj.quanti_var_.coord * inv_M.dot(obj.quanti_var_.coord)).sum(axis=0)
    #Variance explained by each factor
    explained_variance = DataFrame(c_[obj.call_.obj.eig_.iloc[:obj.call_.ncp,0],ss_loadings,r2_score],
                                   index=[f"Dim{x+1}" for x in range(obj.call_.ncp)],
                                   columns=["Weighted","Unweighted","R2"])

    #convert to ordered dictionary
    others_ = {"vaccounted": vaccounted, "explained_variance": explained_variance}
    #convert to namedtuple
    others_ = namedtuple("others",others_.keys())(*others_.values())
    return namedtuple("statsPCArotResult",["corr_","others_"])(corr_,others_)