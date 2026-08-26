# -*- coding: utf-8 -*-
from numpy import sqrt
from scipy.stats import chi2_contingency
from pandas import DataFrame
from collections import namedtuple
from sklearn.utils.validation import check_is_fitted

def statsCA(obj):
    """
    Statistics with Correspondence Analysis

    Performs statistics with correspondence analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    Returns
    -------
    result : statCAResult
        A object with the following attributes:

        goodness_ : goodness
            An object with the following attributes:

            test : DataFrame of shape (2, 3)
                The pearson's chi-squared test and The log-likelihood ratio (i.e the "G-test").    
            association : DataFrame of shape (6, 2)
                The degree of association between two nominal variables ("cramer", "tschuprow", "pearson").

        residual_ : residual
            An object with the following attributes:

            resid : DataFrame of shape (n_rows, n_columns) 
                The model residuals.
            resid_std : DataFrame of shape (n_rows, n_columns) 
                The standardized residuals.
            resid_adj : DataFrame of shape (n_rows, n_columns) 
                The adjusted residuals.
            contrib : DataFrame of shape (n_rows, n_columns) 
                The contribution to chi-squared.
            att_rep_ind : DataFrame of shape (n_rows, n_columns)  
                The attraction repulsion index.

        kaiser_ : DataFrame of shape (1,2)
            The kaiser threshold.

    References
    ----------
    [1] Ricco Rakotomalala. `Etude des dépendances - Variables qualitatives <https://hal.science/hal-05110267v1>`_. 2025.
    
    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, statsCA
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children.data)
    CA(col_sup=(5,6,7),row_sup=range(14,18),sup_var=8)
    >>> # statistics with correspondence analysis
    >>> stats = statsCA(clf)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class CA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CA":
        raise TypeError("obj must be an object of class CA")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #diagnostics tests - multivariate goodness of fit tests
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set dimensions
    n_rows, n_cols = obj.call_.X.shape
    #chi-squared statistics
    chi2_stat, chi2_pvalue, dof, expected_freq = chi2_contingency(obj.call_.X,lambda_=None,correction=False)
    #log - likelihood test (G - test)
    g_stat, g_pvalue = chi2_contingency(obj.call_.X, lambda_="log-likelihood")[:2]
    #convert to DataFrame
    test = DataFrame([[chi2_stat,dof,chi2_pvalue],[g_stat,dof,g_pvalue]],columns=["statistic","dof","pvalue"],index=["Pearson's Chi-Square Test","log-likelihood (G-test)"])
    #association test
    phi2, phi_max, chi2_max = chi2_stat/obj.call_.total, sqrt(min(n_rows - 1, n_cols - 1)), obj.call_.total*min(n_rows - 1, n_cols - 1)
    #Cramer's V
    cramer_v = sqrt(phi2 / min(n_cols - 1, n_rows - 1))
    #Tschuprow's T
    tschuprow_t, tschuprow_max = sqrt(phi2 / sqrt((n_rows - 1) * (n_cols - 1))), (min(n_rows - 1, n_cols - 1)/max(n_rows - 1, n_cols - 1))**(1/4)
    #Pearson's C
    pearson_c, pearson_max = sqrt(phi2 / (1 + phi2)), sqrt((min(n_rows, n_cols) - 1)/min(n_rows, n_cols))
    #Corrected pearson's C
    pearson_c_n = pearson_c/pearson_max
    #convert to pd.DataFrame
    association = DataFrame([[chi2_stat,chi2_max],[sqrt(phi2),phi_max],[cramer_v,1],[tschuprow_t,tschuprow_max],[pearson_c,pearson_max],[pearson_c_n,1]],
                            columns = ["statistic","upper bound"],index = ["Chi-squared","Phi","Cramer's V","Tschuprow's T","Pearson's C","Norm. Pearson's C"])
    #convert to ordered dictionary
    goodness_ = {"test": test, "association": association}
    #convert to namedtuple
    goodness = namedtuple("goodness",goodness_.keys())(*goodness_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #residuals
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #absolute residuals and attraction repulsion index
    resid, att_rep_ind = obj.call_.X - expected_freq,  obj.call_.X / expected_freq
    #standardized residuals
    resid_std = resid /sqrt(expected_freq)
    #adjusted residuals and chi2 contributions
    resid_adj, chi2_ctr = (resid_std.T / sqrt(1-obj.call_.row_w)).T / sqrt(1-obj.call_.col_w), (resid_std**2)/chi2_stat
    #convert to ordered dictionary
    residuals_ = {"resid": resid, "resid_std": resid_std, "resid_adj": resid_adj, "contrib": chi2_ctr, "att_rep_ind": att_rep_ind}
    #convert to namedtuple
    residuals = namedtuple("residuals",residuals_.keys())(*residuals_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #compute others indicators 
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #kaiser threshold
    kaiser = DataFrame([[obj.eig_.iloc[:,0].mean(),100/obj.eig_.shape[0]]],columns=["threshold","proportion"],index=["Kaiser"])

    #convert to namedtuple
    return namedtuple("statsCAResult",["goodness","residuals","kaiser"])(goodness,residuals,kaiser)