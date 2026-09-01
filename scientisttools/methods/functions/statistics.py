# -*- coding: utf-8 -*-
from numpy import ones, array, linalg, ndarray, average, cov, sqrt, unique
from pandas import DataFrame, Series, concat
from statsmodels.api import WLS, add_constant

#intern functions
from .utils import convert_series_to_dataframe, check_is_dataframe, is_all_numeric_dtype, is_all_object_or_category_dtype
from .get_indices import get_indices
from .cov2corr import cov2corr

def wmean(X, w=None):
    """
    Weighted average

    Compute the weighted average of all columns in X.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data containing the data to be averaged. 
    
    w : 1d array-like of shape (n_samples,) default = None
         Weights associated with the values in X.

    Returns
    -------
    wmean : Series of shape (n_columns,)
        The weighted average of X.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    return Series(average(X,axis=0,weights=w),index=X.columns,name="center")

def wvar(X, w=None,ddof=0):  
    """
    Weighted variance

    Compute the weighted variance of all columns in X.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data containing the data to be averaged.
    
    w : 1d array-like of shape (n_samples,) default = None
         Weights associated with the values in X.

    ddof : int, default = 0
        If not None the default value implied by bias is overridden. Note that ``ddof=1`` will return the unbiased estimate, 
        even if both fweights and aweights are specified, and ``ddof=0`` will return the simple average. 
        The default value is :math:`0`.

    Returns
    -------
    wvar : Series of shape (n_columns,)
        The weighted variance of ``X``.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    return Series([cov(m=X.iloc[:,j],aweights=w,ddof=ddof) for j in range(X.shape[1])],index=X.columns,name="variance").astype("float")

def wstd(X, w=None, ddof=0):  
    """
    Weighted standard deviation

    Compute the weighted standard deviation of all columns in ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data containing the data to be averaged. ``X`` must be an object of class ``pandas.Series`` or ``pandas.DataFrame``.
    
    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in ``X``.

    ddof : int, default = 0
        If not ``None`` the default value implied by bias is overridden. Note that ``ddof=1`` will return the unbiased estimate, 
        even if both fweights and aweights are specified, and ``ddof=0`` will return the simple average. 
        The default value is :math:`0`.

    Returns
    -------
    wstd : Series of shape (n_columns,)
        The weighted standard deviation of ``X``.
    """
    return Series(wvar(X=X,w=w,ddof=ddof).transform(sqrt).values,index=X.columns,name="scale")

def wcov(X, w=None, ddof=0):
    """
    Weighted Covariance Matrix

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.
    
    w : 1d array-like of shape (n_samples,), optional, default = None
        Weights associated with the values in ``X``.

    ddof : int, default = 0
        If not ``None`` the default value implied by bias is overridden. Note that ``ddof=1`` will return the unbiased estimate, 
        even if both fweights and aweights are specified, and ``ddof=0`` will return the simple average. 
        The default value is :math:`0`.

    Returns
    -------
    wcov : DataFrame of shape (n_columns, n_columns)
        Weighted covariance matrix.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    return DataFrame(cov(X,rowvar=False,aweights=w,ddof=ddof),index=X.columns,columns=X.columns)

def wcorr(X, w=None, ddof=0):
    """
    Weighted pearson correlation coefficient
    
    Performs weighted pearson coerrelation coefficient matrix

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns).
        Input data.

    w : 1d array-like, default = None
        An optional rows weights.

    ddof : int, default = 0
        If not None the default value implied by bias is overridden. Note that ddof=1 will return the unbiased estimate, 
        even if both fweights and aweights are specified, and ddof=0 will return the simple average. 
        The default value is :math:`0`.

    Returns
    -------
    wcorr : DataFrame of shape (n_columns, n_columns)
        The weighted correlation matrix of the variables.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    return cov2corr(wcov(X=X,w=w,ddof=ddof))

def wpcorr(X,partial=None,w=None, ddof=0):
    """
    Weighted Linear partial correlation
    
    Performans the sample weighted linear partial correlation coefficients between pairs of variables, controlling for all other remaining variables

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data with the different variables. Each column is taken as a variable.

    partial : str, default = None
        The partial variable. if None, then 

    w : 1d array-like of shape (n_samples,), optional, default = None
        Weights associated with the values in X.

    ddof : int, default = 0
        If not ``None`` the default value implied by bias is overridden. Note that ``ddof=1`` will return the unbiased estimate, 
        even if both fweights and aweights are specified, and ``ddof=0`` will return the simple average. 
        The default value is :math:`0`.

    Returns
    -------
    pcorr : DataFrame of shape (p, p)
        partial[i, j] contains the partial correlation of X[:, i] and X[:, j] controlling for all other remaining variables.

    Examples
    --------
    >>> from scientisttools.dataset import beer
    >>> from scientisttools import wpcorr
    >>> pcorr = pcorrcoef(beer)
    >>> pcorr
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if partial is None:
        #pearson correlation matrix
        M = wcorr(X=X,w=w,ddof=ddof)
        inv_M = DataFrame(linalg.pinv(M,hermitian=True),index=X.columns,columns=X.columns)
        #weighted partial correlation matrix
        pcorr = -1*cov2corr(inv_M)
        for c in pcorr.columns:
            pcorr.loc[c,c] = 1
    else:
        #split X into z (partial variables) and x (dependent variables)
        z, x = X[partial], X.drop(columns=partial)
        #resid
        Xhat = concat((Series(WLS(endog=x[k].astype(float),exog=add_constant(z),weights=w).fit().resid,index=x.index,name=k) for k in x.columns),axis=1)
        #weighted pearson correlation
        pcorr = wcorr(X=Xhat,w=w,ddof=ddof)
    return pcorr  
    
def summarize(X):
    """
    Summarize DataFrame
    
    Parameters
    ----------
    X : DataFrame with shape (n_samples, n_columns) or Series of shape (n_samples,).
        X contains either numerics or categoricals columns.

    Returns
    -------
    result : DataFrame of shape
        Summaries
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if is_all_numeric_dtype(X):
        res = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        res["count"] = res["count"].astype("int")
        return res
    elif is_all_object_or_category_dtype(X):
        def freq_prop(q):
            eff = (X[q].value_counts().to_frame("count").reset_index()
                       .rename(columns={q : "categorie"})
                       .assign(proportion = lambda x : x["count"]/x["count"].sum()))
            eff.insert(0,"variable",q)
            return eff
        return concat((freq_prop(q) for q in X.columns),axis=0, ignore_index=True)
    else:
        raise TypeError("All columns must be either numerics or categoricals.")

def func_groupby(X, by, func="mean", w=None, ddof=0):
    """
    Weighted statistics by group

    Performns weighted statistics of continuous variables by group.
    
    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        X Input data. X contains continuous variables.
      
    by : array-like of shape (n_samples,) or (n_samples, n_columns)
        by Input data. by contains categorical variables.

    func : str, default = "mean"
        Statistics which should be performns. Possible values are:

        * "sum" for sum, 
        * "mean" for average, 
        * "var" for variance and,
        * "std" for standard deviation.

    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in X.

    Returns
    -------
    groupby : DataFrame of shape (n_levels, n_columns)
        The conditional statistics.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert to an object of class pd.DataFrame if an object of class pd.Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()
    if isinstance(by,Series):
        by = by.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X and Y is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    if not isinstance(by,DataFrame):
        raise TypeError(f"{type(by)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X and by have same number of rows
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.shape[0] != by.shape[0]:
        raise TypeError("Not convenient dimension between X and by")

    #set number of rows
    n_rows = X.shape[0]

    #set w for average, variance and standard deviation
    if func != "sum":
        if w is None: 
            w = ones(n_rows)/n_rows
        elif not isinstance(w,(list,tuple,ndarray,Series)): 
            raise TypeError("'w' must be a 1d array-like of rows weights.")
        elif len(w) != n_rows: 
            raise ValueError(f"'w' must be a 1d array-like with length {n_rows}.")
        else: 
            w = array(w)/sum(w)

    def groupby(q):
        modalite = sorted(unique(by[q]))
        def statsby(kq):
            idx = get_indices(by[q],kq)
            if func == "sum": 
                return X.iloc[idx,:].sum(axis=0)
            elif func == "mean": 
                return wmean(X=X.iloc[idx,:],w=w[idx])
            elif func == "var": 
                return wvar(X=X.iloc[idx,:],w=w[idx],ddof=ddof)
            elif func == "std": 
                return wstd(X=X.iloc[idx,:],w=w[idx],ddof=ddof)
            else:
                raise ValueError(f"Not convenient func value. Allowed values are : sum, mean, var and std. but get {func}")
        return concat((statsby(kq=kq).to_frame(kq) for kq in modalite),axis=1).T    
    return concat((groupby(q) for q in by.columns),axis=0)