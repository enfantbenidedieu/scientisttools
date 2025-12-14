# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, average, cov, sqrt
from pandas import DataFrame, concat
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas import concat, DataFrame, Series

#intern functions
from .get_indices import get_indices

def summarize(X:DataFrame) -> DataFrame:
    """
    Summarize DataFrame
    -------------------

    Usage
    -----
    ```python
    >>> summarize(X)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with shape (n_samples, n_columns) or a pandas Series of shape (n_samples,).
        X contains either numerics or categoricals columns.

    Return(s)
    ---------
    a pandas DataFrame

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if isinstance(X,Series): #convert to DataFrame if X is a pandas Series
        X = X.to_frame()

    if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    if all(is_numeric_dtype(X[k]) for k in X.columns): #if all variables are numerics
        res = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        res["count"] = res["count"].astype("int")
        return res
    elif all(is_string_dtype(X[q]) for q in X.columns): #if all columns are categoricals
        def freq_prop(q):
            eff = X[q].value_counts().to_frame("count").reset_index().rename(columns={q : "categorie"}).assign(proportion = lambda x : x["count"]/x["count"].sum())
            eff.insert(0,"variable",q)
            return eff
        return concat((freq_prop(q) for q in X.columns),axis=0, ignore_index=True)
    else:
        TypeError("All columns must be either numerics or categoricals.")

def conditional_sum(X:DataFrame,Y:DataFrame) -> DataFrame:
    """
    Conditional Sum
    ---------------

    Usage
    -----
    ```python
    >>> conditional_sum(X,Y)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,)
        X contains numerics variables
    
    `Y`: a pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,)
        Y contains categoricals variables.

    Return(s)
    ---------
    a pandas DataFrame 

    Author(s)
    ---------  
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com  
    """
    if isinstance(X,Series): #convert to DataFrame if X is a pandas Series
        X = X.to_frame()

    if isinstance(Y,Series): #convert to DataFrame if Y is a pandas Series
        Y = Y.to_frame()

    if not isinstance(X,DataFrame): #check if X is an instance of DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    

    if not isinstance(Y,DataFrame): #check if X is an instance of DataFrame class
        raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    

    #conditional sum
    def cond_sum(q):
        data = concat((X,Y[q]),axis=1).groupby(by=q,as_index=True).sum()
        data.index.name = None
        return data
    return concat((cond_sum(q) for q in Y.columns),axis=0)

def conditional_wmean(X,Y,weights=None):
    """
    Conditional Weighted Average
    ----------------------------

    Description
    -----------
    Compute the conditional weighted average

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,)
        X contains numerics variables.

    `Y`: a pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,)
        Y contains categoricals variables.

    `weights`: None or a list or a tuple or a 1D array or a pandas Series of length `n_samples` containing rows weights.

    Return(s)
    ---------
    a pandas DataFrame with conditional weighted average
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if isinstance(X,Series): #convert to DataFrame if X is a pandas Series
        X = X.to_frame()

    if isinstance(Y,Series): #convert to DataFrame if Y is a pandas Series
        Y = Y.to_frame()

    if not isinstance(X,DataFrame): #check if X is an instance of DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    

    if not isinstance(Y,DataFrame): #check if X is an instance of DataFrame class
        raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    

    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list or a tuple or a 1D array or a pandas Series of weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list or a tuple or a 1D array or a pandas Series with length {X.shape[0]}.")
    else:
        weights = array([x/sum(weights) for x in weights])
    
    def wmean(q):
        modalite = sorted(Y[q].unique().tolist())
        def mean_k(kq):
            idx = get_indices(Y[q],kq)
            return DataFrame(average(X.iloc[idx,:],axis=0,weights=weights[idx]).reshape(1,-1),index=[kq],columns=X.columns)
        return concat((mean_k(kq=kq) for kq in modalite),axis=0)
    return concat((wmean(q=q) for q in Y.columns),axis=0)

def conditional_wstd(X,Y,weights=None):
    """
    Conditional Weighted Standard Deviation
    ---------------------------------------

    Description
    -----------
    Compute the conditional weighted standard deviation

    Parameters
    ----------
    `X`: pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,)
        X contains numerics variables

    `Y`: pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,). 
        Y contains categoricals variables

    `weights`: None or a list or a tuple or a 1D array or a pandas Series of length `n_samples` contains rows weights.

    Return(s)
    ---------
    a pandas DataFrame containing conditional weighted standard deviation.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if isinstance(X,Series): #convert to DataFrame if X is a pandas Series
        X = X.to_frame()

    if isinstance(Y,Series): #convert to DataFrame if Y is a pandas Series
        Y = Y.to_frame()

    if not isinstance(X,DataFrame): #check if X is an instance of DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    

    if not isinstance(Y,DataFrame): #check if X is an instance of DataFrame class
        raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")    

    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list or a tuple or a 1D array or a pandas Series of weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list or a tuple or a 1D array or a pandas Series with length {X.shape[0]}.")
    else:
        weights = array([x/sum(weights) for x in weights])
    
    def wstd(q):
        modalite = sorted(Y[q].unique().tolist())
        def std_k(kq):
            idx = get_indices(Y[q],kq)
            return DataFrame([[sqrt(cov(X.iloc[idx,k],aweights=weights[idx],ddof=0)) for k in X.columns]],index=[kq],columns=X.columns)
        return concat((std_k(kq=kq) for kq in modalite),axis=0)
    return concat((wstd(q=q) for q in Y.columns),axis=0)