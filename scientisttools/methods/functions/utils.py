# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
from pandas.api.types import is_numeric_dtype

def is_bool(x):
    return isinstance(x,bool)

def is_dataframe(x):
    return isinstance(x,DataFrame)

def is_dict(x):
    return isinstance(x,dict)

def is_namedtuple(v):
    return isinstance(v,tuple) and hasattr(v, "_fields")

def is_series(x):
    return isinstance(x,Series)

def convert_series_to_dataframe(
        X
):
    """
    Convert pd.Series to pd.DataFrame
    
    Parameters
    ----------
    X : 1d array-like

    """
    if is_series(X):
        X = X.to_frame()
    return X

def check_is_dataframe(
        X
):
    """
    Performs is_dataframe validation

    Check if X is an instance of class pd.DataFrame

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data for which check should be done
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not is_dataframe(X):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

def check_is_series(
        X
):
    """
    Performs is_series validation

    Check if X is an instance of class pd.Series

    Parameters
    ----------
    X : DataFrame of shape (n_samples,)
        Input data for which check should be done
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not is_series(X):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.Series. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.Series.html")

def check_is_bool(
        X
):
    """
    Performs is_bool validation

    Check if X is a boolean

    Parameters
    ----------
    X : bool

    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is a boolean
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not is_bool(X):
        raise TypeError(f"{type(X)} is not supported.")

def check_is_object_or_category_dtype(
        X
):
    """
    Performs is_object_or_category_dtype validation

    Parameters
    ----------
    X : 1d array-like of shape (n_samples,)
        Input data for which check should be done
    """
    if not (X.dtype in ["object","category"]):
        return False
    else:
        return True
    
def is_object_or_category_dtype(
        X
):
    """
    Performs is_object_or_category_dtype validation

    Parameters
    ----------
    X : 1d array-like of shape (n_samples,)
        Input data for which check should be done
    """
    if not (X.dtype in ["object","category"]):
        return False
    else:
        return True

def is_all_object_or_category_dtype(
        X
):
    """
    Performs is_all_object_or_category_dtype validation

    Parameters
    ----------
    X : 2d array-like of shape (n_samples, n_columns)
        Input data for which check should be done.

    Returns
    -------
    bool
    """
    if not all(is_object_or_category_dtype(X[k]) for k in X.columns):
        return False
    else:
        return True
    
def is_all_numeric_dtype(
        X
):
    """
    Performs is_all_numeric_dtype validation

    Parameters
    ----------
    X : 2d array-like of shape (n_samples, n_columns)
        Input data for which check should be done

    Returns
    -------
    bool
    """
    if not all(is_numeric_dtype(X[k]) for k in X.columns):
        return False
    else:
        return True
    
def col_dtype(
        X
):
    """
    Single column date type

    """
    if is_numeric_dtype(X):
        return "quanti"
    elif is_object_or_category_dtype(X):
        return "quali"
    else:
        raise TypeError("Not conventient columns type")
    
def cols_dtypes(
        X
):
    """
    Columns data types
    
    """
    return [col_dtype(X.iloc[:,i]) for i in  range(X.shape[1])]
    
def check_is_all_object_or_category_dtype(
        X
):
    """
    Performs is_all_object_or_category_dtype validation

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data for which check should be done
    """
    if not is_all_object_or_category_dtype(X):
        raise TypeError("All columns in X must be either object or category dtype.")
    else:
        pass

def check_is_all_numeric_dtype(
        X
):
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if all columns in X are numerics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not is_all_numeric_dtype(X):
        raise TypeError("All columns in X must numerics.")
    else:
        pass