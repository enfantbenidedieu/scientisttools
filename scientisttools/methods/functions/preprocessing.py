# -*- coding: utf-8 -*-
from numpy import number
from pandas import DataFrame, options
options.mode.copy_on_write = True #to avir
from pandas.api.types import is_numeric_dtype

#intern functions
from .utils import check_is_dataframe, is_object_or_category_dtype
from .func_fillna import func_fillna
from .revalue import revalue

def preprocessing(
        X
) -> DataFrame:
    """
    Preprocessing

    Performs preprocessing (drop levels, fill NA with mean, convert to ordinal factor) on a pandas DataFrame.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data, where ``n_samples`` in the number of samples and ``n_columns`` is the number of columns.

    Returns
    -------
    X : DataFrame of shape (n_samples, n_columns)
        Preprocessed data.    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X=X)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index name as None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X.index.name = None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #drop level if ndim greater than 1 and reset columns name
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.columns.nlevels > 1:
        X.columns = X.columns.droplevel()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X contains columns 
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if any(not (is_numeric_dtype(X[c]) or is_object_or_category_dtype(X[c])) for c in X.columns):
        raise TypeError("Columns in X must be either numeric, object or category.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fill NA with mean
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X_quanti = X.select_dtypes(include=number)
    if not X_quanti.empty:
        X[X_quanti.columns] = func_fillna(X=X_quanti,method="mean")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fill NA with most_frequent
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X_quali = X.select_dtypes(exclude=number)
    if not X_quali.empty:
        X[X_quali.columns] = revalue(X=func_fillna(X=X_quali,method="most_frequent"))

    return X