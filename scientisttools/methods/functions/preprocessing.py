# -*- coding: utf-8 -*-
from numpy import number

#intern functions
from .utils import is_dataframe
from .recodecont import recodecont
from .revalue import revalue

def preprocessing(X):
    """
    Preprocessing
    -------------

    Description
    -----------
    Performs preprocessing (drop levels, fill NA with mean, convert to ordinal factor) on a pandas DataFrame

    Usage
    -----
    ```python
    >>> preprocessing(X)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_columns)
        Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

    Returns
    -------
    `X`: a pandas DataFrame of shape (n_samples, n_columns)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    is_dataframe(X=X)

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
    #fill NA with mean
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X_quanti = X.select_dtypes(include=number)
    if not X_quanti.empty:
        X_quanti = recodecont(X=X_quanti).X
        for k in X_quanti.columns:
            X[k] = X_quanti[k]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert categorical variables to factor
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X_quali = X.select_dtypes(include=["object","category"])
    if not X_quali.empty:
        X_quali = revalue(X=X_quali)
        for q in X_quali.columns:
            X[q] = X_quali[q]

    return X