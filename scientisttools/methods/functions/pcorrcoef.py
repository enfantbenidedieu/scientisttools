# -*- coding: utf-8 -*-
from numpy import linalg, eye, sqrt
from pandas import DataFrame, api

def pcorrcoef(X:DataFrame) -> DataFrame:
    """
    Linear partial correlation
    --------------------------

    Description
    -----------
    Performans the sample linear partial correlation coefficients between pairs of variables, controlling for all other remaining variables

    Usage
    -----
    ```python
    >>> pcorrcoef(X)
    ```

    Parameters
    ----------
    X : pandas DataFrame, shape (n, p)
        DataFrame with the different variables. Each column is taken as a variable.

    Returns
    -------
    partial : pandas DataFrame, shape (p, p)
        partial[i, j] contains the partial correlation of X[:, i] and X[:, j] controlling for all other remaining variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> #load beer dataset
    >>> from scientisttools import load_beer(), pcorrcoef
    >>> beer = load_beer()
    >>> pcorr = pcorrcoef(beer)
    >>> pcorr
    
    ```
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pandas DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #check if all variables are numerics
    all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")
    
    #inverse of correlation matrix
    inv_corr = linalg.inv(X.corr(method="pearson"))
    #partial correlation matrix
    partial = DataFrame(eye(X.shape[1]),index=X.columns,columns=X.columns)
    for i in range(X.shape[1]-1):
        for j in range(i+1,X.shape[1]):
            partial.iloc[i,j] = -inv_corr[i,j]/sqrt(inv_corr[i,i]*inv_corr[j,j])
            partial.iloc[j,i] = partial.iloc[i,j]
    return partial  