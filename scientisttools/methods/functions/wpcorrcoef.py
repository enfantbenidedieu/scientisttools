# -*- coding: utf-8 -*-
from numpy import linalg, eye, sqrt
from pandas import DataFrame, Series, concat
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm

#intern function
from .wcorrcoef import wcorrcoef

def wpcorrcoef(X:DataFrame,partial=None,weights=None) -> DataFrame:
    """
    Weighted Linear partial correlation
    -----------------------------------

    Description
    -----------
    Performans the sample weighted linear partial correlation coefficients between pairs of variables, controlling for all other remaining variables

    Usage
    -----
    ```python
    >>> pcorrcoef(X,weights=None)
    ```

    Parameters
    ----------
    `X`: pandas DataFrame of shape (n_samples, n_columns)
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
    if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
        raise TypeError("All columns must be numeric")

    if partial is None:
        #inverse of weigthed pearson correlation matrix
        inv_wcorr = linalg.inv(wcorrcoef(X=X,weights=weights))
        #weighted partial correlation matrix
        partial = DataFrame(eye(X.shape[1]),index=X.columns,columns=X.columns)
        for i in range(X.shape[1]-1):
            for j in range(i+1,X.shape[1]):
                partial.iloc[i,j] = -inv_wcorr[i,j]/sqrt(inv_wcorr[i,i]*inv_wcorr[j,j])
                partial.iloc[j,i] = partial.iloc[i,j]
    else:
        #split X into z (partial variables) and x (dependent variables)
        z, x = X[partial], X.drop(columns=partial)
        #resid
        Xhat = concat((Series(sm.WLS(endog=x[k].astype(float),exog=sm.add_constant(z),weights=weights).fit().resid,index=x.index,name=k) for k in x.columns),axis=1)
        #wighted pearson correlation
        partial = wcorrcoef(X=Xhat,weights=weights)
    return partial  