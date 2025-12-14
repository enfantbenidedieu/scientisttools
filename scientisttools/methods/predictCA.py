# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from typing import NamedTuple

#intern functions
from .functions.predict_sup import predict_sup

def predictCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new rows with Correspondence Analysis
    ------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new rows with Correspondence Analysis

    Usage
    -----
    ```python
    >>> predictCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `X`: a pandas DataFrame in which to look for columns with which to predict. X must contain columns with the same names as the original data

    Return
    ------
    a namedtuple of pandas DataFrames/Series containing all the results for the new rows, including:

    `coord`: coordinates of the new rows,

    `cos2`: squared cosinus of the new rows,

    `dist2`: squared distance to origin of the new rows.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_children, CA, predictCA
    >>> from scientisttools import CA, predictCA
    >>> children = load_children()
    >>> res_ca = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> res_ca.fit(children)
    >>> #prediction on supplementary rows
    >>> row_sup = load_children("row_sup")
    >>> predict = predictCA(res_ca,X=row_sup)
    >>> predict.coord #coordinates of new individuals
    >>> predict.cos2 #cos2 of new individuals
    >>> predict.dist2 #dist2 of new individuals
    ```
    """
    if self.model_ != "ca": #check if self is an object of class CA
        raise TypeError("'self' must be an object of class CA")
    
    if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] = ncols
        raise ValueError("'columns' aren't aligned")

    if not all(is_numeric_dtype(X[j]) for j in X.columns): #check if all variables are numerics
        raise TypeError("All columns in X must be numeric")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the CA result")
    X = X.loc[:,self.call_.X.columns] #reorder columns

    #frequencies of new rows
    freq = X.div(self.call_.total)
    #margins for new rows
    marge = freq.sum(axis=1)
    #standardization: z_ij = (fij/(fi.*f.j)) - 1
    Z = freq.div(marge,axis=0).div(self.call_.col_marge,axis=1).sub(1)
    #statistics for new rows
    predict_ = predict_sup(X=Z,Y=self.svd_.V,weights=self.call_.col_marge,axis=0)
    return namedtuple("predictCAResult",predict_.keys())(*predict_.values())