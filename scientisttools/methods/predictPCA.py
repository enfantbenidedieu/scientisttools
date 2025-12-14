# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from typing import NamedTuple

#intern function
from .functions.predict_sup import predict_sup

def predictPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Principal Component Analysis
    ------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Principal Component Analysis.

    Usage
    -----
    ```python
    >>> predictPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    namedtuple of pandas DataFramed/Series containing all the results for the new individuals, including:
    
    `coord`: coordinates for the new individuals,

    `cos2`: squared cosinus for the new individuals,

    `dist2`: squared distance to origin for the new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_decathlon
    >>> from scientisttools import PCA, predictPCA
    >>> decathlon = load_decathlon("actif")
    >>> res_pca = PCA()
    >>> res_pca.fit(decathlon)
    >>> #predict on new individuals
    >>> ind_sup = load_decathlon("ind_sup")
    >>> predict = predictPCA(res_pca,X=ind_sup)
    >>> predict.coord #coordinate of new individuals
    >>> predict.cos2 #squared cosinus of new individuals
    >>> predict.dist2 #squared distance to origin of new individuals
    ```
    """
    if self.model_ != "pca": #check if self is an object of class PCA
        raise TypeError("'self' must be an object of class PCA")
    
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")
    
    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
        raise TypeError("All columns must be numerics")

    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")
    #reorder columns
    X = X.loc[:,self.call_.X.columns]

    #standardization: Z = (X - mu)/sigma
    Z = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1)
    #statistics for news individuals
    predict = predict_sup(X=Z,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
    #convert to namedtuple
    return namedtuple("predictPCAResult",predict.keys())(*predict.values())