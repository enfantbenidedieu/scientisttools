# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from typing import NamedTuple
from collections import namedtuple

#intern function
from .functions.predict_sup import predict_sup

def predictbcPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Between-class Principal Component Analysis
    --------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Between-class Principal Component Analysis.

    Usage
    -----
    ```python
    >>> predictbcPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

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
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, predictbcPCA
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> #predict on new individuals
    >>> predict = predictbcPCA(res_bcpca,X=meaudret)
    >>> predict.coord #coordinate of new individuals
    >>> predict.cos2 #squared cosinus of new individuals
    >>> predict.dist2 #squared distance to origin of new individuals
    ```
    """
    if self.model_ != "bcpca": #check if self is an object of class bcPCA
        raise TypeError("'self' must be an object of class bcPCA")
    
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the bcPCA result")
    #reorder columns and drop group label
    X = X.loc[:,self.call_.X.columns].drop(columns=self.call_.group)

    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
        raise TypeError("All columns must be numeric")

    #standardization: Z = (X - mu)/sigma
    Z = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).sub(self.call_.levels_center,axis=1).div(self.call_.levels_scale,axis=1).mul(self.call_.var_weights,axis=1)
    #statistics for news individuals
    predict = predict_sup(X=Z,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
    #convert to namedtuple
    return namedtuple("predictbcPCAResult",predict.keys())(*predict.values())