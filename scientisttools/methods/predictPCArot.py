# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from typing import NamedTuple

#intern functions
from .functions.utils import is_dataframe

def predictPCArot(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Varimax rotation in Principal Component Analysis
    ----------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Varimax rotation in Principal Component Analysis.

    Usage
    -----
    ```python
    >>> predictPCArot(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class PCArot

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    a namedtuple of pandas DataFrames/Series containing all the results for the new individuals, including:
    
    `coord`: coordinates for the new individuals after rotation,

    `cos2`: squared cosinus for the new individuals after rotation,

    `dist2`: squared distance to origin for the new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_autos2006, PCA, PCArot, predictPCArot
    >>> autos2006 = load_autos2006("all")
    >>> res_pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_pca.fit(autos2006)
    >>> res_pcarot = PCArot(n_components=2)
    >>> res_pcarot.fit(res_pca)
    >>> #predict on new individuals
    >>> X_ind_sup = load_autos2006("ind_sup")
    >>> predict = predictPCArot(res_pca,X_ind_sup)
    >>> predict.coord #coordinates of new individuals after rotation
    >>> predict.cos2 #cos2 of new individuals after rotation
    >>> predict.dist2 #dist2 of new individuals
    ```
    """
    if self.model_ != "pcarot": #check if self is an object of class PCArot
        raise TypeError("'self' must be an object of class PCArot")
    
    is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
      
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")
    
    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
        raise TypeError("All columns must be numerics")

    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")
    X = X.loc[:,self.call_.X.columns] #reorder columns

    #standardization: Z = (X - mu)/sigma
    Z = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1)
    #coordinates of the new individuals
    coord = Z.mul(self.call_.var_weights,axis=1).dot(self.var_.coord).div(self.svd_.vs[:self.call_.n_components],axis=1)
    #dist2 of the new individuals
    sqdisto = Z.pow(2).mul(self.call_.var_weights,axis=1).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #cos2 for the new individuals
    sqcos = coord.pow(2).div(sqdisto,axis=0)
    #convert to namedtuple
    return namedtuple("predictPCArotResult",["coord","cos2","dist2"])(coord,sqcos,sqdisto)