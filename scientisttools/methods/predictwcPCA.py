# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from typing import NamedTuple

#intern functions
from .functions.utils import is_dataframe
from .functions.predict_sup import predict_sup

def predictwcPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Within-class Principal Component Analysis
    -------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Within-class Principal Component Analysis

    Usage
    -----
    ```python
    >>> predictwcPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class wcPCA

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    a namedtuple of pandas DataFrame/Series containing all the results for the new individuals, including:
    
    `coord`: coordinates of new individuals,

    `cos2`: squared cosinus of new individuals,

    `dist2`: squared distance to origin of new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import meaudret
    >>> from scientisttools import wcPCA, predictwcPCA
    >>> res_wcpca = wcPCA(group=9,sup_var=range(10,24))
    >>> res_wcpca.fit(meauret)
    >>> #predict on active individuals
    >>> predict = predictwcPCA(res_wcpca,X=res_wcpca.call_.X)
    >>> predict.coord #coordinates of active individuals
    >>> predict.cos2 #cos2 of active individuals
    >>> predict.dist2 #dist2 of active individuals
    ```
    """
    if self.model_ != "wcpca": #check if self is an object of class wcPCA
        raise TypeError("'self' must be an object of class wcPCA")
    
    is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
        
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the wcPCA result")
    X = X.loc[:,self.call_.X.columns] #reorder columns

    y, x = X[self.call_.group], X.drop(columns=self.call_.group) #split X into x and y

    if not all(is_numeric_dtype(x[k]) for k in x.columns): #check if all variables are numerics
        raise TypeError("All columns in data must be numeric")
    
    if not all(isinstance(x, str) for x in y): #check if y is categorics
        raise TypeError("y must be categorics")
    
    #standardization: z (x - mu)/sigma
    Z = x.sub(self.call_.center.loc[y.values,:].values).sub(self.call_.xc_center,axis=1).div(self.call_.xc_scale,axis=1)
    #statistics for news individuals
    predict = predict_sup(X=Z,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
    #convert to namedtuple
    return namedtuple("predictwcPCAResult",predict.keys())(*predict.values())