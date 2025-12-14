# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, concat
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from typing import NamedTuple
import statsmodels.api as sm

#intern function
from .functions.utils import is_dataframe
from .functions.predict_sup import predict_sup

def predictpPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Partial Principal Component Analysis (pPCA)
    ---------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Partial Principal Component Analysis.

    Usage
    -----
    ```python
    >>> predictpPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class pPCA

    `X`: a pandas Dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    a namedtuple of pandas Dataframes containing all the results for the new individuals, including:
    
    `coord`: coordinates of the new individuals,

    `cos2`: squared cosinus of the new individuals,

    `dist2`: squared distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_autos2006
    >>> from scientisttools import pPCA, predictpPCA
    >>> autos2006 = load_autos2006("actif")
    >>> res_ppca = pPCA(partial="CYL")
    >>> res_ppca.fit(autos2006)
    >>> #load new individuals
    >>> X_ind_sup = load_autos2006("ind_sup")
    >>> predict = predictpPCA(res_ppca,X_ind_sup)
    >>> predict.coord #coordinate of the new individuals
    >>> predict.cos2 #cos2 of the new individuals
    >>> predict.dist2 #dist2 of the new individuals
    ```
    """
    if self.model_ != "ppca": #check if self is an object of class pPCA
        raise TypeError("'self' must be an object of class PartialPCA")
    
    is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
     
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if columns are aligned
        raise ValueError("'columns' aren't aligned")
    
    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
        raise TypeError("All columns in X must be numerics")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the pPCA result")
    X = X.loc[:,self.call_.X.columns] #reorder columns

    #split X into z (partial variables) and x (dependent variables)
    z, x = X[self.call_.partial], X.drop(columns=self.call_.partial)

    #residuals for new observations
    Xhat = concat((Series(x[k].sub(self.separate_model_[i].predict(sm.add_constant(z))),index=x.index,name=k) for i,k in enumerate(x.columns)),axis=1)
    #standardization: Z = (X - mu)/sigma
    Zhat = Xhat.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1)
    #statistics for supplementary individuals
    predict = predict_sup(X=Zhat,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
    #convert to namedtuple
    return namedtuple("predictpPCAResult",predict.keys())(*predict.values())