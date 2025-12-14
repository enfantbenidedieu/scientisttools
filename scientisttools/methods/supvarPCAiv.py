# -*- coding: utf-8 -*-
from numpy import array, average, cov, sqrt, ones
from pandas import DataFrame, Series, get_dummies, concat
from pandas.api.types import is_numeric_dtype, is_string_dtype
import statsmodels.api as sm
from collections import namedtuple
from typing import NamedTuple

#intern function
from .functions.utils import is_dataframe
from .functions.recodecont import recodecont
from .functions.predict_sup import predict_sup

def supvarPCAiv(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables projection with Principal Components Analysis with instrumental variables
    -------------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary quantitative variables with Principal Components Analysis with instrumental variables

    Usage
    -----
    ```python
    >>> supvarPCAiv(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class PCAiv

    `X`: a pandas DataFrame of supplementary quantitative variables.

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables, including:
    
    * `coord`: coordinates of the supplementary quantitative variables,
    
    * `cos2`: squared cosinus of the supplementary quantitative variables.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import rhone
    >>> from scientisttools import PCAiv, supvarPCAiv
    >>> res_pcaiv = PCAiv(iv=(15,16,17))
    >>> res_pcaiv.fit(rhone)
    >>> #supplementary quantitative variables
    >>> X_quanti_sup = rhone.iloc[:,:15]
    >>> quanti_sup = supvarPCAiv(res_pca, X_quanti_sup)
    >>> quanti_sup.coord #coordinates for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    ```
    """
    if self.model_ != "pcaiv": #check if self is and object of class PCAiv
        raise TypeError("'self' must be an object of class PCAiv")
    
    if isinstance(X,Series): #if pandas Series, transform to pandas DataFrame
        X = X.to_frame()
        
    is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
        
    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")
    
    if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
            raise TypeError("All variables must be numeric")

    n_cols = X.shape[1] #number of supplementary columns
    
    #ordinary least squared with instrumental variables
    def olsiv(k, y, x, weights):
        def x_cast(j):
            if is_numeric_dtype(x[j]):
                return x[j]
            if is_string_dtype(x[j]):
                return get_dummies(x[j],drop_first=True,dtype=int)
        features = concat((x_cast(j=j) for j in x.columns),axis=1)
        ols = sm.WLS(endog=y[k].astype(float),exog=sm.add_constant(features),weights=weights).fit()
        return Series(ols.fittedvalues,index=y.index,name=k)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fill NA with mean
    X = recodecont(X=X).X
    #compute weighted average for supplementary quantitative variables
    center = Series(average(X,axis=0,weights=self.call_.ind_weights),index=X.columns,name="center")
    scale = Series(array([sqrt(cov(X.iloc[:,k],rowvar=False,aweights=self.call_.ind_weights,ddof=0)) for k in range(n_cols)]),index=X.columns,name="scale")
    #standardization: Z = (X - mu)/sigma
    ys = X.sub(center,axis=1).div(scale,axis=1)
    #ordinaly least squared with instrumental variables
    yhat = concat((olsiv(k=k,y=ys,x=self.call_.Xs,weights=self.call_.ind_weights) for k in ys.columns),axis=1)
    #compute weighted average for supplementary quantitative variables
    yhat_center = Series(average(yhat,axis=0,weights=self.call_.ind_weights),index=yhat.columns,name="center")
    yhat_scale = Series(ones(n_cols),index=yhat.columns,name="scale")
    #standardization: Z = (X - mu)/sigma
    Z = yhat.sub(yhat_center,axis=1).div(yhat_scale,axis=1)
    #statistics for supplementary quantitative variables
    quanti_sup_ = predict_sup(X=Z,Y=self.svd_.U,weights=self.call_.ind_weights,axis=1)
    del quanti_sup_['dist2'] #delete dist2

    #convert to namedtuple
    return namedtuple("supvarPCAivResult",quanti_sup_.keys())(*quanti_sup_.values())