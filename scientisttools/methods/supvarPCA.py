# -*- coding: utf-8 -*-
from numpy import array, average, cov, sqrt, ones
from pandas import DataFrame, Series
from collections import namedtuple, OrderedDict
from typing import NamedTuple

#intern function
from .functions.splitmix import splitmix
from .functions.recodecont import recodecont
from .functions.recodecat import recodecat
from .functions.predict_sup import predict_sup
from .functions.summarize import conditional_wmean
from .functions.function_eta2 import function_eta2

def supvarPCA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables projection with Principal Components Analysis
    -------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Principal Components Analysis

    Usage
    -----
    ```python
    >>> supvarPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `X`: a pandas DataFrame of supplementary variables.

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary variables, including: 

    `quanti`: a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables, including:
        * `coord`: coordinates of the supplementary quantitative variables,
        * `cos2`: squared cosinus of the supplementary quantitative variables.
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results of the supplementary qualitative variables/levels including:
        * `barycentre`: the conditional average of active variables,
        * `coord`: coordinates of the supplementary levels,
        * `cos2`: squares cosinus of the supplementary levels,
        * `vtest`: value-test of the supplementary levels,
        * `dist2`: squared distance to origin of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_decathlon
    >>> from scientisttools import PCA, supvarPCA
    >>> decathlon = load_decathlon("actif")
    >>> res_pca = PCA()
    >>> res_pca.fit(decathlon)
    >>> #supplementary quantitative and qualitative variables
    >>> X_sup_var = load_decathlon("sup_var")
    >>> sup_var_predict = supvarPCA(res_pca, X_sup_var)
    >>> quanti_sup = sup_var_predict.quanti_sup
    >>> quanti_sup.coord #coordinates for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    >>> quali_sup = sup_var_predict.quali_sup
    >>> quali_sup.barycentre #conditional average of the active variables
    >>> quali_sup.coord #coordinates for the supplementary levels
    >>> quali_sup.cos2 #cos2 for the supplementary levels
    >>> quali_sup.vtest #vtest for the supplementary levels
    >>> quali_sup.dist2 #dist2 for the supplementary levels
    >>> quali_sup.eta2 #eta2 for the supplementary qualitative variables
    ```
    """
    if self.model_ != "pca": #check if self is and object of class PCA
        raise TypeError("'self' must be an object of class PCA")
    
    if isinstance(X,Series): #if pandas Series, transform to pandas DataFrame
        X = X.to_frame()
        
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")

    #split X
    split_X = splitmix(X=X)
    X_quanti_sup, X_quali_sup, n_rows, n_quanti_sup, n_quali_sup = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quanti_sup > 0:
        #fill NA with mean
        X_quanti_sup = recodecont(X=X_quanti_sup).X
        #compute weighted average for supplementary quantitative variables
        center_sup = average(X_quanti_sup,axis=0,weights=self.call_.ind_weights)
        if self.standardize:
            scale_sup = array([sqrt(cov(X_quanti_sup.iloc[:,k],rowvar=False,aweights=self.call_.ind_weights,ddof=0)) for k in range(n_quanti_sup)])
        else:
            scale_sup = ones(n_quanti_sup)
        #convert to pandas Series
        center_sup, scale_sup = Series(center_sup,index=X_quanti_sup.columns,name="center"), Series(scale_sup,index=X_quanti_sup.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        Z_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=self.svd_.U,weights=self.call_.ind_weights,axis=1)
        del quanti_sup_['dist2'] #delete dist2
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary qualitative variables/levels
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quali_sup > 0 :
        #recode supplementary qualitative variables
        rec = recodecat(X_quali_sup)
        X_quali_sup, dummies_sup = rec.X, rec.dummies
        #conditional average of original data
        X_levels_sup = conditional_wmean(X=self.call_.X,Y=X_quali_sup,weights=self.call_.ind_weights)
        #standardization: Z = (X - mu)/sigma
        Z_levels_sup = X_levels_sup.sub(self.call_.center, axis=1).div(self.call_.scale,axis=1)
        #statistics for supplementary levels
        quali_sup_ = predict_sup(X=Z_levels_sup,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
        #vtest for the supplementary levels
        p_k_sup = dummies_sup.mul(self.call_.ind_weights,axis=0).sum(axis=0)
        levels_sup_vtest = quali_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(self.svd_.vs[:self.call_.n_components],axis=1)
        #eta2 for the supplementary qualitative variables
        quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=self.call_.ind_weights,excl=None)
        #convert to ordered dictionary
        quali_sup_ = OrderedDict(barycentre=X_levels_sup,coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarPCAResult",["quanti_sup","quali_sup"])(quanti_sup,quali_sup)