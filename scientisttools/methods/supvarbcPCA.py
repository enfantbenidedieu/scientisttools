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

def supvarbcPCA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables projection in Between-class Principal Components Analysis
    ---------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Between-class Principal Components Analysis

    Usage
    -----
    ```python
    >>> supvarbcPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

    `X`: a pandas DataFrame of supplementary variables.

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary variables, including: 

    `quanti`: a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables, including:
        * `coord`: coordinates of the supplementary quantitative variables,
        * `cos2`: squared cosinus of the supplementary quantitative variables.
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results of the supplementary qualitative variables/levels including :
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
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, supvarbcPCA
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> #supplementary quantitative and qualitative variables
    >>> X_sup_var = load_meaudret("sup_var")
    >>> sup_var_predict = supvarbcPCA(res_bcpca, X_sup_var)
    >>> quanti_sup = sup_var_predict.quanti
    >>> quanti_sup.coord #coordinates for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    >>> quali_sup = sup_var_predict.quali
    >>> quali_sup.coord #coordinates for the supplementary levels
    >>> quali_sup.cos2 #cos2 for the supplementary levels
    >>> quali_sup.vtest #vtest for the supplementary levels
    >>> quali_sup.dist2 #dist2 for the supplementary levels
    >>> quali_sup.eta2 #eta2 for the supplementary qualitative variables
    ```
    """
    if self.model_ != "bcpca": #check if self is and object of class bcPCA
        raise TypeError("'self' must be an object of class bcPCA")
    
    if isinstance(X,Series): #if pandas Series, transform to pandas DataFrame
        X = X.to_frame()
        
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")

    #plit X
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
        #conditional weighted average
        X_levels_quanti_sup = conditional_wmean(X=Z_quanti_sup,Y=self.call_.X[self.call_.group],weights=self.call_.ind_weights)
        #compute weighted average and weighted standard deviation for supplementary conditional
        levels_center_quanti_sup = Series(average(X_levels_quanti_sup,axis=0,weights=self.call_.levels_weights),index=X_levels_quanti_sup.columns,name="center")
        levels_scale_quanti_sup = Series(ones(n_quanti_sup),index=X_levels_quanti_sup.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        Z_levels_quanti_sup = X_levels_quanti_sup.sub(levels_center_quanti_sup,axis=1).div(levels_scale_quanti_sup,axis=1)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_sup(X=Z_levels_quanti_sup,Y=self.svd_.U,weights=self.call_.levels_weights,axis=1)
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
        rec = recodecat(X=X_quali_sup)
        X_quali_sup, dummies_sup = rec.X, rec.dummies
        #conditional mean - Barycenter of original data
        X_levels_sup = conditional_wmean(X=self.call_.Z,Y=X_quali_sup,weights=self.call_.ind_weights)
        #standardization: Z = (X - mu)/sigma
        Z_levels_sup = X_levels_sup.sub(self.call_.levels_center,axis=1).div(self.call_.levels_scale,axis=1)
        #statistics for supplementary levels
        quali_sup_ = predict_sup(X=Z_levels_sup,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
        #vtest for the supplementary levels
        p_k_sup = dummies_sup.mul(self.call_.ind_weights,axis=0).mean(axis=0)
        levels_sup_vtest = quali_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(self.svd_.vs[:self.call_.n_components],axis=1)
        #eta2 for the supplementary qualitative variables
        quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=self.call_.ind_weights,excl=None)
        #convert to ordered dictionary
        quali_sup_ = OrderedDict(coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None

    #convert to namedtuple
    return namedtuple("supvarbcPCAResult",["quanti_sup","quali_sup"])(quanti_sup,quali_sup)