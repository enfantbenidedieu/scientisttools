# -*- coding: utf-8 -*-
from numpy import array, average, cov, sqrt, ones
from pandas import DataFrame, Series
from collections import namedtuple, OrderedDict
from typing import NamedTuple

#intern function
from .functions.utils import is_dataframe
from .functions.splitmix import splitmix
from .functions.recodecont import recodecont
from .functions.recodecat import recodecat
from .functions.predict_sup import predict_sup
from .functions.summarize import conditional_wmean
from .functions.function_eta2 import function_eta2

def supvarwcPCA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables projection in Within-class Principal Components Analysis
    --------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Within-class Principal Components Analysis

    Usage
    -----
    ```python
    >>> supvarwcPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class wcPCA

    `X`: a pandas DataFrame of supplementary variables.

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary variables including : 

    `quanti_sup`: a namedtuple of pandas DataFrames containing the results of the supplementary quantitative variables, including :
        * `coord`: coordinates of supplementary quantitative variables,
        * `cos2`: squared cosinus of supplementary quantitative variables.
    
    `quali_sup`: a namedtuple of pandas DataFrames/Series containing the results of the supplementary qualitative variables/levels, including :
        * `coord`: coordinates for supplementary levels,
        * `cos2`: squared cosinus for supplementary levels,
        * `vtest`: value-test for supplementary levels,
        * `dist2`: squared distance to origin for supplementary levels,
        * `eta2`: squared correlation ratio for supplementary qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import wcPCA, supvarwcPCA
    >>> meaudret = load_meaudret("actif")
    >>> res_wcpca = wcPCA(group=9)
    >>> res_wcpca.fit(meaudret)
    >>> #supplementary variables (quantitative and qualitative)
    >>> X_var_sup = load_meaudret("sup_var")
    >>> sup_var = supvarWithinPCA(res_wcpca, X = X_var_sup)
    >>> #supplementary quantitative variables
    >>> quanti_sup = sup_var.quanti
    >>> quanti_sup.coord #coordinates of supplementary quantitative variables
    >>> quanti_sup.vos2 #cos2 of supplementary quantitative variables
    >>> #supplementary qualitative variables
    >>> quali_sup = sup_var.quali
    >>> quali_sup.coord #coordinates of supplementary levels
    >>> quali_sup.cos2 #cos2 of supplementary levels
    >>> quali_sup.vtest #vtest of supplementary levels
    >>> quali_sup.dist2 #dist2 of supplementary levels
    >>> quali_sup.eta2 #eta2 of supplementary qualitative variables
    ```
    """
    if self.model_ != "wcpca": #check if self is and object of class wcPCA
        raise TypeError("'self' must be an object of class wcPCA")
    
    if isinstance(X,Series): #if pandas series, transform to pandas dataframe
        X = X.to_frame()
        
    is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
    
    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")

    split_X = splitmix(X=X) #split X
    X_quanti_sup, X_quali_sup, n_rows, n_quanti_sup, n_quali_sup = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quanti_sup > 0:
        #fill NA with mean
        X_quanti_sup = recodecont(X_quanti_sup).X
        #conditional weighted average
        center_sup = conditional_wmean(X=X_quanti_sup,Y=self.call_.X[self.call_.group],weights=self.call_.ind_weights)
        #center by conditional weighted average
        Xc_quanti_sup = X_quanti_sup.sub(center_sup.loc[self.call_.X[self.call_.group].values,:].values)
        #compute weighted average for supplementary quantitative variables
        xc_center_sup = average(Xc_quanti_sup,axis=0,weights=self.call_.ind_weights)
        if self.standardize:
            xc_scale_sup = array([sqrt(cov(Xc_quanti_sup.iloc[:,k],rowvar=False,aweights=self.call_.ind_weights,ddof=0)) for k in range(n_quanti_sup)])
        else:
            xc_scale_sup = ones(n_quanti_sup)
        #convert to pandas Series
        xc_center_sup, xc_scale_sup = Series(xc_center_sup,index=Xc_quanti_sup.columns,name="center"), Series(xc_scale_sup,index=Xc_quanti_sup.columns,name="scale")
        #standardization : Z = (X - mu)/sigma
        Z_quanti_sup = Xc_quanti_sup.sub(xc_center_sup,axis=1).div(xc_scale_sup,axis=1)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=self.svd_.U,weights=self.call_.ind_weights,axis=1)
        del quanti_sup_['dist2'] #delete dist2
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary qualitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quali_sup > 0:
        #recode supplementary qualitative variables
        rec = recodecat(X=X_quali_sup)
        X_quali_sup, dummies_sup = rec.X, rec.dummies
        #conditional average of original data
        X_levels_sup = conditional_wmean(X=self.call_.Xc,Y=X_quali_sup,weights=self.call_.ind_weights)
        #standardization: Z = (X - mu)/sigma
        Z_levels_sup = X_levels_sup.sub(self.call_.xc_center,axis=1).div(self.call_.xc_scale,axis=1)
        #statistics for supplementary levels
        quali_sup_ = predict_sup(X=Z_levels_sup,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
        #vtest for the supplementary levels
        p_k_sup = dummies_sup.mul(self.call_.ind_weights,axis=0).sum(axis=0)
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
    return namedtuple("supvarwcPCAResult",["quanti_sup","quali_sup"])(quanti_sup,quali_sup)