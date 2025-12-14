# -*- coding: utf-8 -*-
from numpy import zeros,sqrt,average,cov,array
from pandas import DataFrame, Series
from collections import namedtuple, OrderedDict
from typing import NamedTuple
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#intern function
from .functions.utils import is_dataframe
from .functions.splitmix import splitmix
from .functions.recodecont import recodecont
from .functions.recodecat import recodecat
from .functions.summarize import conditional_wmean
from .functions.predict_sup import predict_sup
from .functions.function_eta2 import function_eta2

def supvarpPCA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables projection in Partial Principal Components Analysis
    ---------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Partial Principal Components Analysis.

    Usage
    -----
    ```python
    >>> supvarpPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class pPCA

    `X`: a pandas Dataframe of supplementary variables

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary variables, including: 

    `quanti`: a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables, including :
        * `coord`: coordinates of supplementary quantitative variables,
        * `cos2`: squared cosinus of the supplementary quantitative variables,
        * `statistics`: statistics for linear regression between supplementary quantitative variables and partial variables.
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results of the supplementary qualitative variables/levels, including :
        * `coord`: coordinates of the supplementary levels,
        * `cos2`: squared cosinus of the supplementary levels,
        * `vtest`: value-test of the supplementary levels,
        * `dist2`: squared distance to origin of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
     ```python
    >>> from scientisttools.datasets import load_autos2006
    >>> from scientisttools import pPCA, supvarpPCA
    >>> autos2006 = load_autos2006("actif")
    >>> res_ppca = pPCA(partial=0)
    >>> res_ppca.fit(autos2006)
    >>> #supplementary variables (quantitative & qualitative)
    >>> X_sup_var = load_autos2006("sup_var")
    >>> sup_var_predict = supvarpPCA(res_ppca,X_sup_var)
    >>> quanti_sup = sup_var_predict.quanti_sup
    >>> quanti_sup.coord #coordinates for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    >>> quanti_sup.statistics #statistics of linear regression
    >>> quali_sup = sup_var_predict.quali_sup
    >>> quali_sup.coord #coordinates for the supplementary levels
    >>> quali_sup.cos2 #cos2 for the supplementary levels
    >>> quali_sup.vtest #vtest for the supplementary levels
    >>> quali_sup.dist2 #dist2 for the supplementary levels
    >>> quali_sup.eta2 #eta2 for the supplementary qualitative variables
    ``` 
    """
    if self.model_ != "ppca": #check if self is and object of class pPCA
        raise TypeError("'self' must be an object of class pPCA")
    
    if isinstance(X,Series): #if pandas Series, transform to pandas DataFrame
        X = X.to_frame()

    is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
    
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
        #xteract coefficients and intercept
        ols_results = DataFrame(zeros((X_quanti_sup.shape[1],len(self.call_.partial)+4)),columns = [*["intercept"],*self.call_.partial,*["R2","Adj. R2","RMSE"]],index=X_quanti_sup.columns)
        Xhat_quanti_sup = DataFrame(columns=X_quanti_sup.columns,index=X_quanti_sup.index.tolist()).astype("float")
        for k in X_quanti_sup.columns:
            ols = sm.WLS(endog=X_quanti_sup[k].astype(float),exog=sm.add_constant(self.call_.X[self.call_.partial]),weights=self.call_.ind_weights).fit()
            ols_results.loc[k,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(X_quanti_sup[k],ols.fittedvalues,squared=False)]]
            Xhat_quanti_sup.loc[:,k] = ols.resid

        #compute weighted average and weighted standard deviation for supplementary quantitative variables
        center_sup = Series(average(Xhat_quanti_sup,axis=0,weights=self.call_.ind_weights),index=X_quanti_sup.columns,name="center")
        scale_sup = Series(array([sqrt(cov(Xhat_quanti_sup.iloc[:,k],rowvar=False,aweights=self.call_.ind_weights,ddof=0)) for k in range(n_quanti_sup)]),index=X_quanti_sup.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        Zhat_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_sup(X=Zhat_quanti_sup,Y=self.svd_.U,weights=self.call_.ind_weights,axis=1)
        del quanti_sup_['dist2'] #delete dist2
        #add statistics
        quanti_sup_ = OrderedDict(**quanti_sup_, **OrderedDict(statistics=ols_results))
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary qualitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quali_sup > 0:
        #recode supplementary qualitative variables
        rec = recodecat(X_quali_sup)
        X_quali_sup, dummies_sup = rec.X, rec.dummies
        #conditional average of original data
        Xhat_levels_sup = conditional_wmean(X=self.call_.Xhat,Y=X_quali_sup,weights=self.call_.ind_weights)
        #standardization: Z = (X - mu)/sigma
        Zhat_levels_sup = Xhat_levels_sup.sub(self.call_.center, axis=1).div(self.call_.scale,axis=1)
        #statistics for supplementary levels
        quali_sup_ = predict_sup(X=Zhat_levels_sup,Y=self.svd_.V,weights=self.call_.var_weights,axis=0)
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

    # Store all informations
    return namedtuple("supvarpPCAResult",["quanti_sup","quali_sup"])(quanti_sup,quali_sup)