# -*- coding: utf-8 -*-
from numpy import array, sqrt, average, cov
from pandas import DataFrame, Series, concat
from collections import namedtuple
from typing import NamedTuple

#intern functions
from .functions.splitmix import splitmix
from .functions.recodecont import recodecont
from .functions.recodecat import recodecat
from .functions.revalue import revaluate_cat_variable
from .functions.function_eta2 import function_eta2

def supvarMCA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables projection in Multiple Correspondence Analysis
    ----------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Multiple Correspondence Analysis.

    Usage
    -----
    ```python
    >>> supvarMCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `X`: pandas DataFrame of supplementary variables

    Returns
    -------
    a namedtuple of namedtuple containing all the results for supplementary variables, including: 

    `quanti`: a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables, including:
        * `coord`: coordinates of supplementary quantitative variables,
        * `cos2`: squared cosinus of supplementary quantitative variables.
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results of the supplementary qualitative variables/levels, including:
        * `coord`: coordinates of supplementary levels,
        * `cos2`: squared cosinus of supplementary levels,
        * `vtest`: value-test of supplementary levels,
        * `dist2`: squared distance to origin of supplementary levels,
        * `eta2`: squared correlation ratio of supplementary qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_poison
    >>> from scientisttools import MCA, supvarMCA
    >>> poison = load_poison()
    >>> res_mca = MCA(sup_var = range(4))
    >>> res_mca.fit(poison)
    >>> #statistics for supplementary variables
    >>> sup_var = load_poison("sup_var")
    >>> sup_var_predict = supvarPCA(res_mca, X = sup_var)
    >>> #statistics for supplementary quantitative variables
    >>> quanti_sup = sup_var_predict.quanti
    >>> quanti_sup.coord #coordinates of supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 of supplementary quantitative variables
    >>> #statistics for supplementary qualitative variables
    >>> quali_sup = sup_var_predict.quali
    >>> quali_sup.coord #coordinates of supplementary levels
    >>> quali_sup.cos2 #cos2 of supplementary levels
    >>> quali_sup.vtest #vtest of supplementary levels
    >>> quali_sup.dist2 #dist2 of supplementary levels
    >>> quali_sup.eta2 #eta2 of supplementary qualitative variables
    ```
    """
    if self.model_ != "mca": #check if self is and object of class MCA
        raise TypeError("'self' must be an object of class MCA")
    
    if isinstance(X,Series): #if pandas series, transform to pandas dataframe
        X = X.to_frame()
    
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")
    
    split_X = splitmix(X=X) #split X
    X_quanti, X_quali, n_rows, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

    #initialize
    Z = DataFrame().astype(float)
    if n_quanti > 0:
        #fill NA with mean
        X_quanti = recodecont(X=X_quanti).X
        #compute weighted average and weighted standard deviation
        center = Series(average(X_quanti,axis=0,weights=self.call_.ind_weights),index=X_quanti.columns,name="center")
        scale = Series(array([sqrt(cov(X_quanti.iloc[:,k],aweights=self.call_.ind_weights,ddof=0)) for k in range(n_quanti)]),index=X_quanti.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        Z_quanti = X_quanti.sub(center,axis=1).div(scale,axis=1)
        #concatenate
        Z = concat((Z,Z_quanti),axis=1)

    if n_quali > 0:
        #check if two columns have the same categories
        X_quali = revaluate_cat_variable(X=X_quali) 
        #recode supplementary qualitative variables
        rec = recodecat(X=X_quali)
        X_quali, dummies = rec.X, rec.dummies
        #proportion of supplementary levels
        p_k = dummies.mul(self.call_.ind_weights,axis=0).sum(axis=0)
        #standardization (z_ik=(y_ik/p_k)-1)
        Z_quali = dummies.div(p_k,axis=1).sub(1)
        #concatenate
        Z = concat((Z,Z_quali),axis=1)

    #coordinates of the supplementary variables
    coord = Z.mul(self.call_.ind_weights,axis=0).T.dot(self.svd_.U)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 of the supplementary variables
    sqdisto  = Z.pow(2).mul(self.call_.ind_weights,axis=0).sum(axis=0)
    sqdisto.name = "Sq. Dist."
    #cos2 of the supplementary variables
    sqcos = coord.pow(2).div(sqdisto,axis=0)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for the supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quanti > 0:
        #convert to namedtuple
        quanti_ = namedtuple("quanti_sup",["coord","cos2"])(coord.iloc[:n_quanti,:], sqcos.iloc[:n_quanti,:])
    else:
        quanti_ = None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for the supplementary qualitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quali > 0:
        #coordinates, cos2 and dist2 of supplementary levels
        levels_coord, levels_sqcos, levels_sqdisto = coord.iloc[n_quanti:,:], sqcos.iloc[n_quanti:,:], sqdisto.iloc[n_quanti:]
        #normalized coordinates of the supplementary levels - barycenter
        levels_coord_n = levels_coord.mul(self.svd_.vs[:self.call_.n_components],axis=1)
        #vtest of the supplementary levels
        levels_vtest = levels_coord.mul(sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0)
        #eta2 of the supplementary qualitative variables
        quali_var_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=self.call_.ind_weights,excl=self.call_.excl)
        #convert to namedtuple
        quali_ = namedtuple("quali_sup",["coord","coord_n","cos2","vtest","eta2","dist2"])(levels_coord,levels_coord_n,levels_sqcos,levels_vtest,quali_var_sqeta,levels_sqdisto)
    else:
        quali_ = None
    
    #convert to namedtuple
    return namedtuple("supvarMCAResult",["quanti","quali"])(quanti_,quali_)