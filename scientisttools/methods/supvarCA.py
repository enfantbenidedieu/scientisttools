# -*- coding: utf-8 -*-
from numpy import average, cov, sqrt
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from collections import namedtuple, OrderedDict
from typing import NamedTuple

#intern functions
from .functions.predict_sup import predict_sup
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.summarize import conditional_sum
from .functions.function_eta2 import function_eta2

def supvarCA(self,X,Y) -> NamedTuple:
    """
    Supplementary columns/variables projection with Correspondence Analysis
    -----------------------------------------------------------------------

    Description
    -----------
    Performns the coordinates, squared cosinus and squared distance to origin of supplementary columns/variables with Correspondence Analysis

    Usage
    -----
    ```python
    >>> supvarCA(self,X,Y)   
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `X`: a pandas DataFrame of supplementary columns

    `Y`: a pandas DataFrame of supplementary variables

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary columns/variables including : 

    `col`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary columns, including:
        * `coord`: coordinates of supplementary columns,
        * `cos2`: squared cosinus of supplementary columns,
        * `dist2`: squared distance to origin of supplementary columns.

    `quanti`: a namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables, including:
        * `coord`: coordinates of supplementary quantitative variables,
        * `cos2`: squared cosinus of supplementary quantitative variables.
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary qualitative variables/levels, including:
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
    >>> from scientisttools.datasets import load_children
    >>> from scientisttools import CA, supvarCA
    >>> children = load_children()
    >>> res_ca = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> res_ca.fit(children)
    >>> #supplementary columns/variables (quantitative and qualitative)
    >>> X_col_sup, X_sup_var = load_children("col_sup"), load_children("sup_var")
    >>> sup_var_predict = supvarCA(res_ca,X=X_col_sup,Y=X_sup_var,axis=1))
    >>> #extract supplementary columns informations
    >>> col_sup = sup_var_predict.col
    >>> col_sup.coord #coordinates of supplementary columns
    >>> col_sup.cos2 #cos2 of supplementary columns
    >>> col_sup.dist2 #dist2 of supplementary columns
    >>> #extract supplementary quantitative variables informations
    >>> quanti_sup = sup_var_predict.quanti
    >>> quanti_sup.coord #coordinates of supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 of supplementary quantitative variables
    >>> #extract supplementary qualitative variables informations
    >>> quali_sup = sup_var_predict.quali
    >>> quali_sup.coord #coordinates of supplementary levels
    >>> quali_sup.cos2 #cos2 of supplementary levels
    >>> quali_sup.vtest #vtest of supplementary levels
    >>> quali_sup.dist2 #dist2 of supplementary levels
    >>> quali_sup.eta2 #eta2 of supplementary qualitative variables
    ```
    """
    if self.model_ != "ca": #check if self is an object of class CA
        raise TypeError("'self' must be an object of class CA")
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary columns
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X is not None:
        if isinstance(X,Series): #if pandas Series, convert to pandas DataFrame
            X = X.to_frame()
        
        if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if X.shape[0] != self.call_.X.shape[0]: #check if X_col_sup.shape[0] = nrows
            raise ValueError("'rows' aren't aligned")

        if not all(is_numeric_dtype(X[j]) for j in X.columns): #check if all variables are numerics
            raise TypeError("All columns in X must be numerics")
        
        #frequencies of supplementary columns
        freq_col_sup = X.mul(self.call_.row_weights,axis=0).div(self.call_.total)
        #margins for supplementary columns
        col_sup_marge = freq_col_sup.sum(axis=0)
        #standardization: z_ij = (fij/(fi.*f.j)) - 1
        Z_col_sup = freq_col_sup.div(self.call_.row_marge,axis=0).div(col_sup_marge,axis=1).sub(1)
        #statistics for supplementary columns
        col_sup_ = predict_sup(X=Z_col_sup,Y=self.svd_.U,weights=self.call_.row_marge,axis=1)
        #convert to namedtuple
        col_sup = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())
    else:
        col_sup = None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary variables (quantitative and/or variables)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if Y is not None:
        if isinstance(Y,Series): #if pandas Series, convert to pandas dataframe
            Y = Y.to_frame()
        
        if not isinstance(Y,DataFrame): #check if Y is an instance of pd.DataFrame class
            raise TypeError(f"{type(Y)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if Y.shape[0] != self.call_.X.shape[0]: #check if Y.shape[0] = nrows
            raise ValueError("'rows' aren't aligned")
        
        split_Y = splitmix(X=Y) #split Y
        X_quanti_sup, X_quali_sup, n_rows, n_quanti_sup, n_quali_sup = split_Y.quanti, split_Y.quali, split_Y.n, split_Y.k1, split_Y.k2

        #statistics for supplementary quantitative variables
        if n_quanti_sup > 0:
            #compute weighted average and weighted standard deviation
            center_sup = Series(average(X_quanti_sup,axis=0,weights=self.call_.row_marge),index=X_quanti_sup.columns,name="center")
            scale_sup = Series([sqrt(cov(X_quanti_sup.iloc[:,k],aweights=self.call_.row_marge,ddof=0)) for k in range(n_quanti_sup)],index=X_quanti_sup.columns,name="scale")
            #standardization: Z = (X - mu)/sigma
            Z_quanti_sup = X_quanti_sup.mul(self.call_.row_weights,axis=0).sub(center_sup,axis=1).div(scale_sup,axis=1)
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=self.svd_.U,weights=self.call_.row_marge,axis=1)
            del quanti_sup_['dist2'] #delete dist2
            #convert to namedtuple
            quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
        else:
            quanti_sup = None

        #statistics for supplementary qualitative variables
        if n_quali_sup > 0:
            #recode supplementary qualitative variables
            rec = recodecat(X=X_quali_sup)
            X_quali_sup, dummies_sup = rec.X, rec.dummies
            #conditional sum
            X_row_quali_sup = conditional_sum(X=self.call_.X,Y=X_quali_sup)
            #frequencies of supplementary levels
            freq_levels_sup = X_row_quali_sup.div(self.call_.total)
            #margins for supplementary levels
            levels_sup_marge = freq_levels_sup.sum(axis=1)
            #standardization: z_ij = (fij/(fi.*f.j)) - 1
            Z_levels_sup = freq_levels_sup.div(levels_sup_marge,axis=0).div(self.call_.col_marge,axis=1).sub(1)
            #statistics for supplementary levels
            quali_sup_ = predict_sup(X=Z_levels_sup,Y=self.svd_.V,weights=self.call_.col_marge,axis=0)
            #proportion of supplementary levels
            p_k_sup = dummies_sup.mul(self.call_.row_weights,axis=0).mul(self.call_.row_marge,axis=0).sum(axis=0)
            levels_vtest = quali_sup_["coord"].mul(sqrt((self.call_.total-1)/(1/p_k_sup).sub(1)),axis=0)
            #eta2 for the supplementary qualitative variables
            quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.row_.coord,weights=self.call_.row_marge,excl=None)
            #convert to ordered dictionary
            quali_sup_ = OrderedDict(coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
            #convert to namedtuple
            quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
        else:
            quali_sup = None
    else:
        quanti_sup, quali_sup = None, None
    
    # Store all informations
    return namedtuple("supvarCAResult",["col_sup","quanti_sup","quali_sup"])(col_sup,quanti_sup,quali_sup)