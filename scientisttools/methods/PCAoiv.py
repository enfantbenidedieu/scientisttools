# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, linalg, average, sqrt, cov, number
from pandas import DataFrame, Series, concat, get_dummies
from pandas.api.types import is_numeric_dtype, is_string_dtype
import statsmodels.api as sm
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.gfa import gfa
from .functions.gsvd import gsvd
from .functions.predict_sup import predict_sup
from .functions.summarize import summarize
from .functions.corrmatrix import corrmatrix

class PCAoiv(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis with Orthogonal Instrumental Variables (PCAOIV)
    ----------------------------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Principal Component Analysis with Orthogonal Instrumental Variables consists in two steps:
    1. Computation of one linear regression for each dependent variables, with this variable as response and all instrumental variables as explanatory variables.
    2. Principal Component Analysis of the set of residuals from the regression in 1.

    Usage
    -----
    ```python
    >>> PCAOIV(iv = None, n_components = 5, ind_weights = None, parallelize=False)
    ```

    Parameters
    ----------
    `iv`: an integer or a list/tuple of string specifying the name of the instrumental (explanatory) variables (quantitative).
    
    `n_components`: number of dimensions kept in the results (by default 5).
    
    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1), the weights are given only for active individuals.

    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True`: parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False`: parallelize using pandas apply

    Attributes
    ----------
    `call_`: a namedtuple with some informations:
        * `Xtot`: a pandas DataFrame with all (dependent and instrumental) variables
        * `X`: a pandas DataFrame with dependent variables
        * `Z`: a pandas DataFrame with instrumental variables
        * `iv`: a list of string indicating names of the instrumental variables
        * `n_components`: an integer indicating the number of components kept
        * `ind_weights`: a pandas Series containing individuals weights
        * `n_workers`: an integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
    
    `svd_`: namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD)
        * `vs`: 1D numpy array containing the singular values
        * `U`: 2D numpy array whose columns contain the left singular vectors
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `ratio_`: the share of inertia explained by the instrumental variables

    `ind_`: `namedtuple of pandas DataFrames containing all the results for the active individuals.
        * `coord`: factor coordinates (scores) of the individuals
        * `cos2`: squared cosinus of the individuals
        * `contrib`: relative contributions of the individuals
        * `infos`: additionals informations (weight, squared distance to origin and inertia) of the individuals

    `var_`: namedtuple of pandas DataFrames containing all the results for the active variables
        * `coord`: factor coordinates (scores) of the variables
        * `cos2`: squared cosinus of the variables
        * `contrib`: relative contributions of the variables
        * `infos`: additionals informations (weight, squared distance to origin and inertia) of the variables

    `quanti_sup_`: namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables:
        * `coord`: factor coordinates (scores) of the supplementary quantitative variables, 
        * `cos2`: square cosinus of the supplementary quantitative variables

    `summary_quanti_`: descriptive statistics (mean, standard deviation, etc.) for quantitative variables (actives and supplementary)

    `quali_sup_`: namedtuple of pandas DataFrames containing all the results for the supplementary categorical variables:
        * `coord`: coordinates of each categories of each variables
        * `cos2`: squared cosinus of each categories of each variables
        * `dist2`: squared distance to origin of each categories of each variables
        * `vtest`: value-test (which is a criterion with a Normal distribution) of each categories of each variables
        * `eta2`: the square correlation coefficient between a qualitative variable and a dimension

    `summary_quali_`: summary statistics for qualitative variables

    `model_`: string specifying the model fitted = 'pcaoiv'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Bry X. (1996), Analyses factorielles multiple, Economica

    * Lebart L., Morineau A. et Warwick K., 1984, Multivariate Descriptive Statistical Analysis, John Wiley and sons, New-York.)

    * Lebreton, J. D., Sabatier, R., Banco G. and Bacou A. M. (1991) Principal component and correspondence analyses with respect to instrumental variables : an overview of their role in studies of structure-activity and species- environment relationships. In J. Devillers and W. Karcher, editors. Applied Multivariate Analysis in SAR and Environmental Studies, Kluwer Academic Publishers, 85--114.
    
    See Also
    --------
    `get_pcaoiv_ind`, `get_pcaoiv_var`, `get_pcaoiv`, `summaryPCAOIV`, `fviz_pcaoiv_ind`, `fviz_pcaoiv_var`, `fviz_pcaoiv_biplot`

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import rhone
    >>> from scientisttools import PCAOIV, summaryPCAOIV
    >>> res_pcaoiv = PCAOIV(iv=(15,16,17))
    >>> res_pcaoiv.fit(rhone)
    >>> summaryPCAOIV(res_pcaoiv)
    ```
    """
    def __init__(self,
                 iv = None,
                 n_components = 5,
                 ind_weights = None,
                 quanti_sup = None):
        self.iv = iv
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.quanti_sup = quanti_sup

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (quantitative and/or qualitative).

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary quantitative variables
        quanti_sup_label = get_sup_label(X=X, indexes=self.quanti_sup, axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set instrumental variables label and index
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is None:
            raise ValueError("'iv' must be assigned.")  
        elif isinstance(self.iv,str):
            iv_label =  [self.iv] 
        elif isinstance(self.iv,(int,float)):
            iv_label = [X.columns[int(self.iv)]]
        elif isinstance(self.iv,(list,tuple)):
            if all(isinstance(x,str) for x in self.iv):
                iv_label = [str(x) for x in self.iv] 
            elif all(isinstance(x,(int,float)) for x in self.iv):
                iv_label = X.columns[[int(x) for x in self.iv]].tolist()

        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary quantitative variables
        if self.quanti_sup is not None:
            X_quanti_sup, X = X.loc[:,quanti_sup_label], X.drop(columns=quanti_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal component analysis with orthogonal instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into explanatory (instrumental) variables and dependent variables
        z, x = X.loc[:,iv_label], X.drop(columns=iv_label)
        
        if not all(is_numeric_dtype(x[k]) for k in x.columns): #check if dependent variables are numerics
            raise TypeError("Dependent variables should all be numeric")
        
        #number of rows/columns
        n_rows, n_cols = x.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])

        #convert weights to series
        ind_weights, var_weights =  Series(ind_weights,index=x.index,name="weight"), Series(ones(n_cols),index=x.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: Z = (X - mu)/sigma
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        center = Series(average(x,axis=0,weights=ind_weights),index=x.columns,name = "center")
        scale = Series(array([sqrt(cov(x.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)]),index=x.columns,name = "scale")
        #standardization : Z = (X - mu)/sigma
        Xs = x.sub(center,axis=1).div(scale,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize quantitative variables in Z
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Zs = z.copy()
        for k in Zs.columns:
            if is_numeric_dtype(Zs[k]):
                #compute weighted average and weightesstandard deviation
                k_center, k_scale = average(Zs[k],weights=ind_weights), sqrt(cov(Zs[k],aweights=ind_weights,ddof=0))
                #standardization: Z = (X - mu)/sigma
                Zs[k] = Zs[k].sub(k_center).div(k_scale)

        #ordinary least squared with instrumental variables
        def olsoiv(k, x, z, weights):
            def x_cast(j):
                if is_numeric_dtype(z[j]):
                    return z[j]
                if is_string_dtype(z[j]):
                    return get_dummies(z[j],drop_first=True,dtype=int)
            features = concat((x_cast(j=j) for j in z.columns),axis=1)
            ols = sm.WLS(endog=x[k].astype(float),exog=sm.add_constant(features),weights=weights).fit()
            return Series(ols.resid,index=x.index,name=k)
        xhat = concat((olsoiv(k=k, x=Xs, z=Zs, weights=ind_weights) for k in Xs.columns),axis=1) 

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center xhat
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #apply non-normed principal component analysis
        xhat_center = Series(average(xhat,axis=0,weights=ind_weights),index=xhat.columns,name="center")
        xhat_scale = Series(ones(n_cols),index=xhat.columns,name="scale")
        #standardization: z = (x - mu)/sigma
        Z = xhat.sub(xhat_center,axis=1).div(xhat_scale,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q), linalg.matrix_rank(R), n_rows - 1, n_cols))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,Xhat=xhat,Zs=Zs,iv=iv_label,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,
                            n_components=n_components,max_components=max_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Z,row_weights=ind_weights,col_weights=var_weights,max_components=max_components,n_components=n_components)

        #extract elements
        self.svd_, self.eig_ = fit_.svd, fit_.eig

        #convert to namedtuple
        self.ind_, self.var_ = namedtuple("ind",fit_.row.keys())(*fit_.row.values()), namedtuple("var",fit_.col.keys())(*fit_.col.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #ratio
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average 
        xs_center = Series(average(Xs,axis=0,weights=ind_weights),index=Xs.columns,name="center")
        xs_scale = Series(ones(n_cols),index=Xs.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        zs = Xs.sub(xs_center,axis=1).div(xs_scale,axis=1)

        #QR decomposition (to set maximum number of components)
        zs_Q, zs_R = linalg.qr(zs)
        zs_max_components = int(min(linalg.matrix_rank(zs_Q),linalg.matrix_rank(zs_R), n_rows - 1, n_cols))
    
        #compute weighted average and weighted standard deviation
        pca = gsvd(X=zs,row_weights=ind_weights,col_weights=var_weights,n_components=zs_max_components)

        #ratio
        self.ratio_ = sum(fit_.eig["Eigenvalue"])/sum(pca.vs[:zs_max_components]**2)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            n_quanti_sup = X_quanti_sup.shape[1]
            #compute weighted average for supplementary quantitative variables
            center_sup = Series(average(X_quanti_sup,axis=0,weights=ind_weights),index=X_quanti_sup.columns,name="center")
            scale_sup = Series(array([sqrt(cov(X_quanti_sup.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_quanti_sup)]),index=X_quanti_sup.columns,name="scale")
            #standardization: Z = (X - mu)/sigma
            xs_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
            #ordinaly least squared with instrumental variables
            xhat_quanti_sup = concat((olsoiv(k=k,x=xs_quanti_sup,z=Zs,weights=ind_weights) for k in xs_quanti_sup.columns),axis=1)
            #compute weighted average for supplementary quantitative variables
            xhat_quanti_sup_center = Series(average(xhat_quanti_sup,axis=0,weights=ind_weights),index=xhat_quanti_sup.columns,name="center")
            xhat_quanti_sup_scale = Series(ones(n_quanti_sup),index=xhat_quanti_sup.columns,name="scale")
            #standardization: Z = (X - mu)/sigma
            Z_quanti_sup = xhat_quanti_sup.sub(xhat_quanti_sup_center,axis=1).div(xhat_quanti_sup_scale,axis=1)
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=fit_.svd.U,weights=ind_weights,axis=1)
            del quanti_sup_['dist2'] #delete dist2
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

        #all quantitative variables in original dataframe
        all_quanti = Xtot.select_dtypes(include=number)
        #descriptive statistics of quantitatives variables 
        self.summary_quanti_ = summarize(X=all_quanti)

        #correlation tests
        all_vars = Xtot.copy()
        self.corrtest_ = corrmatrix(X=all_vars,weights=ind_weights)

        self.model_ = "pcaoiv"
        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (quantitative and/or qualitative).

        `y`: None
            y is ignored
        
        Returns
        -------
        `X_new`: a pandas DataFrame of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord