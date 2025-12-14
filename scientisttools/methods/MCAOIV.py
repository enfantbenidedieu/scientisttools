# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, number
from pandas import DataFrame, Series, Categorical, api, concat, get_dummies
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.api as sm
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.recodecont import recodecont
from .functions.summarize import summarize
from .functions.splitmix import splitmix
from .MCA import MCA
from .PCA import PCA

class MCAOIV(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis with Orthogonal Instrumental Variables (MCAOIV)
    --------------------------------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Multiple Correspondence Analysis with Orthogonal Instrumental Variables consists in three steps: 
    1. (Specific) Multiple Correspondence Analysis of dependent variables, keeping all the dimensions of the space 
    2. Computation of one linear regression for each dimension in the (specific) Multiple Correspondence Analysis, with individual coordinates as response and all variables instrumental variables as explanatory variables. 
    3. Principal Component Analysis of the set of residuals from the regressions in 2.

    Usage
    -----
    ```python
    >>> MCAOIV(iv = None, excl=None, n_components = 5, ind_weights = None, parallelize=False)
    ```

    Parameters
    ----------
    `iv`: an integer or a list/tuple of string specifying the name of the instrumental (explanatory) variables (quantitative and/or qualitative).

    `excl`: an integer or a list indicating the "junk" categories (by default None). It can be a list/tuple of the names of the categories or a list/tuple of the indexes in the disjunctive table.
    
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
        * `excl`: None or a list of string indicating names of the excluded categories
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

    `model_`: string specifying the model fitted = 'mcaiv'

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
    `get_mcaoiv_ind`, `get_mcaoiv_var`, `get_mcaoiv`, `summaryMCAOIV`, `fviz_mcaoiv_ind`, `fviz_mcaoiv_var`, `fviz_mcaoiv_biplot`

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCAOIV, summaryMCAOIV
    >>> #multiple correspondence analysis with orthogonal instrumental variables
    >>> res_mcaoiv = MCAOIV(iv=(0,1,2,3))
    >>> res_mcaoiv.fit(poison)
    >>> summaryMCAOIV(res_mcaoiv)
    >>> #specific multiple correspondence analysis with orthogonal instrumental variables
    >>> res_spemcaoiv = MCAOIV(iv=(0,1),excl=(0,2))
    >>> res_spemcaoiv.fit(poison)
    >>> summaryMCAOIV(res_spemcaoiv)
    ```
    """
    def __init__(self,
                 iv = None,
                 excl = None,
                 n_components = 5,
                 ind_weights = None,
                 parallelize = False):
        self.iv = iv
        self.excl = excl
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.parallelize = parallelize

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (qualitative and/or qualitative).

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself
        
        Examples
        --------
        ```python
        >>> from scientisttools.datasets import poison
        >>> from scientisttools import MCAIV
        >>> res_mcaiv = MCAIV(iv=(0,1,2,3))
        >>> res_mcaiv.fit(poison)
        ```
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of pd.DataFrame class
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if parallelize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.parallelize,bool):
            raise TypeError("'parallelize' must be a boolean.")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #checks if categoricals variables is in X
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for q in is_quali.columns:
                X[q] = Categorical(X[q],categories=sorted(X[q].dropna().unique().tolist()),ordered=True)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #checks if quantitative variables is in X - fill NA with mean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=number)
        if is_quanti.shape[1]>0:
            for k in is_quanti.columns:
                X[k] = recodecont(X[k]).X

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

        #split X into explanatory (instrumental) variables and dependent variables
        z, x = X.loc[:,iv_label], X.drop(columns=iv_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all variables are categorics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        all_cat = all(api.types.is_string_dtype(x[k]) for k in x.columns)
        if not all_cat:
            raise TypeError("Dependent variables should all be categorics.")
        
        #number of rows/columns
        n_rows = x.shape[0]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list/tuple/array/Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array/Series with length {n_rows}.")
        else:
            ind_weights = array(self.ind_weights)

        #convert weights to series
        ind_weights =  Series(ind_weights,index=x.index,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize quantitative variables in X
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        zs = z.copy()
        for k in zs.columns:
            if api.types.is_numeric_dtype(zs[k]):
                #compute weighted average and standard deviation
                d = DescrStatsW(zs[k],weights=ind_weights,ddof=0)
                #standardization : Z = (X - mu)/sigma
                zs[k] = mapply(zs[k].to_frame(),lambda x : (x - d.mean)/d.std,axis=1,progressbar=False,n_workers=n_workers)

        #multiple correspondence analysis (MCA)
        res_mca0 = MCA(excl=self.excl).fit(x)
        ncomp = int((res_mca0.eig_.iloc[:,0].values > 1e-10).sum())
        res_mca1 = MCA(excl=self.excl,n_components=ncomp).fit(x)
        coord = res_mca1.ind_.coord

        #ordinary least squared with instrumental variables
        def olsoiv(i):
            def x_cast(k):
                if api.types.is_numeric_dtype(zs[k]):
                    return zs[k]
                if api.types.is_string_dtype(zs[k]):
                    return get_dummies(zs[k],drop_first=True,dtype=int)
            features = concat(map(lambda k : x_cast(k=k),zs.columns),axis=1)
            ols = sm.WLS(endog=coord[i].astype(float),exog=sm.add_constant(features),weights=ind_weights).fit()
            return Series(ols.resid,index=coord.index,name=i)
        yhat = concat(map(lambda i : olsoiv(i=i),coord.columns),axis=1) 

        #apply Principal Component Analysis (PCA)
        res = PCA(standardize=False,n_components=self.n_components,ind_weights=ind_weights,sup_var=x.columns.tolist(),rotate=None).fit(concat((yhat,x),axis=1))
        pca = PCA(standardize=False,n_components=self.n_components,ind_weights=ind_weights,rotate=None).fit(coord)

        #ratio
        self.ratio_ = sum(res.eig_["Eigenvalue"])/sum(pca.eig_["Eigenvalue"])

        #store call informations
        call_ = OrderedDict(Xtot=X,X=x,Y=y,iv=iv_label,excl=res_mca1.call_.excl,n_components=res.call_.n_components,ind_weights=ind_weights,n_workers=n_workers)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #Extract all importants attributes
        self.svd_, self.eig_, self.ind_, self.var_, self.quali_sup_ = res.svd_, res.eig_, res.ind_, res.var_, res.quali_sup_

        #descriptive statistics
        split_X = splitmix(X=X)
        self.summary_quali_ = summarize(X=split_X.quali)

        #add supplementary quantitative variables informations
        if hasattr(res,"quanti_sup_"):
            self.quanti_sup_, self.summary_quanti_ = res.quanti_sup_, summarize(X=split_X.quanti)

        self.model_ = "mcaiv"
        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (quantitative and/or qualitative).

        `y`: None
            y is ignored
        
        Returns
        -------
        `X_new`: pandas DataFrame with numeric variables
            Transformed values.
        
        Examples
        --------
        ```python
        >>> from scientisttools.datasets import poison
        >>> from scientisttools import MCAIV
        >>> res_mcaiv = MCAIV(iv=(0,1,2,3))
        >>> ind_coord = res_mcaiv.fit_transform(poison)
        ```
        """
        self.fit(X)
        return self.ind_.coord