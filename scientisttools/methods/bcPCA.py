# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, average, cov, sqrt, linalg, number
from pandas import DataFrame, Series, get_dummies
from pandas.api.types import is_numeric_dtype
from collections import namedtuple, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.gfa import gfa
from .functions.gsvd import gsvd
from .functions.summarize import conditional_wmean
from .functions.predict_sup import predict_sup
from .functions.recodecat import recodecat
from .functions.function_eta2 import function_eta2
from .functions.splitmix import splitmix
from .functions.association import association
from .functions.summarize import summarize
from .functions.corrmatrix import corrmatrix

class bcPCA(BaseEstimator,TransformerMixin):
    """
    Between-class Principal Component Analysis (bcPCA)
    --------------------------------------------------

    Description
    -----------
    Performs Between-class Principal Component Analysis (bcPCA) with supplementary individuals and/or supplementary variables. Missing values are replaced by the column mean.

    Details
    -------
    Between-class Principal Component Analysis consists in two steps : 
    1. Computation of the barycenter of data rows for each category of classe 
    2. Principal Component Analysis of the set of barycenters

    Usage
    -----
    ```python
    >>> bcPCA(group = None, standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, sup_var = None)
    ```

    Parameters
    ----------
    `group`: an integer or a string indicating the indexes or the names of the class variable.

    `standardize`: a boolean, default = True
        * If `True`: the data are scaled to unit variance.
        * If `False`: the data are not scaled to unit variance.

    `n_components`: number of dimensions kept in the results (by default 5)
    
    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1 for uniform individuals weights)
    
    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True`: parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False`: parallelize using pandas apply

    Attributes
    ----------
    `ratio_`: a numeric specifying the between-class inertia percentage.

    `call_`: a namedtuple with some informations, including
        * `Xtot`: pandas DataFrame with all data (numeric and class)
        * `x`: pandas DataFrame with numeric data
        * `y`: pandas Series with class variables 
        * `n_components`: integer indicating the number of components kept
        * `ind_weights`: pandas Series containing individuals weights
        * `n_workers`: integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
    
    `svd_`: a namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD), including
        * `vs`: 1-D numpy array containing the singular values
        * `U`: 2-D numpy array whose columns contain the left singular vectors
        * `V`: 2-D numpy array whose columns contain the right singular vectors.

    `eig_`: pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `group_`: a namedtuple of pandas DataFrames containing all the results for the groups, including:
        * `coord`: coordinates of the groups,
        * `cos2`: squared cosinus of the groups,
        * `contrib`: relative contributions of the groups,
        * `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the groups.

    `var_`: a namedtuple of pandas DataFrames containing all the results for the variables, including:
        * `coord`: coordinates of the variables,
        * `cos2`: squared cosinus of the variables,
        * `contrib`: relative contributions of the variables,
        * `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.

    `ind_`: a namedtuple of pandas DataFrames/Series containing all the results for the active individuals, including:
        * `coord`: coordinates of the individuals
        * `cos2`: squared cosinus of the individuals
        * `infos`: additionals informations (weight and squared distance to origin) of the individuals.
    
    `ind_sup_`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary individuals, including:
        * `coord`: coordinates of the supplementary individuals, 
        * `cos2`: squared cosinus of the supplementary individuals,
        * `dist2`: squared distance to origin of the supplementary individuals.

    `quanti_sup_`: a namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables, including:
        * `coord`: coordinates of the supplementary quantitative variables, 
        * `cos2`: squared cosinus of the supplementary quantitative variables.

    `quali_sup_`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary qualitative variables/levels, including:
        * `coord`: coordinates of the supplementary levels,
        * `cos2`: squared cosinus of the supplementary levels,
        * `vtest`: value-test (which is a criterion with a Normal distribution) of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension
        * `dist2`: squared distance to origin of the supplementary levels.
    
    `model_`: a string indicating the model fitted = 'bcpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Bry X. (1996), Analyses factorielles multiple, Economica

    * Lebart L., Morineau A. et Warwick K., 1984, Multivariate Descriptive Statistical Analysis, John Wiley and sons, New-York.)

    See Also
    --------
    `get_bcpca_ind`, `get_bcpca_var`, `get_bcpca_group`, `get_bcpca`, `summarybcPCA`, `fviz_bcpca_ind`, `fviz_bcpca_var`, `fviz_bcpca_group`

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, summarybcPCA
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> summarybcPCA(res_bcpca)
    ```
    """
    def __init__(self,
                 group = None,
                 standardize = True,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 sup_var = None):
        self.group = group
        self.standardize = standardize
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.sup_var = sup_var

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (quantitative and qualitative).

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
        #check if standardize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.standardize,bool):
            raise TypeError("'standardize' must be a boolean.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary individuals labels
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)

        #get supplementary variables labels
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set group variables label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            raise ValueError("'group' must be assigned.")  
        elif isinstance(self.group,str):
            grp_label =  self.group
        elif isinstance(self.group,(int,float)):
            grp_label = X.columns[int(self.group)]
        else:
            raise TypeError("'group' must be either a string or an integer.")

        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Between-class principal components analysis (bcPCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into y and x
        y, x = X[grp_label], X.drop(columns=grp_label)

        if not all(is_numeric_dtype(x[k]) for k in x.columns): #check if all variables in x are numerics
            raise TypeError("All columns in x must be numerics")
        
        #check if y is categorics
        if not all(isinstance(kq, str) for kq in y):
            raise TypeError("y must be categorics")
        
        #number of rows/columns
        n_rows, n_cols = x.shape
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])

        #set variables weights
        if self.var_weights is None:
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list or a tuple or a 1-D array or a pandas Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to series
        ind_weights, var_weights =  Series(ind_weights,index=x.index,name="weight"), Series(var_weights,index=x.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z = (x - mu)/sigma
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        center = average(x,axis=0,weights=ind_weights)
        if self.standardize:
            scale = array([sqrt(cov(x.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)])
        else:
            scale = ones(x.shape[1])
        #convert to Series
        center, scale = Series(center,index=x.columns,name="center"), Series(scale,index=x.columns,name="scale")
        
        #standardization : z = (x - mu)/sigma
        Z = x.sub(center,axis=1).div(scale,axis=1)
        
        #disjunctive table
        dummies = get_dummies(y,prefix=None,dtype=int)
        #number of levels and levels weights
        n_levels, levels_weights = dummies.shape[1], dummies.mul(ind_weights,axis=0).sum(axis=0)
        levels_weights.name = "weight"
        #barycenter
        X_levels = conditional_wmean(X=Z,Y=y,weights=ind_weights)
        #apply non-normed principal component analysis
        levels_center, levels_scale = average(X_levels,axis=0,weights=levels_weights), ones(n_cols)
        #convert to Series
        levels_center, levels_scale = Series(levels_center,index=X_levels.columns,name="center"), Series(levels_scale,index=X_levels.columns,name="scale")
        #standardization: z = (x - mu)/sigma
        Z_levels = X_levels.sub(levels_center,axis=1).div(levels_scale,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z_levels)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_levels - 1, n_cols))
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
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,group=grp_label,ind_weights=ind_weights,var_weights=var_weights,levels_weights=levels_weights,center=center,scale=scale,levels_center=levels_center,levels_scale=levels_scale,
                            n_components=n_components,max_components=max_components,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Z_levels,row_weights=levels_weights,col_weights=var_weights,max_components=max_components,n_components=n_components)

        #extract elements
        self.svd_, self.eig_, group_, var_ = fit_.svd, fit_.eig, fit_.row, fit_.col

        #convert to namedtuple
        self.group_, self.var_ = namedtuple("group",group_.keys())(*group_.values()), namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: Z = (X - mu)/sigma
        Z_ind = Z.sub(levels_center,axis=1).div(levels_scale,axis=1)
        #statistics for individuals
        ind_ = predict_sup(X=Z_ind,Y=fit_.svd.V,weights=var_weights,axis=0)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #ratio
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Z_Q, Z_R = linalg.qr(Z)
        Z_max_components = int(min(linalg.matrix_rank(Z_Q),linalg.matrix_rank(Z_R), n_rows - 1, n_cols))
    
        #compute weighted average and weighted standard deviation
        pca = gsvd(X=Z,row_weights=ind_weights,col_weights=var_weights,n_components=Z_max_components)

        #ratio
        self.ratio_ = sum(fit_.eig["Eigenvalue"])/sum(pca.vs[:Z_max_components]**2)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #drop group label
            X_ind_sup = X_ind_sup.drop(columns=grp_label)
            #standardization: Z = (X-mu)/sigma
            Z_ind_sup = X_ind_sup.sub(center,axis=1).div(scale,axis=1).sub(levels_center,axis=1).div(levels_scale,axis=1)
            #statistics for individuals
            ind_sup_ = predict_sup(X=Z_ind_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind",ind_sup_.keys())(*ind_sup_.values())
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            if self.ind_sup is not None:
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2
            
            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0:
                #compute weighted average and weighted standard deviation for supplementary quantitative variables
                center_sup = average(X_quanti_sup,axis=0,weights=ind_weights)
                if self.standardize:
                    scale_sup = array([sqrt(cov(X_quanti_sup.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_quanti_sup)])
                else:
                    scale_sup = ones(n_quanti_sup)
                #convert to pandas Series
                center_sup, scale_sup = Series(center_sup,index=X_quanti_sup.columns,name="center"), Series(scale_sup,index=X_quanti_sup.columns,name="scale")
                #standardization : Z = (X - mu)/sigma
                Z_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
                #conditional weighted average
                X_levels_quanti_sup = conditional_wmean(X=Z_quanti_sup,Y=y,weights=ind_weights)
                #compute weighted average and weighted standard deviation for supplementary conditional
                levels_center_quanti_sup = Series(average(X_levels_quanti_sup,axis=0,weights=levels_weights),index=X_levels_quanti_sup.columns,name="center")
                levels_scale_quanti_sup = Series(ones(n_quanti_sup),index=X_levels_quanti_sup.columns,name="scale")
                #standardization: Z = (X - mu)/sigma
                Z_levels_quanti_sup = X_levels_quanti_sup.sub(levels_center_quanti_sup,axis=1).div(levels_scale_quanti_sup,axis=1)
                #statistics for supplementary quantitative variables
                quanti_sup_ = predict_sup(X=Z_levels_quanti_sup,Y=fit_.svd.U,weights=levels_weights,axis=1)
                del quanti_sup_['dist2'] #delete dist2
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #recode supplementary qualitative variables
                rec = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec.X, rec.dummies
                #conditional mean - Barycenter of original data
                X_levels_sup = conditional_wmean(X=Z,Y=X_quali_sup,weights=ind_weights)
                #standardization: Z = (X - mu)/sigma
                Z_levels_sup = X_levels_sup.sub(levels_center,axis=1).div(levels_scale,axis=1)
                #statistics for supplementary levels
                quali_sup_ = predict_sup(X=Z_levels_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
                #vtest for the supplementary levels
                p_k_sup = dummies_sup.mul(ind_weights,axis=0).sum(axis=0)
                levels_sup_vtest = quali_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(fit_.svd.vs[:n_components],axis=1)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=ind_["coord"],weights=ind_weights,excl=None)
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

                #descriptive descriptive of qualitative variables
                self.summary_quali_ = summarize(X=X_quali_sup)

                #degree of association - multivariate goodness
                if n_quali_sup > 1:
                    self.association_ = association(X=X_quali_sup) 

        #all quantitative variables in original dataframe
        all_quanti = Xtot.select_dtypes(include=number)
        #drop supplementary individuals
        if self.ind_sup is not None:
            all_quanti = all_quanti.drop(index=ind_sup_label)
        #descriptive statistics of quantitatives variables 
        self.summary_quanti_ = summarize(X=all_quanti)

        #correlation tests
        all_vars = Xtot.copy()
        if self.ind_sup is not None:
            all_vars = all_vars.drop(index=ind_sup_label)
        self.corrtest_ = corrmatrix(X=all_vars,weights=ind_weights)

        self.model_ = "bcpca"
        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (dependent and class).

        `y`: None
            y is ignored
        
        Returns
        -------
        `X_new`: a pandas DataFrame of shape(n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
            raise ValueError("'columns' aren't aligned")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the bcPCA result")
        X = X.loc[:,self.call_.X.columns].drop(columns=self.call_.group) #reorder columns and drop group label

        if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
            raise TypeError("All columns must be numeric")

        #standardisation and apply transition relation
        coord = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).sub(self.call_.levels_center,axis=1).div(self.call_.levels_scale,axis=1).mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord