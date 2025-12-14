# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, number, sqrt, average, cov, linalg
from pandas import DataFrame, Series, concat
from pandas.api.types import is_numeric_dtype
from collections import namedtuple, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.summarize import conditional_wmean
from .functions.gfa import gfa
from .functions.gsvd import gsvd
from .functions.predict_sup import predict_sup
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.function_eta2 import function_eta2
from .functions.summarize import summarize
from .functions.association import association
from .functions.corrmatrix import corrmatrix
from .functions.utils import is_dataframe

class wcPCA(BaseEstimator,TransformerMixin):
    """
    Within-class Principal Component Analysis (wcPCA)
    -------------------------------------------------

    Description
    -----------
    Performs Within-class Principal Component Analysis (wcPCA) with supplementary individuals and/or supplementary variables (quantitative and/or categorical). Missing values are replaced by the column mean.

    Details
    -------
    Within-class Principal Component Analysis is a principal component analysis where the active variables are centered on the mean of their class/group instead of the overall mean.

    Usage
    -----
    ```python
    >>> wcPCA(group = None, standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, var_sup = None)
    ```

    Parameters
    ----------
    `group`: an integer or a string indicating the indexes or names of the class/group variable.

    `standardize`: a boolean, default = True
        * If `True`: the data are scaled to unit variance.
        * If `False`: the data are not scaled to unit variance.

    `n_components`: an integer indicating th number of dimensions kept in the results (by default 5)

    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.

    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `var_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary variables (quantitative and/or qualitative).
    
    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True`: parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False`: parallelize using pandas apply

    Attributes
    ----------
    `ratio_`: a numeric specifying the between-class inertia percentage.

    `call_`: namedtuple with some informations
        * `Xtot`: pandas DataFrame with all data
        * `X`: pandas DataFrame with active elements
        * `x`: pandas DataFrame with numeric data
        * `y`: pandas Series with class variables 
        * `group`: string indicating the name of the class/group variable
        * `n_components`: integer indicating the number of components kept
        * `ind_weights`: pandas Series containing individuals weights
        * `var_weights`: pandas Series containing variables weights
        * `center`: pandas DataFrame with mean by group
        * `n_workers`: integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals
        * `sup_var`: None or a list of string indicating names of the supplementary variables
        * `pca`: an instance of class PCA
    
    `svd_`: namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD)
        * `vs`: 1D numpy array containing the singular values
        * `U`: 2D numpy array whose columns contain the left singular vectors
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `ind_`: `namedtuple of pandas DataFrames containing all the results for the active individuals.
        * `coord`: factor coordinates (scores) of the individuals,
        * `cos2`: squared cosinus (cos2) of the individuals,
        * `contrib`: relative contributions of the individuals,
        * `infos`: additionals informations (weight, squared distance to origin and inertia) of the individuals.

    `var_`: namedtuple of pandas DataFrames containing all the results for the variables:
        * `coord`: factor coordinates (scores) of the variables,
        * `cos2`: squared cosinus (cos2) of the variables,
        * `contrib`: relative contributions of the variables,
        * `infos`: additionals informations (weight, squared distance to origin and inertia) of the variables.

    `ind_sup_`: namedtuple of pandas DataFrames/Series containing all the results for the supplementary individuals.
        * `coord`: factor coordinates (scores) of the supplementary individuals,
        * `cos2`: squared cosinus (cos2) of the supplementary individuals,
        * `dist2`: squared distance to origin (dist2) of the supplementary individuals.

    `quanti_sup_`: namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables.
        * `coord`: factor coordinates (scores) of the supplementary quantitative variables,
        * `cos2`: squared cosinus (cos2) of the supplementary quantitative variables.

    `quali_sup_`: namedtuple of pandas DataFrames/Series containing all the results for the supplementary categorical variables.
        * `coord`: factor coordinates (scores) of the supplementary levels,
        * `cos2`: squared cosinus (cos2) of the supplementary levels,
        * `vtest`: value-test (vtest) of the supplementary levels, which is a criterion with a Normal distribution,
        * `dist2`: squared distance to origin (dist2) of the supplementary levels,
        * `eta2`: squared correlation ratio (eta2) of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension.
    
    `summary_quanti_`: descriptive statistics for quantitative variables

    `summary_quali_`: frequencies distribution for categorical variable

    `model_`: string specifying the model fitted = 'wcpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Bry X. (1996), Analyses factorielles multiple, Economica

    * Lebart L., Morineau A. et Warwick K., 1984, Multivariate Descriptive Statistical Analysis, John Wiley and sons, New-York.)

    See Also
    --------
    `get_wcpca_ind`, `get_wcpca_var`, `get_wcpca`, `summarywcPCA`, `fviz_wcpca_ind`, `fviz_wcpca_var`

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import meaudret
    >>> from scientisttools import wcPCA, summarywcPCA
    >>> res_wcpca = WithinPCA(group=9,sup_var=range(10,24))
    >>> res_wcpca.fit(meaudret)
    >>> summarywcPCA(res_wcpca)
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
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

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
        #Within-class principal components analysis (wcPCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into y and x
        y, x = X[grp_label], X.drop(columns=grp_label)

        if not all(is_numeric_dtype(x[k]) for k in x.columns): #check if all variables in x are numerics
            raise TypeError("All columns in x must be numerics")
        
        #check if y is categorics
        if not all(isinstance(kq, str) for kq in y):
            raise TypeError("y must be categorics")
        
        #number of rows and columns
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

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=x.index,name="weight"), Series(var_weights,index=x.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: Z = (X - mu)/sigma
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #conditional weighted average
        center = conditional_wmean(X=x,Y=y,weights=ind_weights)
        #center by group average
        Xc = x.sub(center.loc[y.values,:].values)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: Z = (X - mu)/sigma
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and weighted standard deviation
        xc_center = average(Xc,axis=0,weights=ind_weights)
        if self.standardize:
            xc_scale = array([sqrt(cov(Xc.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)])
        else:
            xc_scale = ones(n_cols)
        #convert to Series
        xc_center, xc_scale = Series(xc_center,index=Xc.columns,name="center"), Series(xc_scale,index=Xc.columns,name="scale")
        
        #standardization: Z = (X - mu)/sigma
        Z = Xc.sub(xc_center,axis=1).div(xc_scale,axis=1)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1, n_cols))
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
        call_ = OrderedDict(Xtot=Xtot,X=X,Xc=Xc,Z=Z,group=grp_label,ind_weights=ind_weights,var_weights=var_weights,center=center,xc_center=xc_center,xc_scale=xc_scale,
                            n_components=n_components,max_components=max_components,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Z,row_weights=ind_weights,col_weights=var_weights,max_components=max_components,n_components=n_components)

        #extract elements
        self.svd_, self.eig_, ind_, var_ = fit_.svd, fit_.eig, fit_.row, fit_.col

        #convert to namedtuple
        self.ind_, self.var_ = namedtuple("ind",ind_.keys())(*ind_.values()), namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #ratio
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and weighted standard deviation
        x_center = average(x,axis=0,weights=ind_weights)
        if self.standardize:
            x_scale = array([sqrt(cov(x.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)])
        else:
            x_scale = ones(n_cols)
        #convert to Series
        x_center, x_scale = Series(x_center,index=x.columns,name="center"), Series(x_scale,index=x.columns,name="scale")
        
        #standardization: z = (x - mu)/sigma
        z = x.sub(x_center,axis=1).div(x_scale,axis=1)
        
        #QR decomposition (to set maximum number of components)
        z_Q, z_R = linalg.qr(z)
        z_max_components = int(min(linalg.matrix_rank(z_Q),linalg.matrix_rank(z_R), n_rows - 1, n_cols))
    
        #compute weighted average and weighted standard deviation
        pca = gsvd(X=z,row_weights=ind_weights,col_weights=var_weights,n_components=z_max_components)

        #ratio
        self.ratio_ = sum(fit_.eig["Eigenvalue"])/sum(pca.vs[:z_max_components]**2)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split X into x and y
            y_ind_sup, x_ind_sup = X_ind_sup[self.call_.group], X_ind_sup.drop(columns=self.call_.group) 
            #standardization: Z = (X - mu)/sigma
            Z_ind_sup = x_ind_sup.sub(self.call_.center.loc[y_ind_sup.values,:].values).sub(xc_center,axis=1).div(xc_scale,axis=1)
            #statistics for supplementary individuals
            ind_sup_ = predict_sup(X=Z_ind_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

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
                #conditional weighted average
                center_sup = conditional_wmean(X=X_quanti_sup,Y=y,weights=self.call_.ind_weights)
                #center
                Xc_quanti_sup = X_quanti_sup.sub(center_sup.loc[y.values,:].values)
                #compute weighted average for supplementary quantitative variables
                xc_center_sup = average(Xc_quanti_sup,axis=0,weights=ind_weights)
                if self.standardize:
                    xc_scale_sup = array([sqrt(cov(Xc_quanti_sup.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_quanti_sup)])
                else:
                    xc_scale_sup = ones(n_quanti_sup)
                #convert to pandas Series
                xc_center_sup, xc_scale_sup = Series(xc_center_sup,index=Xc_quanti_sup.columns,name="center"), Series(xc_scale_sup,index=Xc_quanti_sup.columns,name="scale")
                #standardization : Z = (X - mu)/sigma
                Z_quanti_sup = Xc_quanti_sup.sub(xc_center_sup,axis=1).div(xc_scale_sup,axis=1)
                #statistics for supplementary quantitative variables
                quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=fit_.svd.U,weights=ind_weights,axis=1)
                del quanti_sup_['dist2'] #delete dist2
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    
            #statistics for supplementary qualitative variables
            if n_quali_sup > 0:
                #recode
                rec = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec.X, rec.dummies
                #conditional mean - Barycenter of original data
                X_levels_sup = conditional_wmean(X=Xc,Y=X_quali_sup,weights=ind_weights)
                #standardization: Z = (X - mu)/sigma
                Z_levels_sup = X_levels_sup.sub(xc_center,axis=1).div(xc_scale,axis=1)
                #statistics for supplementary levels
                quali_sup_ = predict_sup(X=Z_levels_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
                #vtest for the supplementary levels
                p_k_sup = dummies_sup.mul(ind_weights,axis=0).sum(axis=0)
                levels_sup_vtest = quali_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(fit_.svd.vs[:n_components],axis=1)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=fit_.row["coord"],weights=ind_weights,excl=None)
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
                
        self.model_ = "wcpca"
        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y`: None
            y is ignored
        
        Returns
        -------
        `X_new`: a pandas DataFrame of shape (n_samples, n_components) with numeric variables
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def inverse_transform(self,X:DataFrame) -> DataFrame:
        """
        Transform data back to its original space
        -----------------------------------------

        Description
        -----------
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: a pandas DataFrame of shape (n_samples, n_columns)
            Original data, where `n_samples` is the number of samples and `n_columns` is the number of columns
        """
        is_dataframe(X=X) #check if X is an instance of class pd.DataFrame

        y, x = X.loc[:,self.call_.group], X.drop(columns=self.call_.group) #split X into x and y
        
        n_components = min(x.shape[1],self.call_.n_components) #set number of components
        eigvals = self.var_.coord.pow(2).T.dot(self.call_.var_weights)[:n_components] #eigen values
        #inverse transform
        X_original = x.iloc[:,:n_components].dot(self.var_.coord.iloc[:,:n_components].div(sqrt(eigvals),axis=1).T)
        #applyy principal component analysis transformation
        X_original = X_original.mul(self.call_.xc_scale,axis=1).add(self.call_.xc_center,axis=1)
        #apply within principal component analysis transformation and concatenate
        X_original = concat((X_original.add(self.call_.center.loc[y.values,:].values),y),axis=1)
        return X_original
    
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
        is_dataframe(X=X) #check if X is an instance of class pd.DataFrame

        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == ncols
            raise ValueError("'columns' aren't aligned")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the wcPCA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns
        
        y, x = X[self.call_.group], X.drop(columns=self.call_.group) #split X into x and y

        if not all(is_numeric_dtype(x[k]) for k in x.columns): #check if all variables in x are numerics
            raise TypeError("All columns in data must be numerics")
        
        if not all(isinstance(x, str) for x in y): #check if y is categorics
            raise TypeError("y must be categorics")
        
        #standardisation: z = (x - mu)/sigma
        Z = x.sub(self.call_.center.loc[y.values,:].values).sub(self.call_.xc_center,axis=1).div(self.call_.xc_scale,axis=1)
        #transition relation
        coord = Z.mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord