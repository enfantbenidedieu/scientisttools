# -*- coding: utf-8 -*-
from warnings import warn
from scipy.stats import chi2
from numpy import ndarray, array, ones, sqrt, linalg, log, flip,cumsum,mean, diag, c_, real, tril, insert, diff,nan, apply_along_axis, number
from pandas import DataFrame, Series, Categorical, api, concat
from typing import NamedTuple
from collections import namedtuple,OrderedDict
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.recodecont import recodecont
from .functions.summarize import summarize
from .functions.svd_triplet import svd_triplet
from .functions.rotate_factors import rotate_factors
from .functions.pcorrcoef import pcorrcoef
from .functions.kaiser_msa import kaiser_msa
from .functions.predict_sup import predict_ind_sup, predict_quanti_sup, predict_quali_sup
from .functions.revaluate_cat_variable import revaluate_cat_variable
from .functions.conditional_average import conditional_average
from .functions.association import association

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)
    ----------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables. Missing values are replaced by the column mean.

    Usage
    -----
    ```python
    >>> PCA(standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, quanti_sup = None, quali_sup = None, rotate = 'varimax', rotation_kwargs = None, parallelize=False)
    ```
    
    Parameters
    ----------
    `standardize`: a boolean, default = True
        * If `True`: the data are scaled to unit variance.
        * If `False`: the data are not scaled to unit variance.

    `n_components`: number of dimensions kept in the results (by default 5)
    
    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `quanti_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `quali_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary categorical variables

    `rotate`: the type of rotation to performn after fitting the factor analysis model (by default 'varimax'). 
        If set to `None`, no rotation will be performed. Allowed values include: `varimax`, `promax`, `oblimin`, `oblimax`, `quartimin`, `quartimax` and `equamax`.

    `rotate_kwargs`: optional
        Dictionary containing keyword arguments for the rotation method.

    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True`: parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False`: parallelize using pandas apply

    Attributes
    ----------
    `call_`: namedtuple with some informations
        * `Xtot`: pandas DataFrame with all data (active and supplementary)
        * `X`: pandas DataFrame with active data
        * `Z`: pandas DataFrame with standardized data : Z = (X-center)/scale
        * `ind_weights`: pandas Series containing individuals weights
        * `var_weights`: pandas Series containing variables weights
        * `center`: pandas Series containing variables means
        * `scale`: pandas Series containing variables standard deviation : 
            * If `standardize = True`, then standard deviation are computed using variables standard deviation
            * If `standardize = False`, then standard deviation are a vector of ones with length number of variables.
        * `n_components`: integer indicating the number of components kept
        * `rotate`: None or string specifying the rotation method
        * `rotate_kwargs`: empty dict or dictionary containing keyword arguments for the rotation method
        * `n_workers`: integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals
        * `quanti_sup`: None or a list of string indicating names of the supplementary quantitative variables
        * `quali_sup`: None or a list of string indicating names of the supplementary qualitative variables
    
    `svd_`: namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD)
        * `vs`: 1D numpy array containing the singular values
        * `U`: 2D numpy array whose columns contain the left singular vectors
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `eigval_`: namedtuple of array containing eigen values
        * `original`: eigen values of the original matrix
        * `common`: eigen values of the common factor solution

    `vaccounted_`: pandas DataFrame containing the variance accounted

    `rotate_`: a namedtuple containing the rotation matrix and factor correlations matrix
        * `rotmat`: numpy array of rotation matrix (if a rotation has been performed. None otherwise.)
        * `phi`: pandas DataFrame of factor correlations matrix

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

    `others_`: namedtuple of others statistics (Bartlett's test of Spericity, Kaiser threshold, ...)

    `ind_sup_`: namedtuple of pandas DataFrames containing all the results for the supplementary individuals (coordinates, square cosinus)

    `quanti_sup_`: namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes, square cosinus)

    `quali_sup_`: namedtuple of pandas DataFrames containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, vtest which is a criterion with a Normal distribution, and eta2 which is the square correlation coefficient between a qualitative variable and a dimension)

    `summary_quali_`: summary statistics for supplementary qualitative variables if `quali_sup` is not None

    `chi2_test_`: chi-squared test. If supplementary qualitative are greater than 2. 

    `summary_quanti_`: summary statistics for quantitative variables (actives and supplementary)

    `model_`: string specifying the model fitted = 'pca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Bry X. (1996), Analyses factorielles multiple, Economica

    * Bry X. (1999), Analyses factorielles simples, Economica

    * Escofier B., Pagès J. (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    * Saporta G. (2006). Probabilites, Analyse des données et Statistiques. Technip

    * Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    * Rakotomalala, R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    * Tenenhaus, M. (2006). Statistique : Méthodes pour décrire, expliquer et prévoir. Dunod.
    
    See Also
    --------
    `predictPCA`, `supvarPCA`, `get_pca_ind`, `get_pca_var`, `get_pca`, `summaryPCA`, `dimdesc`, `reconst`, `fviz_pca_ind`, `fviz_pca_var`, `fviz_pca_biplot`, `fviz_pca3d_ind`

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, summaryPCA
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> summaryPCA(res_pca)
    ```
    """
    def __init__(self,
                 standardize = True,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 rotate = "varimax",
                 rotate_kwargs = dict(),
                 parallelize=False):
        self.standardize = standardize
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.rotate = rotate
        self.rotate_kwargs = rotate_kwargs
        self.parallelize = parallelize

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
        `self` : object
            Returns the instance itself
        
        Examples
        --------
        ```python
        >>> from scientisttools import decathlon, PCA, summaryPCA
        >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
        >>> res_pca.fit(decathlon)
        ```
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if X is an instance of pd.DataFrame class
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X.index.name = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if standardize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.standardize,bool):
            raise TypeError("'standardize' must be a boolean.")
        
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
        ## rop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## checks if categoricals variables is in X
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for q in is_quali.columns:
                X[q] = Categorical(X[q],categories=sorted(X[q].dropna().unique().tolist()),ordered=True)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## check if supplementary qualitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            if isinstance(self.quali_sup,str):
                quali_sup_label = [self.quali_sup]
            elif isinstance(self.quali_sup,(int,float)):
                quali_sup_label = [X.columns[int(self.quali_sup)]]
            elif isinstance(self.quali_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.quali_sup):
                     quali_sup_label = [str(x) for x in self.quali_sup]
                elif all(isinstance(x,(int,float)) for x in self.quali_sup):
                    quali_sup_label = X.columns[[int(x) for x in self.quali_sup]].tolist()
        else:
            quali_sup_label = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary quantitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            if isinstance(self.quanti_sup,str):
                quanti_sup_label = [self.quanti_sup]
            elif isinstance(self.quanti_sup,(int,float)):
                quanti_sup_label = [X.columns[int(self.quanti_sup)]]
            elif isinstance(self.quanti_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.quanti_sup):
                    quanti_sup_label = [str(x) for x in self.quanti_sup]
                elif all(isinstance(x,(int,float)) for x in self.quanti_sup):
                    quanti_sup_label = X.columns[[int(x) for x in self.quanti_sup]].tolist()
        else:
            quanti_sup_label = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if individuls supplementary
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            if isinstance(self.ind_sup,str):
                ind_sup_label = [self.ind_sup]
            elif isinstance(self.ind_sup,(int,float)):
                ind_sup_label = [X.index[int(self.ind_sup)]]
            elif isinstance(self.ind_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.ind_sup):
                    ind_sup_label = [str(x) for x in self.ind_sup]
                elif all(isinstance(x,(int,float)) for x in self.ind_sup):
                    ind_sup_label = X.index[[int(x) for x in self.ind_sup]].tolist()
        else:
            ind_sup_label = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if missing values in quantitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.isnull().any().any():
            if self.quali_sup is None:
                X = recodecont(X).X
            else:
                col_list = [k for k in X.columns if k not in quali_sup_label]
                for k in col_list:
                    if X.loc[:,k].isnull().any():
                        X.loc[:,k] = X.loc[:,k].fillna(X.loc[:,k].mean())

        # Make a copy of the original data
        Xtot = X.copy()

        # Drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        # Drop supplementary quantitative variables
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        # Drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Principal Components Analysis (PCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")

        # Number of rows/columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list/tuple/array/Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array/Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Set variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list/tuple/array/Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array/Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=X.index,name="weight"), Series(var_weights,index=X.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Standardize
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Compute weighted average and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)
        if self.standardize:
            scale = d1.std
        else:
            scale = ones(X.shape[1])
        #convert to Series
        center, scale, wcorr = Series(d1.mean,index=X.columns,name="center"), Series(scale,index=X.columns,name="scale"), DataFrame(d1.corrcoef,index=X.columns,columns=X.columns)
        
        # Standardization : Z = (X - mu)/sigma
        Z = mapply(X,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1, n_cols))
        #set number of components
        if self.n_components is None:
            n_components = int(max_components)
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,
                            rotate=self.rotate,rotate_kwargs=self.rotate_kwargs,n_workers=n_workers,ind_sup=ind_sup_label,quanti_sup=quanti_sup_label,quali_sup=quali_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##generalized singular value decomposition (GSVD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        svd = svd_triplet(X=Z,row_weights=ind_weights,col_weights=var_weights,n_components=n_components)
        #extract elements
        U, vs, V = svd.U[:,:n_components], svd.vs[:max_components], svd.V[:,:n_components]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eigen_values = vs[:max_components]**2
        difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values)
        #store all informations
        self.eig_ = DataFrame(c_[eigen_values,difference,proportion,cumsum(proportion)],columns=["Eigenvalue","Difference","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(max_components)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables squared distance to origin
        var_sqdisto = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        #variables inertia
        var_inertia = var_sqdisto*var_weights
        #variables percentage of inertia
        var_inertia_pct = 100*var_inertia/sum(var_inertia)
        #vconvert to DataFrame
        var_infos = DataFrame(c_[var_weights,var_sqdisto,var_inertia,var_inertia_pct],columns=["Weight","Sq. Dist.","Inertia","% Inertia"],index=Z.columns)
        #variable factor coordinates
        var_coord = DataFrame(V.dot(diag(vs[:n_components])),index=Z.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
        #add rotation
        if self.rotate is not None:
            if n_components <= 1:
                warn("No rotation will be performed when the number of factors equals 1.")
            else:
                rot = rotate_factors(loadings=var_coord,method=self.rotate,**self.rotate_kwargs)
                var_coord, rotmat, phi = rot.loadings, rot.rotmat, rot.phi
                self.rotate_ = namedtuple("rotate",["rotmat","phi"])(rotmat,phi)
                #update V
                V =  apply_along_axis(func1d=lambda x : x*sqrt(var_coord.pow(2).sum(axis=0)),axis=1,arr=linalg.inv(wcorr).dot(var_coord))
        
        #convert to namedtuple
        self.svd_ = namedtuple("svd_tripletResult",["vs","U","V"])(vs,U,V)

        #variables contributions
        var_ctr = mapply(mapply(var_coord,lambda x : 100*(x**2)*var_weights,axis=0,progressbar=False,n_workers=n_workers), lambda x : x/var_coord.pow(2).sum(axis=0),axis=1,progressbar=False,n_workers=n_workers)
        #variables squared cosine
        var_cos2 = mapply(var_coord,  lambda x : (x**2)/var_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
        #convert to ordered dictionary
        var_ = OrderedDict(zip(["coord","cos2","contrib","infos"],[var_coord,var_cos2,var_ctr,var_infos]))
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variance accounted
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ss_loadings = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
        self.vaccounted_ = DataFrame(c_[ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],index = ["Dim."+str(x+1) for x in range(n_components)],
                                     columns=["SS loadings","Proportion Var","Cumulative Var","Proportion Explained","Cumulative Proportion"])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals squared distance to origin
        ind_sqdisto = mapply(Z,lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        #individuals inertia
        ind_inertia = ind_sqdisto*ind_weights
        #indivduals percentage of inertia
        ind_inertia_pct = 100*ind_inertia/sum(ind_inertia)
        #convert to DataFrame
        ind_infos = DataFrame(c_[ind_weights,ind_sqdisto,ind_inertia,ind_inertia_pct],columns=["Weight","Sq. Dist.","Inertia","% Inertia"],index=Z.index)
        #individuals factor coordinates
        ind_coord = ind_coord = mapply(Z,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(V)
        ind_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        #individuals contributions
        ind_ctr = mapply(mapply(ind_coord,lambda x : 100*(x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers),lambda x : x/var_coord.pow(2).T.dot(var_weights),axis=1,progressbar=False,n_workers=n_workers)
        #individuals squared cosine
        ind_cos2 = mapply(ind_coord,lambda x : (x**2)/ind_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
        #convert to ordered dictionary
        ind_ = OrderedDict(zip(["coord","cos2","contrib","infos"],[ind_coord,ind_cos2,ind_ctr,ind_infos]))
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #corrcoef, reproduces and partial corrcoef correlations
        rcorr, pcorr = var_coord.dot(var_coord.T), pcorrcoef(X=X)
        #residual correlation
        residual_corr = wcorr.sub(rcorr)
        #error
        error = sum(tril(residual_corr,-1)**2)
        #convert to ordered dictionary
        corr_ = OrderedDict(corrcoef=wcorr,pcorrcoef=pcorr,reconst=rcorr,residual=residual_corr,error=error)
        self.corr_ = namedtuple("correlation",corr_.keys())(*corr_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eigvals, _ = linalg.eig(rcorr)
        eigvals = Series(sorted(real(eigvals),reverse=True),index=["Dim."+str(x+1) for x in range(n_cols)],name="common")
        self.eigval_ = namedtuple("eigval",["original","common"])(eigen_values,eigvals)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##others informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fidélité des facteurs - variance of the scores - R2
        f_fidelity = var_coord.mul(V).sum(axis=0)
        f_fidelity.name = "R2"
        #Variance explained by each factor
        explained_variance = DataFrame(c_[eigen_values[:n_components],ss_loadings],index=["Dim."+str(x+1) for x in range(n_components)],columns=["Weighted","Unweighted"])
        #Initial community
        init_comm = 1 - 1/diag(linalg.inv(wcorr))
        #estimated communalities
        final_comm = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
        #communality
        communality = DataFrame(c_[init_comm,final_comm],columns=["Prior","Final"],index=Z.columns)
        #communalities
        communalities = sum(final_comm)
        #uniquenesses
        uniquenesses = 1 - final_comm
        uniquenesses.name = "Uniqueness"
        #Bartlett - statistics
        bartlett_stats = -(n_rows-1-(2*n_cols+5)/6)*sum(log(eigen_values))
        bs_dof = n_cols*(n_cols-1)/2
        bs_pvalue = 1 - chi2.cdf(bartlett_stats,df=bs_dof)
        bartlett = DataFrame([[sum(log(eigen_values)),bartlett_stats,bs_dof,bs_pvalue]],columns=["|CORR.MATRIX|","CHISQ","dof","p-value"],index=["Bartlett's test"])
        #Kaiser threshold
        kaiser = namedtuple("kaiser",["threshold","proportion"])(mean(eigen_values),100/sum(var_inertia))
        #Karlis - Saporta - Spinaki threshold
        kss_threshold =  1 + 2*sqrt((n_cols-1)/(n_rows-1)) 
        #broken-stick crticial values
        broken = Series(flip(cumsum(list(map(lambda x : 1/x,range(n_cols,0,-1)))))[:max_components],name="Broken-stick crit. val.",index=["Dim.".format(x+1) for x in range(max_components)])
        #convert to ordered dictionnary
        others_ = OrderedDict(r2_score=f_fidelity,explained_variance=explained_variance,communality=communality,communalities=communalities,uniquenesses=uniquenesses, bartlett=bartlett, kaiser=kaiser,kaiser_msa=kaiser_msa(X=X),kss=kss_threshold,broken=broken)
        #convert to namedtuple
        self.others_ = namedtuple("others",others_.keys())(*others_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #standardize the data
            Z_ind_sup = mapply(X_ind_sup,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #square distance to origin
            ind_sup_sqdisto = mapply(Z_ind_sup, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
            ind_sup_sqdisto.name = "Sq. Dist."
            #statistics for supplementary individuals
            ind_sup_ = predict_ind_sup(Z=Z_ind_sup,V=self.svd_.V,sqdisto=ind_sup_sqdisto,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            #fill missing with mean
            X_quanti_sup = recodecont(X=X_quanti_sup).X
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_quanti_sup(X=X_quanti_sup,row_coord=self.ind_.coord,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            #check if two columns have the same categories
            X_quali_sup, n_quali_sup = revaluate_cat_variable(X_quali_sup), len(quali_sup_label)
            #conditional mean - Barycenter of original data
            barycentre = conditional_average(X=X,Y=X_quali_sup,weights=self.call_.ind_weights)
            #standardize the barycenter
            Z_quali_sup = mapply(barycentre,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #factor coordinates (scores)
            quali_sup_coord = mapply(Z_quali_sup, lambda x : x*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
            #squared distance to origin
            quali_sup_sqdisto  = mapply(Z_quali_sup, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
            quali_sup_sqdisto.name = "Sq. Dist."
            #categories coefficients
            n_k = concat((X_quali_sup[q].value_counts().sort_index() for q in X_quali_sup.columns),axis=0)
            coef_k = sqrt(((n_rows-1)*n_k)/(n_rows-n_k))
            #statistics for supplementary categories
            quali_sup_ = predict_quali_sup(X=X_quali_sup,row_coord=self.ind_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
            #update value-test with squared eigenvalues
            quali_sup_["vtest"] = mapply(quali_sup_["vtest"],lambda x : x/sqrt(self.var_.coord.pow(2).T.dot(self.call_.var_weights)),axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #merge dictionary
            quali_sup_ = OrderedDict(**OrderedDict(barycentre=barycentre),**quali_sup_)
            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

            #descriptive descriptive of qualitative variables
            self.summary_quali_ = summarize(X=X_quali_sup)

            #degree of association
            if n_quali_sup>1:
                self.association_ = association(X=X_quali_sup) 

        #all quantitative variables in original dataframe
        is_quanti = Xtot.select_dtypes(include=number)
        #multivariate goodness of fit
        if self.ind_sup is not None:
            is_quanti = is_quanti.drop(index=ind_sup_label)
        #descriptive statistics of quantitatives variables 
        self.summary_quanti_ = summarize(X=is_quanti)
        self.model_ = "pca"

        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_samples, n_components)
            Transformed values.
        
        Examples
        --------
        ```python
        >>> from scientisttools import decathlon, PCA, summaryPCA
        >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
        >>> ind_coord = res_pca.fit_transform(decathlon)
        ```
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
        `X`: pandas DataFrame of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas DataFrame of shape (n_samples, n_columns)
            Original data, where `n_samples` is the number of samples and `n_columns` is the number of columns
        
        Examples
        --------
        ```python
        >>> from scientisttools import decathlon, PCA, summaryPCA
        >>> res_pca = PCA(n_components=None,ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
        >>> res_pca.fit(decathlon)
        >>> X_original = res_pca.inverse_transform(res_pca.ind_.coord)
        ```
        """
        # Check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #set number of components
        n_components = min(X.shape[1],self.call_.n_components)
        eigvals = self.var_.coord.pow(2).T.dot(self.call_.var_weights)[:n_components]
        #inverse transform
        X_original = X.iloc[:,:n_components].dot(mapply(self.var_.coord.iloc[:,:n_components],lambda x : x/sqrt(eigvals),axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
        X_original = mapply(X_original,lambda x : (x*self.call_.scale)+self.call_.center,axis=1,progressbar=False,n_workers=self.call_.n_workers)
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
        `X`: pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new` : pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        
        Examples
        --------
        ```python
        >>> from scientisttools import load_decathlon, PCA, summaryPCA
        >>> decathlon = load_decathlon("all")
        >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
        >>> res_pca.fit(decathlon)
        >>> #load supplementary individuals
        >>> ind_sup = load_decathlon("ind_sup")
        >>> ind_sup_coord = res_pca.transform(ind_sup) #coordinate of new individuals
        ```
        """
        #check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None

        #check if X.shape[1] == ncols
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #find intersect
        intersect_col = [x for x in X.columns if x in self.call_.X.columns]
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")
        #reorder columns
        X = X.loc[:,self.call_.X.columns]

        #standardisation and apply transition relation
        coord = mapply(X,lambda x : ((x - self.call_.center)/self.call_.scale)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord
    
def predictPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Principal Component Analysis (PCA)
    ------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Principal Component Analysis (PCA)

    Usage
    -----
    ```python
    >>> predictPCA(self,X:DataFrame)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `X`: pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    namedtuple of pandas DataFrame/Series containing all the results for the new individuals including:
    
    `coord`: factor coordinates

    `cos2`: squared cosinus

    `dist`: squared distance to origin
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_decathlon, PCA, predictPCA
    >>> decathlon = load_decathlon("actif")
    >>> res_pca = PCA(rotation=None)
    >>> res_pca.fit(decathlon)
    >>> #load supplementary individuals
    >>> ind_sup = load_decathlon("ind_sup")
    >>> predict = predictPCA(res_pca,X=ind_sup)
    >>> predict.coord.head() #coordinate of new individuals
    >>> predict.cos2.head() #squared cosinus of new individuals
    >>> predict.dist.head() #squared distance to origin of new individuals
    ```
    """
    #check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    #check if X is an instance of pd.DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set index name as None
    X.index.name = None

    #check if X.shape[1] == n_cols
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")
    
    #check if all variables are numerics
    all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")

    #find intersect
    intersect_col = [x for x in X.columns if x in self.call_.X.columns]
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")
    #reorder columns
    X = X.loc[:,self.call_.X.columns]

    #standardize data
    Z = mapply(X,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #square distance to origin
    sqdisto = mapply(Z, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #statistics for supplementary individuals
    ind_sup_ = predict_ind_sup(Z=Z,V=self.svd_.V,sqdisto=sqdisto,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
    return namedtuple("predictPCAResult",ind_sup_.keys())(*ind_sup_.values())

def supvarPCA(self,X_quanti_sup=None, X_quali_sup=None) -> NamedTuple:
    """
    Supplementary variables in Principal Components Analysis (PCA)
    --------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Principal Components Analysis (PCA)

    Usage
    -----
    ```python
    >>> supvarPCA(self,X_quanti_sup=None, X_quali_sup=None)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `X_quanti_sup`: pandas DataFrame of supplementary quantitative variables (default None)

    `X_quali_sup`: pandas DataFrame of supplementary qualitative variables (default None)

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary variables including : 

    `quanti`: a namedtuple of pandas DataFrame containing the results of the supplementary quantitative variables including :
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
    
    `quali`: a namedtuple of pandas DataFrame/Series containing the results of the supplementary qualitative/categories variables including :
        * `coord`: factor coordinates
        * `cos2`: squares cosinus
        * `vtest`: value-test
        * `dist`: squared distance to origin
        * `eta2`: squared correlation ratio

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_decathlon, PCA, supvarPCA
    >>> decathlon = load_decathlon("actif")
    >>> res_pca = PCA(rotation=None)
    >>> res_pca.fit(decathlon)
    >>> #supplementary quantitative and qualitative variables
    >>> X_quanti_sup, X_quali_sup = load_decathlon("quanti_sup"), load_decathlon("quali_sup")
    >>> sup_var = supvarPCA(res_pca,X_quanti_sup=X_quanti_sup,X_quali_sup=X_quali_sup)
    >>> quanti_sup = sup_var.quanti
    >>> quanti_sup.coord #factor coordinates
    >>> quanti_sup.vos2 #squared cosinus
    >>> quali_sup = sup_var.quali
    >>> quali_sup.coord #factor coordinates
    >>> quali_sup.cos2 #squared cosinus
    >>> quali_sup.vtest #value-test
    >>> quali_sup.dist #squared distance to origin
    >>> quali_sup.eta2 # squared correlation ratio
    ```
    """
    #check if self is and object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Statistics for supplementary quantitatives variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X_quanti_sup is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if X_quanti_sup.shape[0] = nrows
        if X_quanti_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")

        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_quanti_sup[k]) for k in X_quanti_sup.columns)
        if not all_num:
            raise TypeError("All columns in `X_quanti_sup` must be numeric")
        
        #fill missing with mean
        X_quanti_sup = recodecont(X_quanti_sup).X
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(X=X_quanti_sup,row_coord=self.ind_.coord,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quanti_sup =  namedtuple("quanti_stp",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Statistics for supplementary qualitative
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #check if X_quali_sup.shape[0] = nrows
        if X_quali_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")
        
        # Check if all columns are categoricals
        all_cat = all(api.types.is_string_dtype(X_quali_sup[q]) for q in X_quali_sup.columns)
        if not all_cat:
            raise TypeError("All columns in `X_quali_sup` must be categoricals")
        
        #convert to factor
        for q in X_quali_sup.columns:
            X_quali_sup[q] = Categorical(X_quali_sup[q],categories=sorted(X_quali_sup[q].dropna().unique().tolist()),ordered=True)
        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        # conditional average of original data
        barycentre = conditional_average(X=self.call_.X,Y=X_quali_sup,weights=self.call_.ind_weights)
        #standardize the data
        Z_quali_sup = mapply(barycentre,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #categories factor coordinates (scores)
        quali_sup_coord = mapply(Z_quali_sup, lambda x : x*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        #squared distance to origin
        quali_sup_sqdisto  = mapply(Z_quali_sup, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
        quali_sup_sqdisto.name = "Sq. Dist."
        #categories coefficients
        n_k =  concat((X_quali_sup[q].value_counts().sort_index() for q in X_quali_sup.columns),axis=0)
        coef_k = sqrt(((X_quali_sup.shape[0] - 1)*n_k)/(X_quali_sup.shape[0] - n_k))
        #statistics for supplementary categories
        quali_sup_ = predict_quali_sup(X=X_quali_sup,row_coord=self.ind_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #update value-test with squared eigenvalues
        quali_sup_["vtest"] = mapply(quali_sup_["vtest"],lambda x : x/sqrt(self.var_.coord.pow(2).T.dot(self.call_.var_weights)),axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #merge dictionary
        quali_sup_ = OrderedDict(**OrderedDict(barycentre=barycentre),**quali_sup_)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarPCAResult",["quanti","quali"])(quanti_sup,quali_sup)