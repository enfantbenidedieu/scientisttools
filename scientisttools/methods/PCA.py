# -*- coding: utf-8 -*-
from scipy.stats import chi2
from numpy import ndarray, array, ones, sqrt, linalg, log, flip,cumsum,mean
from pandas import DataFrame, Series, Categorical, api, crosstab, concat
from scipy.stats import chi2_contingency
from typing import NamedTuple
from collections import namedtuple,OrderedDict
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.recodecont import recodecont
from .functions.fitfa import fitfa
from .functions.predict_sup import predict_ind_sup, predict_quanti_sup, predict_quali_sup
from .functions.revaluate_cat_variable import revaluate_cat_variable
from .functions.conditional_average import conditional_average

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
    >>> PCA(standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, quanti_sup = None, quali_sup = None, parallelize=False)
    ```
    
    Parameters
    ----------
    `standardize` : a boolean, default = True
        * If `True` : the data are scaled to unit variance.
        * If `False` : the data are not scaled to unit variance.

    `n_components` : number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights` : an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup` : an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `quanti_sup` : an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `quali_sup` : an integer/string/list/tuple indicating the indexes/names of the supplementary categorical variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `call_`: namedtuple with some informations
        * `Xtot`: pandas dataframe with all data (active and supplementary)
        * `X`: pandas dataframe with active data
        * `Z`: pandas dataframe with standardized data : Z = (X-center)/scale
        * `ind_weights`: pandas series containing individuals weights
        * `var_weights`: pandas series containing variables weights
        * `center`: pandas series containing variables means
        * `scale`: pandas series containing variables standard deviation : 
            * If `standardize = True`, then standard deviation are computed using variables standard deviation
            * If `standardize = False`, then standard deviation are a vector of ones with length number of variables.
        * `n_components`: an integer indicating the number of components kept
        * `n_workers`: an integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals
        * `quanti_sup`: None or a list of string indicating names of the supplementary quantitative variables
        * `quali_sup`: None or a list of string indicating names of the supplementary qualitative variables
    
    `svd_`: namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD)

    `eig_`: pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_`: namedtuple of pandas dataframes containing all the results for the active variables (coordinates, correlation between variables and axes, square cosinus, contributions)

    `ind_`: namedtuple of pandas dataframes containing all the results for the active individuals (coordinates, square cosinus, contributions)

    `others_`: namedtuple of others statistics (Bartlett's test of Spericity, Kaiser threshold, ...)

    `ind_sup_`: namedtuple of pandas dataframes containing all the results for the supplementary individuals (coordinates, square cosinus)

    `quanti_sup_`: namedtuple of pandas dataframes containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes, square cosinus)

    `quali_sup_`: namedtuple of pandas dataframes containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, vtest which is a criterion with a Normal distribution, and eta2 which is the square correlation coefficient between a qualitative variable and a dimension)

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
    >>> #load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
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
                 parallelize=False):
        self.standardize = standardize
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
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
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if X is an instance of pd.DataFrame class
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Set index name as None
        X.index.name = None

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

        # Make a copy of the data
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

        # Statistics of quantitatives variables 
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")

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
            ind_weights = array(list(map(lambda x : x/sum(self.ind_weights), self.ind_weights)))
        
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
        center, scale = Series(d1.mean,index=X.columns,name="center"), Series(scale,index=X.columns,name="scale")
        
        # Standardization : Z = (X - mu)/sigma
        Z = mapply(X,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = min(linalg.matrix_rank(Q),linalg.matrix_rank(R))

        if self.n_components is None:
            n_components = int(max_components)
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,quanti_sup=quanti_sup_label,quali_sup=quali_sup_label)
            
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## fit factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z=Z,row_weights=ind_weights,col_weights=var_weights,max_components=max_components,n_components=n_components,n_workers=n_workers)

        # Extract generalized singular values decomposition (GSVD)
        self.svd_ = namedtuple("svd",["vs","U","V"])(fit_.svd.vs[:max_components], fit_.svd.U[:,:n_components],fit_.svd.V[:,:n_components])

        #eigenvalues
        eig_ = fit_.eig
        eig_["Broken-stick crit. val."] = flip(cumsum(list(map(lambda x : 1/x,range(n_cols,0,-1)))))[:max_components]
        self.eig_ = eig_

        # Extract and Convert to NamedTuple
        self.ind_, self.var_ = namedtuple("ind",fit_.row.keys())(*fit_.row.values()), namedtuple("var",fit_.col.keys())(*fit_.col.values())

        # Bartlett - statistics
        bartlett_stats = -(n_rows-1-(2*n_cols+5)/6)*sum(log(self.eig_.iloc[:,0]))
        bs_dof = n_cols*(n_cols-1)/2
        bs_pvalue = 1 - chi2.cdf(bartlett_stats,df=bs_dof)
        bartlett_sphericity_test = DataFrame([sum(log(self.eig_.iloc[:,0])),bartlett_stats,bs_dof,bs_pvalue],index=["|CORR.MATRIX|","statistic","dof","p-value"],columns=["value"])
        # Kaiser threshold
        kaiser = namedtuple("kaiser",["threshold","proportion"])(mean(self.eig_.iloc[:,0]),100/sum(self.var_.infos.iloc[:,2]))
        # Karlis - Saporta - Spinaki threshold
        kss_threshold =  1 + 2*sqrt((n_cols-1)/(n_rows-1)) 

        #convert to namedtuple
        self.others_ = namedtuple("others",["bartlett","kaiser","kss"])(bartlett_sphericity_test,kaiser,kss_threshold)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary individuals
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
        ## Statistics for supplementary quantitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            # Compute weighted average and standard deviation
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=self.call_.ind_weights,ddof=0)
            center_sup = d_quanti_sup.mean
            if self.standardize:
                scale_sup = d_quanti_sup.std
            else:
                scale_sup = ones(X_quanti_sup.shape[1])
            # Standardization the supplementary quantitative variables
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - center_sup)/scale_sup,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_quanti_sup(Z=Z_quanti_sup,U=self.svd_.U,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            # Summary statistics for supplementary quantitative variables
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            #insert group columns concatenate
            summary_quanti.insert(0,"group","active")
            summary_quanti_sup.insert(0,"group","sup")
            summary_quanti = concat((summary_quanti,summary_quanti_sup),axis=0,ignore_index=True)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary qualitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            
            # Check if two columns have the same categories
            X_quali_sup, n_quali_sup = revaluate_cat_variable(X_quali_sup), len(quali_sup_label)

            # Conditional mean - Barycenter of original data
            barycentre = conditional_average(X=X,Y=X_quali_sup,weights=self.call_.ind_weights)
            # Standardize the barycenter
            Z_quali_sup = mapply(barycentre,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            # Square distance to origin
            quali_sup_sqdisto  = mapply(Z_quali_sup, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
            quali_sup_sqdisto.name = "Sq. Dist."
            #categories coefficients
            n_k = concat((X_quali_sup[q].value_counts().sort_index() for q in X_quali_sup.columns),axis=0)
            coef_k = sqrt(((n_rows-1)*n_k)/(n_rows-n_k))
            #statistics for supplementary categories
            quali_sup_ = predict_quali_sup(X=X_quali_sup,Z=Z_quali_sup,Y=self.ind_.coord,V=self.svd_.V,col_coef=coef_k,sqdisto=quali_sup_sqdisto,
                                           row_weights=self.call_.ind_weights,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
            #update value-test with squared eigenvalues
            quali_sup_["vtest"] = mapply(quali_sup_["vtest"],lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #merge dictionary
            quali_sup_ = OrderedDict(**OrderedDict(barycentre=barycentre),**quali_sup_)
            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

            #summary statistiques for qualitative variables
            summary_quali_sup = DataFrame()
            for q in X_quali_sup.columns:
                eff = X_quali_sup[q].value_counts().to_frame("count").reset_index().rename(columns={q : "categorie"}).assign(proportion=lambda x :x["count"]/sum(x["count"]))
                eff.insert(0,"variable",q)
                summary_quali_sup = concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            self.summary_quali_ = summary_quali_sup

            # Chi2 statistic test
            if n_quali_sup>1:
                chi2_test = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for q1 in range(n_quali_sup-1):
                    for q2 in range(q1+1,n_quali_sup):
                        chi = chi2_contingency(crosstab(X_quali_sup.iloc[:,q1],X_quali_sup.iloc[:,q2]),correction=False)
                        row_chi2 = DataFrame(OrderedDict(variable1=X_quali_sup.columns[q1],variable2=X_quali_sup.columns[q2],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                        chi2_test = concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                        idx = idx + 1
                chi2_test["dof"] = chi2_test["dof"].astype("int")
                self.chi2_test_ = chi2_test

        self.summary_quanti_ = summary_quanti

        self.model_ = "pca"

        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: pandas dataframe of shape (n_samples, n_components)
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
        `X`: pandas dataframe of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas dataframe of shape (n_samples, n_columns)
            Original data, where ``n_samples` is the number of samples and `n_columns` is the number of columns
        
        """
        # Check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # set number of components
        n_components = min(X.shape[1],self.call_.n_components)

        #inverse transform
        X_original = X.iloc[:,:n_components].dot(mapply(self.var_.coord.iloc[:,:n_components],lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
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
        `X`: pandas dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        # Check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if X.shape[1] = ncols
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")
        
        #set index name as None
        X.index.name = None
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #factor coordinates
        coord = mapply(X,lambda x : ((x - self.call_.center)/self.call_.scale)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord
    
def predictPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Principal Component Analysis (PCA)
    ------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and square distance to origin of new individuals with Principal Component Analysis (PCA)

    Usage
    -----
    ```python
    >>> predictPCA(self,X:DataFrame)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `X`: pandas dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    namedtuple of dataframes containing all the results for the new individuals including:
    
    `coord`: factor coordinates of the new individuals

    `cos2`: square cosinus of the new individuals

    `dist`: square distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load cars2006 dataset
    >>> from scientisttools import load_cars2006
    >>> D = load_cars2006(which="actif")
    >>> from scientisttools import PCA
    >>> res_pca = PCA(n_components=5)
    >>> res_pca.fit(D)
    >>> # Load supplementary individuals
    >>> ind_sup = load_cars2006(which="ind_sup")
    >>> from scientisttools import predictPCA
    >>> predict = predictPCA(res_pca,X=ind_sup)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Check if columns are aligned
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")
    
    # Set index name as None
    X.index.name = None

    #check if all variables are numerics
    all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")

    # Standardize data
    Z = mapply(X,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #square distance to origin
    sqdisto = mapply(Z, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #statistics for supplementary individuals
    ind_sup_ = predict_ind_sup(Z=Z,V=self.svd_.V,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
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

    `X_quanti_sup`: pandas dataframe of supplementary quantitatives variables (default = None)

    `X_quali_sup`: pandas dataframe of supplementary qualitatives variables (default = None)

    Returns
    -------
    namedtuple of namedtuple containing the results for supplementary variables including : 

    `quanti`: namedtuple containing the results of the supplementary quantitatives variables including :
        * `coord`: factor coordinates of the supplementary quantitatives variables
        * `cos2`: square cosinus of the supplementary quantitatives variables
    
    `quali`: namedtuple containing the results of the supplementary qualitatives/categories variables including :
        * `coord`: factor coordinates of the supplementary categories
        * `cos2`: square cosinus of the supplementary categories
        * `vtest`: value-test of the supplementary categories
        * `dist`: square distance to origin of the supplementary categories
        * `eta2`: square correlation ratio of the supplementary qualitatives variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load cars2006 dataset
    >>> from scientisttools import load_cars2006
    >>> D = load_cars2006(which="actif")
    >>> from scientisttools import PCA
    >>> res_pca = PCA(n_components=5)
    >>> res_pca.fit(D)
    >>> # Supplementary quantitatives variables
    >>> X_quanti_sup = load_cars2006(which="quanti_sup")
    >>> # Supplementary qualitatives variables
    >>> X_quali_sup = load_cars2006(which="quali_sup")
    >>> from scientisttools import supvarPCA
    >>> sup_var_predict = supvarPCA(res_pca,X_quanti_sup=X_quanti_sup,X_quali_sup=X_quali_sup)
    ```
    """
    # Check if self is and object of class PCA
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
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_quanti_sup[k]) for k in X_quanti_sup.columns)
        if not all_num:
            raise TypeError("All columns in `X_quanti_sup` must be numeric")
        
        #fill missing with mean
        X_quanti_sup = recodecont(X_quanti_sup).X
        # Compute weighted average and standard deviation
        d_quanti_sup = DescrStatsW(X_quanti_sup,weights=self.call_.ind_weights,ddof=0)
        # Average
        center_sup = d_quanti_sup.mean
        if self.standardize:
            scale_sup = d_quanti_sup.std
        else:
            scale_sup = ones(X_quanti_sup.shape[1])
        # Standardization data
        Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - center_sup)/scale_sup,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(Z=Z_quanti_sup,U=self.svd_.U,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
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
        # Supplementary qualitatives squared distance
        quali_sup_sqdisto  = mapply(Z_quali_sup, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
        quali_sup_sqdisto.name = "Sq. Dist."
        #categories coefficients
        n_rows, n_k = X_quali_sup.shape[0], concat((X_quali_sup[q].value_counts().sort_index() for q in X_quali_sup.columns),axis=0)
        coef_k = sqrt(((n_rows-1)*n_k)/(n_rows-n_k))
        #statistics for supplementary categories
        quali_sup_ = predict_quali_sup(X=X_quali_sup,Z=Z_quali_sup,Y=self.ind_.coord,V=self.svd_.V,col_coef=coef_k,sqdisto=quali_sup_sqdisto,
                                       row_weights=self.call_.ind_weights,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
        #update value-test with squared eigenvalues
        quali_sup_["vtest"] = mapply(quali_sup_["vtest"],lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #merge dictionary
        quali_sup_ = OrderedDict(**OrderedDict(barycentre=barycentre),**quali_sup_)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarPCAResult",["quanti","quali"])(quanti_sup,quali_sup)