# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
from collections import namedtuple
from mapply.mapply import mapply
import pingouin as pg
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

class EFA(BaseEstimator,TransformerMixin):
    """
    Exploratory Factor Analysis (EFA)
    ---------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    ------------
    Performs a Exploratory Factor Analysis (EFA)

    Usage
    -----
    ```python
    >>> EFA(standardize = True,n_components = None, ind_sup = None, ind_weights = None, var_weights = None, parallelize = False)
    ```

    Parameters
    ----------
    `standardize` : a boolean, default = True
        * If `True` : the data are scaled to unit variance.
        * If `False` : the data are not scaled to unit variance.

    `n_components` : number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple/array of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.

    `var_weights` : an optional variables weights (by default, a list/tuple/array of 1 for uniform variables weights), the weights are given only for the active variables

    `ind_sup` : an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply
    
    Attrbutes
    ---------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_` : namedtuple of pandas dataframes containing all the results for the active variables (factor coordinates, contributions, normalized score coefficients and fidelity)

    `ind_` : namedtuple of pandas dataframes containing all the results for the active individuals (factor coordinates)

    `ind_sup_` : namedtuple of pandas dataframes containing all the results for the supplementary individuals (factor coordinates)

    `summary_quanti_` : summary statistics for quantitative variables

    `call_` : namedtuple with some statistics

    `others_` : namedtuple of pandas dataframes containing :
        * "communality" for communatilities
        * "explained_variance" for explained variance
        * "inertia" for inertia

    `model_` : string specifying the model fitted = 'efa'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Berger J-L (2021), Analyse factorielle exploratoire et analyse en composantes principales : guide pratique, hal-03436771v1

    Lawley, D.N., Maxwell, A.E. (1963), Factor Analysis as a Statistical Method, Butterworths Mathematical Texts, England

    Marley W. Watkins (2018), Exploratory Factor Analysis : A guide to best practice, Journal of Black Psychology, Vol. 44(3) 219-246

    Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0

    Links
    -----
    https://en.wikipedia.org/wiki/Exploratory_factor_analysis

    https://datatab.fr/tutorial/exploratory-factor-analysis

    https://jmeunierp8.github.io/ManuelJamovi/s15.html

    https://stats.oarc.ucla.edu/sas/output/factor-analysis/

    See Also
    --------
    `get_efa_ind`, `get_efa_var`, `get_efa`, `summaryEFA`

    Examples
    --------
    ```python
    >>> 
    ```
    """
    def __init__(self,
                 standardize =True,
                 n_components = None,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 parallelize = False):
        self.standardize = standardize
        self.n_components =n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y` : None
            y is ignored

        Returns
        -------
        `self` : object
            Returns the instance itself
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # cgeck if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set index name as None
        X.index.name = None
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

         #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if individuls supplementary
        #----------------------------------------------------------------------------------------------------------------------------------------
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
        
        # Check if missing values in quantitatives variables
        if X.isnull().any().any():
            for col in X.columns:
                if X.loc[:,col].isnull().any():
                    X.loc[:,col] = X.loc[:,col].fillna(X.loc[:,col].mean())
                    
        # Save dataframe
        Xtot = X.copy()

        #----------------------------------------------------------------------------------------------------------------
        ## Exploratory Factor Analysis (EFA)
        #----------------------------------------------------------------------------------------------------------------
        # Drop supplementary individuls
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        # Number of rows/columns
        n_rows, n_cols = X.shape

         # Summary quantitatives variables
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti
        
        #---------------------------------------------------------------------------------------------------
        ## Set individuals weights
        #---------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = np.ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,np.ndarray)):
            raise TypeError("'ind_weights' must be a list/tuple/array of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array with length {n_rows}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        #---------------------------------------------------------------------------------------------------
        ## Set variables weights
        #---------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = np.ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,np.ndarray)):
            raise TypeError("'var_weights' must be a list/tuple/array of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array with length {n_cols}.")
        else:
            var_weights = np.array(self.var_weights)

        # Compute average mean and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)

        # weighted Pearson correlation
        wcorr = pd.DataFrame(d1.corrcoef,index=X.columns,columns=X.columns)
        
        # Rsquared
        initial_communality = pd.Series([1 - (1/x) for x in np.diag(np.linalg.inv(wcorr))],index=X.columns,name="initial")
        
        #--------------------------------------------------------------------------------------------------------
        ## Standardisation
        #--------------------------------------------------------------------------------------------------------
        # Initializations - scale data
        center = d1.mean
        if self.standardize:
            scale = d1.std
        else:
            scale = np.ones(X.shape[1])
        
        # Standardization : Z = (X - mu)/sigma
        Z = mapply(X,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)

        ###################################### Replace Diagonal of correlation matrix with commu
        # Store initial weighted correlation matrix
        wcorr_c = wcorr.copy()
        for col in X.columns:
            wcorr_c.loc[col,col] = initial_communality[col]
        
        # Eigen decomposition
        eigenvalue, eigenvector = np.linalg.eigh(wcorr_c)

        # Sort eigenvalue
        eigen_values = np.flip(eigenvalue)
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # Set n_components_
        if self.n_components is None:
            n_components = (eigenvalue > 0).sum()
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,(eigenvalue > 0).sum())
        
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        call_ = {"Xtot":Xtot,
                 "X" : X,
                 "Z" : Z,
                 "var_weights" : pd.Series(var_weights,index=X.columns,name="weight"),
                 "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                 "center" : pd.Series(center,index=X.columns,name="center"),
                 "scale" : pd.Series(scale,index=X.columns,name="scale"),
                 "n_components" : n_components,
                 "standardize" : self.standardize,
                 "ind_sup" : ind_sup_label}
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #----------------------------------------------------------------------------------------------------------------------------
        ## Variables informations : factor coordinates, contributions
        #----------------------------------------------------------------------------------------------------------------------------
        # Variables factor coordinates
        var_coord = np.apply_along_axis(func1d=lambda x : x*np.sqrt(np.flip(eigenvalue)[:n_components]),axis=1,arr=np.fliplr(eigenvector)[:,:n_components])
        var_coord = pd.DataFrame(var_coord,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        # F - scores
        factor_score = np.dot(np.linalg.inv(wcorr),var_coord)
        factor_score = pd.DataFrame(factor_score,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        # Fidélité des facteurs
        factor_fidelity = np.sum(factor_score*var_coord,axis=0)
        factor_fidelity = pd.Series(factor_fidelity,index=["Dim."+str(x+1) for x in range(len(factor_fidelity))],name="fidelity")

        # Contribution des variances
        var_contrib = 100*np.square(factor_score)/np.sum(np.square(factor_score),axis=0)
        var_contrib = pd.DataFrame(var_contrib,columns = ["Dim."+str(x+1) for x in range(var_contrib.shape[1])],index=X.columns)
        
        # Store all informations
        var_ = {"coord" : var_coord, "contrib" : var_contrib, "score_coef_n" : factor_score,"fidelity" : factor_fidelity}
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #----------------------------------------------------------------------------------------------------------------------------
        ## Individuals informations : factor coordinates
        #----------------------------------------------------------------------------------------------------------------------------
        ind_coord = Z.dot(factor_score)
        ind_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        self.ind_ = namedtuple("ind",["coord"])(ind_coord)
        
        #----------------------------------------------------------------------------------------------------------------------------
        ## Others statistics
        #----------------------------------------------------------------------------------------------------------------------------
        # Variance restituées
        explained_variance = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        explained_variance.name = "variance"

        # Communalité estimée
        estimated_communality = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
        estimated_communality.name = "estimated"

        #community
        communality = pd.concat((initial_communality,estimated_communality),axis=1).assign(percentage_variance=lambda x : x.estimated/x.initial)

        # Total inertia
        inertia = np.sum(initial_communality)
        others_ = {"communality" : communality,"explained_variance" : explained_variance, "inertia" : inertia}
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #----------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary individuals                                      
        #----------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            # Transform to float
            X_ind_sup = X_ind_sup.astype("float")

            # Standardization
            Z_ind_sup = mapply(X_ind_sup,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)

            # Individuals coordinates
            ind_sup_coord = Z_ind_sup.dot(factor_score)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Store all informations
            self.ind_sup_ = namedtuple("ind_sup",["coord"])(ind_sup_coord)


        self.model_ = "efa"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y` : None
            y is ignored.
        
        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord

    def transform(self,X):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Set index name as None
        X.index.name = None
        
        # Standardize the data
        Z = mapply(X,lambda x : (x - self.call_.center.values)/self.call_.scale.values,axis=1,progressbar=False,n_workers=n_workers)
        
        # Apply transition relation
        coord = Z.dot(self.var_.score_coef_n)
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord