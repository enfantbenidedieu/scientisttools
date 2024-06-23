# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
import pingouin as pg
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .weightedcorrcoef import weightedcorrcoef

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
        * If True : the data are scaled to unit variance.
        * If False : the data are not scaled to unit variance.

    `n_components` : number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.

    `var_weights` : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for the active variables

    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply
    
    Attrbutes
    ---------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_` : dictionary of pandas dataframes containing all the results for the active variables (factor coordinates, contributions, normalized score coefficients and fidelity)

    `ind_` : dictionary of pandas dataframes containing all the results for the active individuals (factor coordinates)

    `ind_sup_` : dictionary of pandas dataframes containing all the results for the supplementary individuals (factor coordinates)

    `summary_quanti_` : summary statistics for quantitative variables

    `call_` : dictionary with some statistics

    `others_` : dictionary of pandas dataframes containing :
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
    get_efa_ind, get_efa_var, get_efa, summaryEFA

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

        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None
        
        # Check if missing values in quantitatives variables
        if X.isnull().any().any():
            for col in X.columns:
                if X.loc[:,col].isnull().any():
                    X.loc[:,col] = X.loc[:,col].fillna(X.loc[:,col].mean())
            print("Missing values are imputed by the mean of the variable.")

        # Save dataframe
        Xtot = X.copy()

        # Drop supplementary individuls
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)
        
        # Set individuals weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        # Set variables weight
        if self.var_weights is None:
            var_weights = np.ones(X.shape[1])
        elif not isinstance(self.var_weights,list):
            raise ValueError("'var_weights' must be a list of variables weights.")
        elif len(self.var_weights) != X.shape[1]:
            raise ValueError(f"'var_weights' must be a list with length {X.shape[1]}.")
        else:
            var_weights = np.array(self.var_weights)
        
        # Summary quantitatives variables
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti
        
        # weighted Pearson correlation
        weighted_corr = pd.DataFrame(weightedcorrcoef(x=X,w=ind_weights),index=X.columns,columns=X.columns)
        
        # Rsquared
        initial_communality = pd.Series([1 - (1/x) for x in np.diag(np.linalg.inv(weighted_corr))],index=X.columns,name="initial")
        
        # Compute average mean and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)

        # Initializations - scale data
        means = d1.mean
        if self.standardize:
            std = d1.std
        else:
            std = np.ones(X.shape[1])
        # Standardize : Z = (X - mu)/sigma
        Z = (X - means.reshape(1,-1))/std.reshape(1,-1)

        ###################################### Replace Diagonal of correlation matrix with commu
        # Store initial weighted correlation matrix
        weighted_corr_copy = weighted_corr.copy()
        for col in X.columns:
            weighted_corr_copy.loc[col,col] = initial_communality[col]
        
        # Eigen decomposition
        eigenvalue, eigenvector = np.linalg.eigh(weighted_corr_copy)

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
        self.call_ = {"Xtot":Xtot,
                      "X" : X,
                      "Z" : Z,
                      "var_weights" : pd.Series(var_weights,index=X.columns,name="weight"),
                      "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                      "means" : pd.Series(means,index=X.columns,name="average"),
                      "std" : pd.Series(std,index=X.columns,name="scale"),
                      "n_components" : n_components,
                      "standardize" : self.standardize,
                      "ind_sup" : ind_sup_label}

        ##########################################################################################################################
        # Compute columns coordinates
        var_coord = np.apply_along_axis(func1d=lambda x : x*np.sqrt(np.flip(eigenvalue)[:n_components]),axis=1,arr=np.fliplr(eigenvector)[:,:n_components])
        var_coord = pd.DataFrame(var_coord,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        # F - scores
        factor_score = np.dot(np.linalg.inv(weighted_corr),var_coord)
        factor_score = pd.DataFrame(factor_score,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        # Fidélité des facteurs
        factor_fidelity = np.sum(factor_score*var_coord,axis=0)
        factor_fidelity = pd.Series(factor_fidelity,index=["Dim."+str(x+1) for x in range(len(factor_fidelity))],name="fidelity")

        # Contribution des variances
        var_contrib = 100*np.square(factor_score)/np.sum(np.square(factor_score),axis=0)
        var_contrib = pd.DataFrame(var_contrib,columns = ["Dim."+str(x+1) for x in range(var_contrib.shape[1])],index=X.columns)
        
        # Store all informations
        self.var_ = {"coord" : var_coord, "contrib" : var_contrib, "normalized_score_coef" : factor_score,"fidelity" : factor_fidelity}

        #################################################################################################################################
        # Individuals coordinates
        ind_coord = Z.dot(factor_score)
        ind_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        self.ind_ = {"coord" : ind_coord}

        ################################################# Others
        # Variance restituées
        explained_variance = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        explained_variance.name = "variance"

        # Communalité estimée
        estimated_communality = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
        estimated_communality.name = "estimated"

        communality = pd.concat((initial_communality,estimated_communality),axis=1)

        # Pourcentage expliquée par variables
        communality = communality.assign(percentage_variance=lambda x : x.estimated/x.initial)

        # Total inertia
        inertia = np.sum(initial_communality)
        self.others_ = {"communality" : communality,"explained_variance" : explained_variance, "inertia" : inertia}

        ##############################################################################################################################################
        #                                        Compute supplementrary individuals statistics
        ###################################################################################################################################################
        if self.ind_sup is not None:
            ###################### Transform to float ##############################
            X_ind_sup = X_ind_sup.astype("float")

            ########### Standarddize data
            Z_ind_sup = (X_ind_sup - means.reshape(1,-1))/std.reshape(1,-1)

            # Individuals coordinates
            ind_sup_coord = Z_ind_sup.dot(factor_score)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Store all informations
            self.ind_sup_ = {"coord" : ind_sup_coord}

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
        return self.ind_["coord"]

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
        
        # Set index name as None
        X.index.name = None
        
        # Standardize the data
        Z = (X - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)
        
        # Apply transition relation
        coord = Z.dot(self.var_["normalized_score_coef"])
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord