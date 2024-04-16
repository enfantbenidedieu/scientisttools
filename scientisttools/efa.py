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

    Description
    ------------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    This class performs a Exploratory Factor Analysis, given a table of
    numeric variables; shape = n_rows x n_columns

    Parameters
    ----------
    standardize : a boolean, default = True
        - If True : the data are scaled to unit variance.
        - If False : the data are not scaled to unit variance.

    n_components : number of dimensions kept in the results (by default 5)

    ind_sup : a list/tuple indicating the indexes of the supplementary individuals

    ind_weights : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights),
                    the weights are given only for active individuals.
    
    var_weights : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for
                    the active variables

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply
    
    Returns:
    --------

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_efa_ind, get_efa_var, get_eaf, summaryEFA
    """
    def __init__(self,
                standardize =True,
                n_components = None,
                ind_sup = None,
                ind_weights = None,
                var_weights = None,
                parallelize = False):
        self.standardize = standardize
        self.n_components =n_components
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored

        Returns:
        --------
        self : object
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
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]

        # Save dataframe
        Xtot = X.copy()

        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ################################################################################################
        # Set individuals weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        # Set variables weight
        if self.var_weights is None:
            var_weights = np.ones(X.shape[1])
        elif not isinstance(self.var_weights,list):
            raise ValueError("Error : 'var_weights' must be a list of variables weights.")
        elif len(self.var_weights) != X.shape[1]:
            raise ValueError(f"Error : 'var_weights' must be a list with length {X.shape[1]}.")
        else:
            var_weights = np.array(self.var_weights)
        
        ################## Summary quantitatives variables ####################
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti
        
        ####### weighted Pearson correlation
        weighted_corr = pd.DataFrame(weightedcorrcoef(x=X,w=ind_weights),index=X.columns.tolist(),columns=X.columns.tolist())
        
        # Rsquared
        initial_communality = pd.Series([1 - (1/x) for x in np.diag(np.linalg.inv(weighted_corr))],index=X.columns.tolist(),name="initial")
        
        #################### Standardize
        ############# Compute average mean and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)

        # Initializations - scale data
        means = d1.mean.reshape(1,-1)
        if self.standardize:
            std = d1.std.reshape(1,-1)
        else:
            std = np.ones(X.shape[1]).reshape(1,-1)
        # Z = (X - mu)/sigma
        Z = (X - means)/std

        ###################################### Replace Diagonal of correlation matrix with commu
        # Store initial weighted correlation matrix
        weighted_corr_copy = weighted_corr.copy()
        for col in X.columns.tolist():
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
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("Error : 'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,(eigenvalue > 0).sum())
        
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot":Xtot,
                      "X" : X,
                      "Z" : Z,
                      "var_weights" : pd.Series(var_weights,index=X.columns.tolist(),name="weight"),
                      "row_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "means" : pd.Series(means[0],index=X.columns.tolist(),name="average"),
                      "std" : pd.Series(std[0],index=X.columns.tolist(),name="scale"),
                      "n_components" : n_components,
                      "standardize" : self.standardize}

        ##########################################################################################################################
        # Compute columns coordinates
        var_coord = np.apply_along_axis(func1d=lambda x : x*np.sqrt(np.flip(eigenvalue)[:n_components]),axis=1,arr=np.fliplr(eigenvector)[:,:n_components])
        var_coord = pd.DataFrame(var_coord,columns = ["Dim."+str(x+1) for x in range(var_coord.shape[1])],index=X.columns.tolist())

        # F - scores
        factor_score = np.dot(np.linalg.inv(weighted_corr),var_coord)
        factor_score = pd.DataFrame(factor_score,columns = ["Dim."+str(x+1) for x in range(factor_score.shape[1])],index=X.columns.tolist())

        # Fidélité des facteurs
        factor_fidelity = np.sum(factor_score*var_coord,axis=0)
        factor_fidelity = pd.Series(factor_fidelity,index=["Dim."+str(x+1) for x in range(len(factor_fidelity))],name="fidelity")

        # Contribution des variances
        var_contrib = 100*np.square(factor_score)/np.sum(np.square(factor_score),axis=0)
        var_contrib = pd.DataFrame(var_contrib,columns = ["Dim."+str(x+1) for x in range(var_contrib.shape[1])],index=X.columns.tolist())
        self.var_ = {"coord" : var_coord, "contrib" : var_contrib, "normalized_score_coef" : factor_score,"fidelity" : factor_fidelity}

        #################################################################################################################################
        # Individuals coordinates
        ind_coord = np.dot(Z,factor_score)
        ind_coord = pd.DataFrame(ind_coord,columns = ["Dim."+str(x+1) for x in range(ind_coord.shape[1])],index=X.index.tolist())
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
            Z_ind_sup = (X_ind_sup - means)/std

            # Individuals coordinates
            ind_sup_coord = np.dot(Z_ind_sup,factor_score)
            ind_sup_coord = pd.DataFrame(ind_coord,columns = ["Dim."+str(x+1) for x in range(ind_coord.shape[1])],index=X.index.tolist())

            self.ind_sup_ = {"coord" : ind_sup_coord}

        self.model_ = "efa"

        return self

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are
            projected on the axes
            X is a table containing numeric values

        y : None
            y is ignored

        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
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

        Z = (X - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)
        #### Apply
        coord = Z.dot(self.var_["normalized_score_coef"])
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]