# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import pingouin as pg
import statsmodels.formula.api as smf

from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

from .pca import PCA
from .weightedcorrcoef import weightedcorrcoef
from .kmo import global_kmo_index, per_item_kmo_index

class PartialPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Component Analysis (PartialPCA)
    --------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Partial Principal Component Analysis with supplementary individuals

    Parameters:
    -----------
    standardize : a boolean, if True (value set by default) then data are scaled to unit variance

    n_components : number of dimensions kept in the results (by default None)

    partiel : name of the partial variables

    ind_weights : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); 
                    the weights are given only for the active individuals
    
    var_weights : an optional variables weights (by default, uniform column weights); 
                    the weights are given only for the active variables
    
    ind_sup : a vector indicating the indexes of the supplementary individuals

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    var_  : a dictionary of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)
    
    ind_ : a dictionary of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    call_ : a dictionary with some statistics

    others_ : a dictionary of others statistics

    model_ : string. The model fitted = 'partialpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    A. Boudou (1982), Analyse en composantes principales partielle, Statistique et analyse des données, tome 7, n°2 (1982), p. 1-21

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_partialpca_ind, get_partialpca_var, get_partialpca, summaryPartialPCA
    """
    def __init__(self,
                 standardize=True,
                 n_components=None,
                 partial=None,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.standardize = standardize
        self.partial = partial
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
        X : pandas/polars DataFrame of float, shape (n_rows, n_columns)

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
        
        # Check if X is a pandas Dataframe
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ####################################
        if self.partial is None:
            raise ValueError("'partial' must be assigned.")
        
        # check if partial is set
        if isinstance(self.partial,str):
            partial = [self.partial]
        elif ((isinstance(self.partial,list) or isinstance(self.partial,tuple)) and len(self.partial)>=1):
            partial = [str(x) for x in self.partial]
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        #################### set ind weights
        # Set row weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise TypeError("'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != X.shape[0]:
            raise TypeError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        ################## Summary quantitatives variables ####################
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti
        
        ####### weighted Pearson correlation
        weighted_corr = weightedcorrcoef(x=X,w=ind_weights)
        weighted_corr = pd.DataFrame(weighted_corr,index=X.columns.tolist(),columns=X.columns.tolist())
        global_kmo = global_kmo_index(X)
        per_var_kmo = per_item_kmo_index(X)
        pcorr = X.pcorr()

        self.others_ = {"weighted_corr" : weighted_corr,
                        "global_kmo" : global_kmo,
                        "kmo_per_var" : per_var_kmo,
                        "partial_corr" : pcorr}
        
        ###### Store initial data
        Xtot = X.copy()

        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])

        #### Drop
        X = X.drop(columns = partial)

        # Set variables weight
        if self.var_weights is None:
            var_weights = np.ones(X.shape[1])
        elif not isinstance(self.var_weights,list):
            raise TypeError("'var_weights' must be a list of variables weights.")
        elif len(self.var_weights) != X.shape[1]:
            raise TypeError(f"'var_weights' must be a list with length {X.shape[1]}.")
        else:
            var_weights = np.array(self.var_weights)

        # Extract coefficients and intercept
        coef = pd.DataFrame(np.zeros((len(partial)+1,X.shape[1])),index = [*["intercept"],*partial],columns=X.columns.tolist())
        metrics = pd.DataFrame(index=X.columns.tolist(),columns=["rsquared","rmse"]).astype("float")
        resid = pd.DataFrame(np.zeros((X.shape[0],X.shape[1])),index=X.index.tolist(),columns=X.columns.tolist()) # Résidu de régression

        model = {}
        for lab in X.columns.tolist():
            res = smf.ols(formula="{}~{}".format(lab,"+".join(partial)), data=Xtot).fit()
            coef.loc[:,lab] = res.params.values
            metrics.loc[lab,:] = [res.rsquared,mean_squared_error(Xtot[lab],res.fittedvalues,squared=False)]
            resid.loc[:,lab] = res.resid
            model[lab] = res
        
        #### Store separate model
        self.separate_model_ = model
        
        ############# Compute average mean and standard deviation
        d1 = DescrStatsW(Xtot,weights=ind_weights,ddof=0)

        # Initializations - scale data
        means = d1.mean.reshape(1,-1)
        if self.standardize:
            std = d1.std.reshape(1,-1)
        else:
            std = np.ones(Xtot.shape[1]).reshape(1,-1)
        # Z = (X - mu)/sigma
        Z = (Xtot - means)/std

        ###################################### Set number of components ##########################################
        if self.n_components is None:
            n_components = min(resid.shape[0]-1,resid.shape[1])
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1")
        else:
            n_components = min(self.n_components,resid.shape[0]-1,resid.shape[1])
        
        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot":Xtot,
                      "X" : X,
                      "Z" : Z,
                      "resid" : resid,
                      "var_weights" : pd.Series(var_weights,index=X.columns.tolist(),name="weight"),
                      "row_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "means" : pd.Series(means[0],index=Xtot.columns.tolist(),name="average"),
                      "std" : pd.Series(std[0],index=Xtot.columns.tolist(),name="scale"),
                      "n_components" : n_components,
                      "standardize" : self.standardize,
                      "partial" : partial}

        # Coefficients normalisés
        normalized_coef = pd.DataFrame(np.zeros((len(self.partial),X.shape[1])),index = self.partial,columns=X.columns.tolist())
        for lab in X.columns.tolist():
            normalized_coef.loc[:,lab] = smf.ols(formula="{}~{}".format(lab,"+".join(self.partial)),data=Z).fit().params[1:]

        ############################ Global PCA
        global_pca = PCA(standardize=self.standardize,n_components=n_components).fit(resid)

        if self.ind_sup is not None:
            #####" Transform to float
            X_ind_sup = X_ind_sup.astype("float")
            ######## Apply regression to compute Residuals
            new_X = pd.DataFrame().astype("float")
            for lab in X.columns.tolist():
                # Model residuals
                new_X[lab] = X_ind_sup[lab] - model[lab].predict(X_ind_sup[partial])
            
            ##### Concatenate the two datasets
            X_ind_resid = pd.concat((resid,new_X),axis=0)

            ##### Apply PCA
            global_pca = PCA(standardize=self.standardize,n_components=n_components).fit(X_ind_resid)
            self.ind_sup_ = global_pca.ind_sup_

        #####
        self.global_pca_ = global_pca
        self.svd_ = global_pca.svd_
        self.eig_ = global_pca.eig_
        self.ind_ = global_pca.ind_
        self.var_ = global_pca.var_
        self.others_["coef"] = coef
        self.others_["normalized_coef"] = normalized_coef
        self.others_["metrics"] = metrics

        self.model_ = "partialpca"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        ----------------------------------------------------------------

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]

    def transform(self,X,y=None):
        """
        Apply the Partial Principal Components Analysis reduction on X
        --------------------------------------------------------------

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

        # Check if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        #####" Transform to float
        X = X.astype("float")
        ######## Apply regression to compute Residuals
        new_X = pd.DataFrame().astype("float")
        for lab in self.separate_model_.keys():
            # Model residuals
            new_X[lab] = X[lab] - self.separate_model_[lab].predict(X[self.call_["partial"]])
        
        #### Standardize residuals
        Z = (new_X - self.global_pca_.call_["means"].values.reshape(1,-1))/self.global_pca_.call_["std"].values.reshape(1,-1)
        # Apply PCA projection on residuals
        coord = mapply(Z,lambda x : x*self.call_["var_weights"],axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"])
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord