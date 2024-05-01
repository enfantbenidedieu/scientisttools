# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

from .svd_triplet import svd_triplet
from .weightedcorrcoef import weightedcorrcoef
from .function_eta2 import function_eta2

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    ----------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    CA performs a Correspondence Analysis, given a contingency table
    containing absolute frequencies ; shape= n_rows x n_columns.
    This implementation only works for dense dataframe.

    It Performs Correspondence Analysis (CA) including supplementary row and/or column points.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    row_weights : an optional row weights (by default, a list/tuple of 1 and each row has a weight equals to its margin); the weights are given only for the active rows

    row_sup : a list/tuple indicating the indexes of the supplementary rows

    col_sup : a list/tuple indicating the indexes of the supplementary columns

    quanti_sup : a list/tuple indicating the indexes of the supplementary continuous variables

    quali_sup : a list/tuple indicating the indexes of the categorical supplementary variables

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    col_ : 	a dictionary of pandas dataframe with all the results for the column variable (coordinates, square cosine, contributions, inertia)

    row_ : a dictionary of pandas dataframe with all the results for the row variable (coordinates, square cosine, contributions, inertia)

    col_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary column points (coordinates, square cosine)

    row_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary row points (coordinates, square cosine)

    quanti_sup_ : if quanti_sup is not None, a dictionary of pandas dataframe containing the results for the supplementary continuous variables (coordinates, square cosine)

    quali_sup_ : if quali.sup is not None, a dictionary of pandas dataframe with all the results for the supplementary categorical variables (coordinates of each categories 
                    of each variables, v.test which is a criterion with a Normal distribution, square correlation ratio)
    
    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'ca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2008), Analyses Factorielles Simples et Multiples.4ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_ca_row, get_ca_col, get_ca, summaryCA, dimdesc

    Examples
    --------
    > X = children # from FactoMineR R package

    > res_ca = CA(row_sup=list(range(14,18)),col_sup=list(range(5,8)),parallelize=True)

    > res_ca.fit(X)

    > summaryCA(res_ca)
    """

    def __init__(self,
                 n_components=None,
                 row_weights = None,
                 row_sup=None,
                 col_sup=None,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.row_weights = row_weights
        self.row_sup = row_sup
        self.col_sup = col_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : a pandas/polars DataFrame of shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.

        y : None
            y is ignored.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            self.n_workers = -1
        else:
            self.n_workers = 1
    
        #################################################################################################
        #   Drop level if ndim greater than 1 and reset columns names
        ################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ###############################################################################################
        #   Checks if categoricals variables in X and transform to factor (category)
        ###############################################################################################
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns.tolist():
                X[col] = X[col].astype("object")
        
        ##############################################################################################
        # Extract supplementary rows
        ##############################################################################################
        if self.row_sup is not None:
            if (isinstance(self.row_sup,int) or isinstance(self.row_sup,float)):
                row_sup = [int(self.row_sup)]
            elif (isinstance(self.row_sup,list) or isinstance(self.row_sup,tuple)) and len(self.row_sup) >=1:
                row_sup = [int(x) for x in self.row_sup]
        
        ##############################################################################################
        # Extract supplementary columns
        ##############################################################################################
        if self.col_sup is not None:
            if (isinstance(self.col_sup,int) or isinstance(self.col_sup,float)):
                col_sup = [int(self.col_sup)]
            elif (isinstance(self.col_sup,list) or isinstance(self.col_sup,tuple)) and len(self.col_sup) >=1:
                col_sup = [int(x) for x in self.col_sup]
        
        ##############################################################################################
        # Extract supplementary qualitatives and put in list
        ##############################################################################################
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif (isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple)) and len(self.quali_sup) >=1:
                quali_sup = [int(x) for x in self.quali_sup]

        ##############################################################################################
        # Extract supplementary quantitatives and put in list
        ##############################################################################################
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif (isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple)) and len(self.quanti_sup) >=1:
                quanti_sup = [int(x) for x in self.quanti_sup]
        
        #####################################################################################################
        # Store data - Save the base in a variables
        #####################################################################################################
        Xtot = X
        
        ################################# Drop supplementary columns #############################################
        if self.col_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in col_sup])
        
        ################################# Drop supplementary quantitatives variables ###############################
        if self.quanti_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quanti_sup])
        
         ################################# Drop supplementary qualitatives variables ###############################
        if self.quali_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quali_sup])
        
        ################################## Drop supplementary rows ##################################################
        if self.row_sup is not None:
            # Extract supplementary rows
            X_row_sup = X.iloc[row_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in row_sup])

        ################################## Start Compute Correspondence Analysis (CA) ###############################
        # Active data
        X = X.astype("int")

        ##### Set row weights
        if self.row_weights is None:
            row_weights = np.ones(X.shape[0])
        elif not isinstance(self.row_weights,list):
            raise ValueError("Error : 'row_weights' must be a list of row weight.")
        elif len(self.row_weights) != X.shape[0]:
            raise ValueError(f"Error : 'row_weights' must be a list with length {X.shape[0]}.")
        
        # Set number of components
        if self.n_components is None:
            n_components = min(X.shape[0]-1,X.shape[1]-1)
        elif isinstance(self.n_components,float):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("Error : 'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,X.shape[0]-1,X.shape[1]-1)

        ####################################################################################################################
        ####### total
        total = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=self.n_workers).sum(axis=0).sum()

        ##### Table des frequences
        freq = mapply(X,lambda x : x*(row_weights/total),axis=0,progressbar=False,n_workers=self.n_workers)
        
        ####### Calcul des marges lignes et colones
        col_marge = freq.sum(axis=0)
        col_marge.name = "col_marge"
        row_marge = freq.sum(axis=1)
        row_marge.name = "row_marge"

        ###### Compute Matrix used in SVD
        Z = mapply(freq,lambda x : x/row_marge,axis=0,progressbar=False,n_workers=self.n_workers)
        Z = mapply(Z,lambda x : (x/col_marge)-1,axis=1,progressbar=False,n_workers=self.n_workers)

        ###### Store call informations
        self.call_ = {"X" : X,"Xtot" : Xtot ,"Z" : Z ,"col_marge" : col_marge,"row_marge" : row_marge,"n_components":n_components}
        
        ######################################## Singular Values Decomposition  (SVD) #############################""
        svd = svd_triplet(X=Z,row_weights=row_marge,col_weights=col_marge,n_components=n_components)
        self.svd_ = svd

        # Eigenvalues
        eigen_values = svd["vs"][:min(X.shape[0]-1,X.shape[1]-1)]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns =["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])

        ################################# Coordinates #################################################
        #### Row coordinates
        row_coord = np.apply_along_axis(func1d=lambda x : x*np.sqrt(eigen_values[:n_components]),axis=1,arr=svd["U"])
        row_coord = pd.DataFrame(row_coord,index=X.index.tolist(),columns=["Dim."+str(x+1) for x in range(row_coord.shape[1])])

        #### Columns coordinates
        col_coord = np.apply_along_axis(func1d=lambda x : x*np.sqrt(eigen_values[:n_components]),axis=1,arr=svd["V"])
        col_coord = pd.DataFrame(col_coord,index=X.columns.tolist(),columns=["Dim."+str(x+1) for x in range(col_coord.shape[1])])

        ################################ Contributions #####################################################################################
        ######### Row contributions
        row_contrib = mapply(row_coord,lambda x: 100*(x**2)*row_marge,axis=0,progressbar=False,n_workers=self.n_workers)
        row_contrib = mapply(row_contrib,lambda x : x/eigen_values[:n_components], axis=1,progressbar=False,n_workers=self.n_workers)
        
        ######## Columns contributions
        col_contrib = mapply(col_coord,lambda x: 100*(x**2)*col_marge,axis=0,progressbar=False,n_workers=self.n_workers)
        col_contrib = mapply(col_contrib,lambda x : x/eigen_values[:n_components], axis=1,progressbar=False,n_workers=self.n_workers)
        
        ################################ Cos2 #################################################################
        ###### Row Cos2
        row_disto = mapply(Z,lambda x : (x**2)*col_marge,axis=1,progressbar=False,n_workers=self.n_workers).sum(axis=1)
        row_disto.name = "dist"
        row_cos2 = mapply(row_coord,lambda x: (x**2)/row_disto,axis=0,progressbar=False,n_workers=self.n_workers)
        
        ###### Columns Cos2
        col_disto = mapply(Z,lambda x : (x**2)*row_marge,axis=0,progressbar=False,n_workers=self.n_workers).sum(axis=0)
        col_disto.name = "dist"
        col_cos2 = mapply(col_coord,lambda x: (x**2)/col_disto, axis = 0,progressbar=False,n_workers=self.n_workers)

        ########################################"" Inertia ####################################################
        row_inertia = row_marge*row_disto
        col_inertia = col_marge*col_disto

        ## Row/columns informations
        row_infos = np.c_[np.sqrt(row_disto),row_marge,row_inertia]
        row_infos = pd.DataFrame(row_infos,columns=["dist","marge","inertia"],index=X.index.tolist())
        col_infos = np.c_[np.sqrt(col_disto),col_marge,col_inertia]
        col_infos = pd.DataFrame(col_infos,columns=["dist","marge","inertia"],index=X.columns.tolist())

        ########################################## Store results
        self.col_ = {"coord" : col_coord, "contrib" : col_contrib, "cos2" : col_cos2, "infos" : col_infos}
        self.row_ = {"coord" : row_coord, "contrib" : row_contrib, "cos2" : row_cos2, "infos" : row_infos}
    
        ############################################################################################################
        #  Compute others indicators 
        #############################################################################################################
        # Weighted X with the row weight
        weighted_X = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=self.n_workers)

        # Compute chi - squared test
        statistic,pvalue,dof, expected_freq = sp.stats.chi2_contingency(weighted_X, lambda_=None,correction=False)

        # log - likelihood - tes (G - test)
        g_test_res = sp.stats.chi2_contingency(weighted_X, lambda_="log-likelihood")

        # Absolute residuals
        resid = weighted_X - expected_freq

        # Standardized resid
        standardized_resid = resid/np.sqrt(expected_freq)

        # Adjusted residuals
        adjusted_resid = mapply(standardized_resid,lambda x : x/np.sqrt(1-row_marge),axis=0,progressbar=False,n_workers=self.n_workers)
        adjusted_resid = mapply(adjusted_resid,lambda x : x/np.sqrt(1-col_marge),axis=1,progressbar=False,n_workers=self.n_workers)

        ##### Chi2 contribution
        chi2_contribution = mapply(standardized_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False,n_workers=self.n_workers)

        # Attraction repulsio,
        attraction_repulsion_index = weighted_X/expected_freq

        # Return indicators
        chi2_test = pd.Series([statistic,dof,pvalue],index=["statistic","dof","pvalue"],name="chi-squared test")
        log_likelihood_test = pd.Series([g_test_res[0],g_test_res[1]],index=["statistic","pvalue"],name="g-test")

        # Association test
        association = pd.Series([sp.stats.contingency.association(X, method=name) for name in ["cramer","tschuprow","pearson"]],
                                index=["cramer","tschuprow","pearson"],name="association")

        # Inertia
        inertia = np.sum(row_inertia)
        kaiser_threshold = np.mean(eigen_values)
        kaiser_proportion_threshold = 100/min(X.shape[0]-1,X.shape[1]-1)
       
       # Others informations
        self.others_ = {"res" : resid,
                      "chi2" : chi2_test,
                      "g_test" : log_likelihood_test,
                      "adj_resid" : adjusted_resid,
                      "chi2_contrib" : chi2_contribution,
                      "std_resid" : standardized_resid,
                      "attraction" : attraction_repulsion_index,
                      "association" : association,
                      "inertia" : inertia,
                      "kaiser_threshold" : kaiser_threshold,
                      "kaiser_proportion_threshold" : kaiser_proportion_threshold}

        # Compute supplementary individuals
        if self.row_sup is not None:
            #######
            X_row_sup = X_row_sup.astype("int")
            # Sum row
            row_sum = X_row_sup.sum(axis=1)
            X_row_sup = mapply(X_row_sup,lambda x : x/row_sum,axis=0,progressbar=False,n_workers=self.n_workers)
            # Supplementary coordinates
            row_sup_coord = np.dot(X_row_sup,svd["V"])
            row_sup_coord = pd.DataFrame(row_sup_coord,index=X_row_sup.index.tolist(),columns=["Dim."+str(x+1) for x in range(row_sup_coord.shape[1])])
            # Sup dist2
            row_sup_dist2 = mapply(X_row_sup,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=self.n_workers).sum(axis=1)
            row_sup_dist2.name = "dist"
            # Sup cos2
            row_sup_cos2 = mapply(row_sup_coord,lambda x : (x**2)/row_sup_dist2,axis=0,progressbar=False,n_workers=self.n_workers)
            # Set informations
            self.row_sup_ = {"coord" : row_sup_coord,"cos2" : row_sup_cos2, "dist" : np.sqrt(row_sup_dist2)}

        # Compute supplementary columns
        if self.col_sup is not None:
            X_col_sup = Xtot.iloc[:,col_sup]
            if self.row_sup is not None:
                X_col_sup = X_col_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in row_sup])
            
            # Transform to int
            X_col_sup = X_col_sup.astype("int")
            ### weighted with row weight
            X_col_sup = mapply(X_col_sup,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=self.n_workers)
            # Compute columns sum
            col_sum = X_col_sup.sum(axis=0)
            X_col_sup = mapply(X_col_sup,lambda x : x/col_sum,axis=1,progressbar=False,n_workers=self.n_workers)
            # Sup coord
            col_sup_coord = np.dot(X_col_sup.T,svd["U"])
            col_sup_coord = pd.DataFrame(col_sup_coord,index=X_col_sup.columns.tolist(),columns=["Dim."+str(x+1) for x in range(col_sup_coord.shape[1])])
            # Sup disto
            col_sup_dist2 = mapply(X_col_sup,lambda x : ((x - row_marge)**2)/row_marge,axis=0,progressbar=False,n_workers=self.n_workers).sum(axis=0)
            col_sup_dist2.name = "dist"
            # Sup Cos2
            col_sup_cos2 = mapply(col_sup_coord,lambda x : (x**2)/col_sup_dist2,axis=0,progressbar=False,n_workers=self.n_workers)
            self.col_sup_ = {"coord" : col_sup_coord,"cos2" : col_sup_cos2,"dist" : np.sqrt(col_sup_dist2)}
        
        #################################################################################################################################
        # Compute supplementary continues variables
        #################################################################################################################################
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.iloc[:,quanti_sup]
            if self.row_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in row_sup])
            
            ##### From frame to DataFrame
            if isinstance(X_quanti_sup,pd.Series):
                X_quanti_sup = X_quanti_sup.to_frame()
            
            ################ Transform to float
            X_quanti_sup = X_quanti_sup.astype("float")

            ##################### Compute statistics
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
                
            ############# Compute average mean
            quanti_sup_coord = weightedcorrcoef(x=X_quanti_sup,y=row_coord,w=row_marge)[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
            quanti_sup_coord = pd.DataFrame(quanti_sup_coord,index=X_quanti_sup.columns.tolist(),columns=["Dim."+str(x+1) for x in range(quanti_sup_coord.shape[1])])
            
            #################### Compute cos2
            quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=self.n_workers)

            # Set all informations
            self.quanti_sup_ = {"coord" : quanti_sup_coord, "cos2" : quanti_sup_cos2,"summary_quanti" : summary_quanti_sup}
        
        # Compute supplementary qualitatives informations
        if self.quali_sup is not None:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.row_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in row_sup])
            
            ############### From Frame to DataFrame
            if isinstance(X_quali_sup,pd.Series):
                X_quali_sup = X_quali_sup.to_frame()
            
             ########### Set all elements as objects
            X_quali_sup = X_quali_sup.astype("object")

            # Sum of columns by group
            quali_sup = pd.DataFrame().astype("float")
            for name in X_quali_sup.columns.tolist():
                data = pd.concat((X,X_quali_sup[col]),axis=1).groupby(by=name,as_index=True).sum()
                data.index.name = None
                quali_sup = pd.concat((quali_sup,data),axis=0)
            ############################################################################################
            # Calculate sum by row
            quali_sum = quali_sup.sum(axis=1)
            # Devide by sum
            quali_sup = mapply(quali_sup,lambda x : x/quali_sum,axis=0,progressbar=False,n_workers=self.n_workers)

            ########################################################################################################
            # Categories coordinates
            quali_sup_coord = np.dot(quali_sup,svd["V"])
            quali_sup_coord = pd.DataFrame(quali_sup_coord,index=quali_sup.index.tolist(),columns=["Dim."+str(x+1) for x in range(quali_sup_coord.shape[1])])

            # Categories dist2
            quali_sup_dist2 = mapply(quali_sup,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=self.n_workers).sum(axis=1)
            quali_sup_dist2.name="dist"

            # Sup Cos2
            quali_sup_cos2 = mapply(quali_sup_coord,lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=self.n_workers)

            ############################# Compute
            # Disjonctif table
            dummies = pd.concat((pd.get_dummies(X_quali_sup[col]) for col in X_quali_sup.columns.tolist()),axis=1)
            # Compute : weighted count by categories
            n_k = mapply(dummies,lambda x : x*row_marge,axis=0,progressbar=False,n_workers=self.n_workers).sum(axis=0)*total

            ######## Weighted of coordinates to have 
            if total > 1:
                coef = np.array([np.sqrt(n_k[i]*((total - 1)/(total - n_k[i]))) for i in range(len(n_k))])
            else:
                coef = np.sqrt(n_k)
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*coef,axis=0,progressbar=False,n_workers=self.n_workers)

            ############## Correlation ratio
            ####################################" Correlation ratio #####################################################
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=row_coord.values,weights=row_marge,n_workers=self.n_workers) for col in X_quali_sup.columns.tolist()),axis=0)
            
            #################################### Summary quali
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X.columns.tolist():
                eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")

            ###############"" Set all informations
            self.quali_sup_ = {"coord" : quali_sup_coord,"cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"eta2" : quali_sup_eta2,"dist" : np.sqrt(quali_sup_dist2),"summary_quali" : summary_quali_sup}
        
        self.model_ = "ca"

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

        # Checif if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Set type to int
        X = X.astype("int")
        row_sum = X.sum(axis=1)
        coord = mapply(X,lambda x : x/row_sum,axis=0,progressbar=False,n_workers=n_workers).dot(self.svd_["V"])
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord
            
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        y : None

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.row_["coord"]