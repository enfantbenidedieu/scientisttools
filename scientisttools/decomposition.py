# -*- coding: utf-8 -*-

# https://kiwidamien.github.io/making-a-python-package.html
##################################### Chargement des librairies
from functools import reduce
import itertools
import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
from scipy.sparse import issparse
import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from scientisttools.utils import (
    global_kmo_index,
    per_item_kmo_index,
    eta2,
    from_dummies,
    svd_triplet,
    function_eta2,
    weightedcorrcoef,
    revaluate_cat_variable)

####################################################################################################
#                       PRINCIPAL COMPONENTS ANALYSIS (PCA)
#####################################################################################################

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)
    ----------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    This is a standard Principal Component Analysis implementation
    based on the Singular Value Decomposition

    Performs Principal Component Analysis (PCA) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Missing values are replaced by the column mean.

    Parameters
    ----------
    standardize : a boolean, default = True
        - If True : the data are scaled to unit variance.
        - If False : the data are not scaled to unit variance.

    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights),
                    the weights are given only for active individuals.
    
    var_weights : an optional variables weights (by default, a list/tuple of 1 for uniform variables weights), the weights are given only for
                    the active variables
    
    ind_sup : a list/tuple indicating the indexes of the supplementary individuals

    quanti_sup : a list/tuple indicating the indexes of the quantitative supplementary variables

    quali_sup : a list/tuple indicating the indexes of the categorical supplementary variables

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    var_ : a dictionary of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine, contributions)

    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    quanti_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes)

    quali_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a 
                 criterion with a Normal distribution, and eta2 which is the square correlation corefficient between a qualitative variable and a dimension)
    
    summary_ quali_	: a summary of the results for the categorical variables if quali_sup is not None

    summary_quanti_	: a summary of the results for the quantitative variables
    
    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'pca'

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
    get_pca_ind, get_pca_var, get_pca, summaryPCA, dimdesc, reconstruct

    Examples
    --------
    > X = decathlon2 # from factoextra R package

    > res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X2.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)

    > res_pca.fit(X)

    > summaryPCA(res_pca)
    """
    def __init__(self,
                 standardize=True,
                 n_components=5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup =None,
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
        # Check if sparse matrix
        if issparse(X):
            raise TypeError("Error : PCA does not support sparse input.")
        
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pd.DataFrame class
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
        
        ###### Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns.tolist():
                X[col] = X[col].astype("object")
        
        ############################
        # Check is quali sup
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]

        #  Check if quanti sup
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        ####################################### Check NA
        if X.isnull().any().any():
            if self.quali_sup is None:
                X = mapply(X, lambda x : x.fillna(x.mean(),inplace=True),axis=0,progressbar=False,n_workers=n_workers)
            else:
                col_list = [x for x in list(range(X.shape[0])) if x not in quali_sup]
                X.iloc[:,col_list] = X.iloc[:,col_list].fillna(X[:,col_list].mean())
            print("Missing values are imputed by the mean of the variable.")

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X

        ####################################### Drop supplementary qualitative columns ########################################
        if self.quali_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quali_sup])
        
        ######################################## Drop supplementary quantitatives columns #######################################
        if self.quanti_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quanti_sup])
        
        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ####################################### Principal Components Analysis (PCA) ##################################################

        ################## Summary quantitatives variables ####################
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti

        ###################################### Set number of components ##########################################
        if self.n_components is None:
            n_components = min(X.shape[0]-1,X.shape[1])
        elif not isinstance(self.n_components,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("Error : 'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,X.shape[0]-1,X.shape[1])
        
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

        ########################## Multiply each columns by squared weight ########################################
        # Row information
        ind_dist2 = mapply(Z,lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        ind_dist2.name = "dist"
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        ind_infos = pd.concat([np.sqrt(ind_dist2),ind_inertia],axis=1)
        ind_infos.insert(1,"weight",ind_weights)

        ################################ Columns informations ##################################################
        var_dist2 = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        var_dist2.name = "dist"
        var_inertia = var_dist2*var_weights
        var_inertia.name = "inertia"
        var_infos = pd.concat([np.sqrt(var_dist2),var_inertia],axis=1)
        var_infos.insert(1,"weight",var_weights)
        
        ####### Singular Value Decomposition (SVD) ################################
        svd = svd_triplet(X=Z,row_weights=ind_weights,col_weights=var_weights,n_components=n_components)
        self.svd_ = svd

        # Eigen - values
        eigen_values = svd["vs"][:min(X.shape[0]-1,X.shape[1])]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
        # store in 
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(eig.shape[0])])

        ################################# Coordinates ################################
        # Individuals coordinates
        ind_coord = svd["U"].dot(np.diag(np.sqrt(eigen_values[:n_components])))
        ind_coord = pd.DataFrame(ind_coord,index=X.index.tolist(),columns=["Dim."+str(x+1) for x in range(ind_coord.shape[1])])

        # Variables coordinates
        var_coord = svd["V"].dot(np.diag(np.sqrt(eigen_values[:self.n_components])))
        var_coord = pd.DataFrame(var_coord,index=X.columns.tolist(),columns=["Dim."+str(x+1) for x in range(var_coord.shape[1])])

        ################################# Contributions ####################################
        # Individuals contributions
        ind_contrib = mapply(ind_coord,lambda x : 100*(x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Variables contributions
        var_contrib = mapply(var_coord,lambda x : 100*(x**2)*var_weights,axis=0,progressbar=False,n_workers=n_workers)
        var_contrib = mapply(var_contrib, lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
    
        ####################################### Cos2 ###########################################
        # Individuals Cos2
        ind_cos2 = mapply(ind_coord,lambda x : x**2/ind_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Variables Cos2
        cor_var  = mapply(var_coord,lambda x : x/np.sqrt(var_dist2),axis=0,progressbar=False,n_workers=n_workers)
        var_cos2 = mapply(cor_var,  lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers)

        #### Weighted Pearson correlation
        weighted_corr = weightedcorrcoef(X,w=ind_weights)
        weighted_corr = pd.DataFrame(weighted_corr,index=X.columns.tolist(),columns=X.columns.tolist())

        #################################### Store result #############################################
        self.ind_ = {"coord":ind_coord,"cos2":ind_cos2,"contrib":ind_contrib,"dist":np.sqrt(ind_dist2),"infos" : ind_infos}
        self.var_ = {"coord":var_coord,"cor":cor_var,"cos2":var_cos2,"contrib":var_contrib,"weighted_corr":weighted_corr,"infos" : var_infos}

        ####################################################################################################
        # Bartlett - statistics
        bartlett_stats = -(X.shape[0]-1-(2*X.shape[1]+5)/6)*np.sum(np.log(eigen_values))
        bs_dof = X.shape[1]*(X.shape[1]-1)/2
        bs_pvalue = 1-st.chi2.cdf(bartlett_stats,df=bs_dof)
        bartlett_sphericity_test = pd.Series([bartlett_stats,bs_dof,bs_pvalue],index=["statistic","dof","p-value"],name="Bartlett Sphericity test")
    
        kaiser_threshold = np.mean(eigen_values)
        kaiser_proportion_threshold = 100/np.sum(var_inertia)
        # Karlis - Saporta - Spinaki threshold
        kss_threshold =  1 + 2*np.sqrt((X.shape[1]-1)/(X.shape[0]-1))
        # Broken stick threshold
        broken_stick_threshold = np.flip(np.cumsum(1/np.arange(X.shape[1],0,-1)))[:n_components]

        self.others_ = {"bartlett" : bartlett_sphericity_test,"kaiser" : kaiser_threshold, "kaiser_proportion" : kaiser_proportion_threshold,"kss" : kss_threshold,"bst" : broken_stick_threshold}
        
        ##############################################################################################################################################
        #                                        Compute supplementrary individuals statistics
        ###################################################################################################################################################
        if self.ind_sup is not None:
            ###################### Transform to float ##############################
            X_ind_sup = X_ind_sup.astype("float")

            ########### Standarddize data
            Z_ind_sup = (X_ind_sup - means)/std

            ###### Multiply by variables weights & Apply transition relation
            ind_sup_coord = mapply(Z_ind_sup,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers)
            ind_sup_coord = np.dot(ind_sup_coord,svd["V"])
            ind_sup_coord = pd.DataFrame(ind_sup_coord,index=X_ind_sup.index.tolist(),columns=["Dim."+str(x+1) for x in range(ind_sup_coord.shape[1])])

            ###### Distance to origin
            ind_sup_dist2 = mapply(Z_ind_sup,lambda  x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_sup_dist2.name = "dist"

            ######## Compute cos2
            ind_sup_cos2 = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Store all informations
            self.ind_sup_ = {"coord" : ind_sup_coord,"cos2" : ind_sup_cos2,"dist" : np.sqrt(ind_sup_dist2)}

        ###############################################################################################################################
        #                               Compute supplementary quantitatives variables statistics
        ###############################################################################################################################
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.iloc[:,quanti_sup]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            ###### Transform to float
            X_quanti_sup = X_quanti_sup.astype("float")

            ################ Summary
            self.summary_quanti_.insert(0,"group","active")
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            summary_quanti_sup.insert(0,"group","sup")
            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

            ############# Compute average mean and standard deviation
            d1 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)

            # Initializations - scale data
            means_sup = d1.mean.reshape(1,-1)
            if self.standardize:
                std_sup = d1.std.reshape(1,-1)
            else:
                std_sup = np.ones(X_quanti_sup.shape[1]).reshape(1,-1)
            # Z = (X - mu)/sigma
            Z_quanti_sup = (X_quanti_sup - means_sup)/std_sup

            ####### Compute Supplementary quantitatives variables coordinates
            var_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            var_sup_coord = np.dot(var_sup_coord.T,svd["U"])
            var_sup_coord = pd.DataFrame(var_sup_coord,index=X_quanti_sup.columns.tolist(),columns = ["Dim."+str(x+1) for x in range(var_sup_coord.shape[1])])

            ############# Supplementary quantitatives variables Cos2
            var_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            var_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),var_sup_cor)
            var_sup_cos2 = mapply(var_sup_coord,lambda x : (x**2)/np.sqrt(var_sup_dist2),axis=0,progressbar=False,n_workers=n_workers)

            # Weighted correlation between supplementary quantitatives variables and actives quantitatives
            weighted_sup_corr = weightedcorrcoef(x=X_quanti_sup,y=X,w=ind_weights)[:X_quanti_sup.shape[1],:]
            weighted_sup_corr = pd.DataFrame(weighted_sup_corr,columns=X_quanti_sup.columns.tolist()+X.columns.tolist(),index=X_quanti_sup.columns.tolist())        

            # Store supplementary quantitatives informations
            self.quanti_sup_ =  {"coord":var_sup_coord,
                                 "cor" : var_sup_coord,
                                 "cos2" : var_sup_cos2,
                                 "weighted_corr" : weighted_sup_corr}

        #################################################################################################################################################
        # Compute supplementary qualitatives variables statistics
        ###############################################################################################################################################
        if self.quali_sup is not None:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
            
            ######################################## Barycentre of DataFrame ########################################
            X_quali_sup = X_quali_sup.astype("object")
            ############################################################################################################
            # Check if two columns have the same categories
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            ####################################" Correlation ratio #####################################################
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,
                                                      n_workers=n_workers) for col in X_quali_sup.columns.tolist()),axis=0)

            ###################################### Coordinates ############################################################
            barycentre = pd.DataFrame().astype("float")
            n_k = pd.Series().astype("float")
            for col in X_quali_sup.columns.tolist():
                vsQual = X_quali_sup[col]
                modalite, counts = np.unique(vsQual, return_counts=True)
                n_k = pd.concat([n_k,pd.Series(counts,index=modalite)],axis=0)
                bary = pd.DataFrame(index=modalite,columns=X.columns.tolist())
                for mod in modalite:
                    idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
                    bary.loc[mod,:] = np.average(X.iloc[idx,:],axis=0,weights=ind_weights[idx])
                barycentre = pd.concat((barycentre,bary),axis=0)
            
            ############### Standardize the barycenter
            bary = (barycentre - means)/std
            quali_sup_dist2  = mapply(bary, lambda x : x**2*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            quali_sup_dist2.name = "dist"

            ################################### Barycentrique coordinates #############################################
            quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers)
            quali_sup_coord = quali_sup_coord.dot(svd["V"])
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(quali_sup_coord.shape[1])]

            ################################## Cos2
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
            
            ################################## v-test
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
            quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((X.shape[0]-n_k[k])/((X.shape[0]-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

            #################################### Summary quali
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("effectif").reset_index().rename(columns={"index" : "modalite"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["effectif"] = summary_quali_sup["effectif"].astype("int")

            # Supplementary categories informations
            self.quali_sup_ = {"coord" : quali_sup_coord,
                               "cos2" : quali_sup_cos2,
                               "vtest" : quali_sup_vtest,
                               "dist" : np.sqrt(quali_sup_dist2),
                               "eta2" : quali_sup_eta2,
                               "barycentre" : barycentre}
            self.summary_quali_ = summary_quali_sup
            
        ########################################################################################################
        # store model name
        self.model_ = "pca"

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
        
        # Check if X is a pandas DataFrame
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

        X = X.astype("float")

        ######### check if X.shape[1] = ncols
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("Error : 'columns' aren't aligned")

        # Apply transition relation
        Z = (X - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)

        ###### Multiply by columns weight & Apply transition relation
        coord = mapply(Z,lambda x : x*self.call_["var_weights"],axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"])
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

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]

##################################################################################################################################
#       Correspondence Analysis (CA)
##################################################################################################################################
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
        statistic,pvalue,dof, expected_freq = st.chi2_contingency(weighted_X, lambda_=None,correction=False)

        # log - likelihood - tes (G - test)
        g_test_res = st.chi2_contingency(weighted_X, lambda_="log-likelihood")

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
        association = [st.contingency.association(X, method=name) for name in ["cramer","tschuprow","pearson"]]
        association = pd.Series(association,index=["cramer","tschuprow","pearson"],name="association")

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

####################################################################################
#       MULTIPLE CORRESPONDENCE ANALYSIS (MCA)
####################################################################################

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)
    ---------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Correspondence Analysis (MCA) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); 
                    the weights are given only for the active individuals
    
    var_weights : an optional variables weights (by default, a list/tuple of 1/(number of active variables) for uniform row weights); 
                    the weights are given only for the active variables
    
    benzecri : boolean, if True benzecri correction is applied

    greenacre : boolean, if True greenacre correction is applied

    ind_sup : a list/tuple indicating the indexes of the supplementary individuals

    quali_sup : a list/tuple indicating the indexes of the categorical supplementary variables

    quanti_sup : a list/tuple indicating the indexes of the quantitative supplementary variables

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    var_ : a dictionary of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine, contributions)

    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    quanti_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes)

    quali_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a 
                 criterion with a Normal distribution, and eta2 which is the square correlation corefficient between a qualitative variable and a dimension)
    
    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'mca'

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
    get_mca_ind, get_mca_var, get_mca, summaryMCA, dimdesc

    Examples
    --------
    > X = poison # from FactoMineR R package

    > res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)

    > res_mca.fit(X)

    > summaryMCA(res_mca)
    """
    def __init__(self,
                 n_components = None,
                 ind_weights = None,
                 var_weights = None,
                 benzecri=True,
                 greenacre=True,
                 ind_sup = None,
                 quali_sup = None,
                 quanti_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.benzecri = benzecri
        self.greenacre = greenacre
        self.ind_sup = ind_sup
        self.quali_sup = quali_sup
        self.quanti_sup = quanti_sup
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
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(include=np.number)
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        
        ############################
        # Check is quali sup
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]

        #  Check if quanti sup
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = list(int(self.quanti_sup))
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = list(int(self.ind_sup))
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X

        ####################################### Drop supplementary qualitative columns ########################################
        if self.quali_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quali_sup])
        
        ######################################## Drop supplementary quantitatives columns #######################################
        if self.quanti_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quanti_sup])
        
        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ####################################### Multiple Correspondence Anlysis (MCA) ##################################################
        # Check if 
        X = revaluate_cat_variable(X)
        #########################################################################################################
        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X.columns.tolist():
            eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")
        self.summary_quali_ = summary_quali

        ################################### Chi2 statistic test ####################################
        chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
        idx = 0
        for i in np.arange(X.shape[1]-1):
            for j in np.arange(i+1,X.shape[1]):
                tab = pd.crosstab(X.iloc[:,i],X.iloc[:,j])
                chi = st.chi2_contingency(tab,correction=False)
                row_chi2 = pd.DataFrame({"variable1" : X.columns.tolist()[i],
                                         "variable2" : X.columns.tolist()[j],
                                         "statistic" : chi.statistic,
                                         "dof"       : chi.dof,
                                         "pvalue"    : chi.pvalue},index=[idx])
                chi2_test = pd.concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                idx = idx + 1
        # Transform to int
        chi2_test["dof"] = chi2_test["dof"].astype("int")
        self.chi2_test_ = chi2_test

        ############################################### Dummies tables ############################################
        dummies = pd.concat((pd.get_dummies(X[col]) for col in X.columns.tolist()),axis=1)
        
        ###################################### Set number of components ########################################## 
        if self.n_components is None:
            n_components =  dummies.shape[1] - X.shape[1]
        elif not isinstance(self.n_components,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("Error : 'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,dummies.shape[1] - X.shape[1])
        
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

        ################### Set variables weights ##################################################
        var_weights = pd.Series(name="weight").astype("float")
        if self.var_weights is None:
            for col in X.columns.tolist():
                var_weights[col] = 1/X.shape[1]
        elif not isinstance(self.var_weights,pd.Series):
            raise ValueError("Error : 'var_weights' must be a pandas series where index are variables names and values are variables weights.")
        else:
            for col in X.columns.tolist():
                var_weights[col] = self.var_weights[col]/self.var_weights.values.sum()

        #############################################################################################
        # Effectif par modalite
        I_k = dummies.sum(axis=0)
        # Prorportion par modalité
        p_k = dummies.mean(axis=0)
        Z = pd.concat((dummies.loc[:,k]*(1/p_k[k])-1 for k  in dummies.columns.tolist()),axis=1)

        ###### Define mod weights
        mod_weights = pd.Series(name="weight").astype("float")
        for col in X.columns.tolist():
            data = pd.get_dummies(X[col])
            weights = data.mean(axis=0)*var_weights[col]
            mod_weights = pd.concat((mod_weights,weights),axis=0)
        
        self.call_ = {"Xtot" : Xtot ,
                      "X" : X, 
                      "dummies" : dummies,
                      "Z" : Z , 
                      "row_marge" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "col_marge" : mod_weights,
                      "var_weights" : var_weights,
                      "n_components" : n_components}

        #################### Singular Value Decomposition (SVD) ########################################
        svd = svd_triplet(X=Z,row_weights=ind_weights,col_weights=mod_weights.values,n_components=n_components)
        # Store Singular Value Decomposition (SVD) information
        self.svd_ = svd

        # Eigen - values
        eigen_values = svd["vs"][:(dummies.shape[1]-X.shape[1])]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        ###############################################################
        # Store all informations
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns = ["eigenvalue","difference","proportion","cumulative"],
                                 index=["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        # save eigen value grather than threshold
        lambd = eigen_values[eigen_values>(1/X.shape[1])]
        
        # Benzecri correction
        if self.benzecri:
            if len(lambd) > 0:
                # Apply Benzecri correction
                lambd_tilde = ((X.shape[1]/(X.shape[1]-1))*(lambd - 1/X.shape[1]))**2
                # Cumulative percentage
                s_tilde = 100*(lambd_tilde/np.sum(lambd_tilde))
                # Benzecri correction
                self.benzecri_correction_ = pd.DataFrame(np.c_[lambd_tilde,s_tilde,np.cumsum(s_tilde)],
                                                   columns=["eigenvalue","proportion","cumulative"],
                                                    index = ["Dim."+str(x+1) for x in np.arange(0,len(lambd))])

        # Greenacre correction
        if self.greenacre:
            if len(lambd) > 0:
                # Apply Greenacre correction
                lambd_tilde = ((X.shape[1]/(X.shape[1]-1))*(lambd - 1/X.shape[1]))**2
                s_tilde_tilde = X.shape[1]/(X.shape[1]-1)*(np.sum(eigen_values**2)-(dummies.shape[1]-X.shape[1])/(X.shape[1]**2))
                tau = 100*(lambd_tilde/s_tilde_tilde)
                self.greenacre_correction_ = pd.DataFrame(np.c_[lambd_tilde,tau,np.cumsum(tau)],
                                                    columns=["eigenvalue","proportion","cumulative"],
                                                    index = ["Dim."+str(x+1) for x in np.arange(0,len(lambd))])
        
        ######################################################################################################################
        #################################             Coordinates                             ################################
        ######################################################################################################################
        # Individuals coordinates
        ind_coord = svd["U"].dot(np.diag(np.sqrt(eigen_values[:n_components])))
        ind_coord = pd.DataFrame(ind_coord,index=X.index.tolist(),columns=["Dim."+str(x+1) for x in range(ind_coord.shape[1])])

        # Variables coordinates
        var_coord = svd["V"].dot(np.diag(np.sqrt(eigen_values[:n_components])))
        var_coord = pd.DataFrame(var_coord,index=dummies.columns.tolist(),columns=["Dim."+str(x+1) for x in range(var_coord.shape[1])])

        # Normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        corrected_var_coord = mapply(var_coord,lambda x: x*np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)

        ####################################### Contribution ########################################
        # Individuals contributions
        ind_contrib = mapply(ind_coord,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Variables contributions
        var_contrib = mapply(var_coord,lambda x : (x**2)*mod_weights.values,axis=0,progressbar=False,n_workers=n_workers)
        var_contrib = mapply(var_contrib,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        
        ######################################### Cos2 - Quality of representation ##################################################"
        # Row and columns cos2
        ind_dist2 = mapply(Z,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        ind_dist2.name = "dist"
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        ind_infos = pd.concat([np.sqrt(ind_dist2),ind_inertia],axis=1)
        ind_infos.insert(1,"weight",ind_weights)

        ################################ Columns informations ##################################################
        var_dist2 = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        var_dist2.name = "dist"
        var_inertia = var_dist2*mod_weights
        var_inertia.name = "inertia"
        var_infos = pd.concat([np.sqrt(var_dist2),var_inertia],axis=1)
        var_infos.insert(1,"weight",mod_weights)

        #######################################" Cos2 ########################################################
        # Individuals Cos2
        ind_cos2 = mapply(ind_coord,lambda x : (x**2)/ind_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # Variables Cos2
        var_cos2 = mapply(var_coord,lambda x : (x**2)/var_dist2,axis=0,progressbar=False,n_workers=n_workers)

        ####################################################################################################################
        # Valeur test des modalités
        var_vtest = pd.concat(((var_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*I_k[k])/(X.shape[0]-I_k[k]))).to_frame().T for k in I_k.index.tolist()),axis=0)

        ########################################################################################################################
        #       Qualitative informations
        ####################################" Correlation ratio #####################################################
        quali_eta2 = pd.concat((function_eta2(X=X,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X.columns.tolist()),axis=0)
        
        # Contribution des variables
        quali_contrib = pd.DataFrame().astype("float")
        for col in X.columns.tolist():
            modalite = np.unique(X[col]).tolist()
            contrib = var_contrib.loc[modalite,:].sum(axis=0).to_frame(col).T
            quali_contrib = pd.concat((quali_contrib,contrib),axis=0)

        # Inertia for the variables
        quali_inertia = pd.Series([(len(np.unique(X[col]))-1)/X.shape[0] for col in X.columns.tolist()],index=X.columns.tolist(),name="inertia")

        #####################################
        self.ind_ = {"coord" : ind_coord, "contrib" : ind_contrib, "cos2" : ind_cos2, "infos" : ind_infos}
        self.var_ = {"coord" : var_coord, "corrected_coord":corrected_var_coord,"contrib" : var_contrib, "cos2" : var_cos2, "infos" : var_infos,
                     "vtest" : var_vtest, "eta2" : quali_eta2, "inertia" : quali_inertia,"var_contrib" : quali_contrib}

        # Inertia
        inertia = (dummies.shape[1]/X.shape[1]) - 1

        # Eigenvalue threshold
        kaiser_threshold = 1/X.shape[1]
        kaiser_proportion_threshold = 100/inertia

        self.others_ = {"inertia" : inertia,
                        "threshold" : kaiser_threshold,
                        "proportion" : kaiser_proportion_threshold}
        
        #################################################################################################################
        #   Supplementary individuals informations
        #################################################################################################################
        # Compute supplementary individuals statistics
        if self.ind_sup is not None:
            # Convert to object
            X_ind_sup = X_ind_sup.astype("object")
            # Create dummies table for supplementary
            Y = np.zeros((X_ind_sup.shape[0],dummies.shape[1]))
            for i in np.arange(0,X_ind_sup.shape[0],1):
                values = [X_ind_sup.iloc[i,k] for k in np.arange(0,X.shape[1])]
                for j in np.arange(0,dummies.shape[1],1):
                    if dummies.columns.tolist()[j] in values:
                        Y[i,j] = 1
            ind_sup_dummies = pd.DataFrame(Y,columns=dummies.columns.tolist(),index=X_ind_sup.index.tolist())

            ################################ Supplementary row coordinates ##################################################
            ind_sup_coord = pd.DataFrame(np.zeros(shape=(X_ind_sup.shape[0],n_components)),index=X_ind_sup.index.tolist(),
                                     columns=["Dim."+str(x+1) for x in range(0,n_components)]).astype("float")
            for col in X.columns.tolist():
                modalite = np.unique(X[col]).tolist()
                data1 = ind_sup_dummies.loc[:,modalite]
                data2 = var_coord.loc[modalite,:]
                coord = (var_weights[col])*data1.dot(data2)
                ind_sup_coord = ind_sup_coord.add(coord)
            ind_sup_coord = mapply(ind_sup_coord,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)

            ################################ Supplementary row Cos2 ##########################################################
            Z_sup = pd.concat((ind_sup_dummies.loc[:,k]*(1/p_k[k])-1 for k  in ind_sup_dummies.columns.tolist()),axis=1)
            ind_sup_dist2 = mapply(Z_sup,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_sup_dist2.name = "dist"

            ##### Cos2
            ind_sup_cos2 = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            ##### 
            self.ind_sup_ = {"coord" : ind_sup_coord, "cos2" : ind_sup_cos2, "dist" : ind_sup_dist2}
        
        if self.quali_sup:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])

            #####################"
            X_quali_sup = X_quali_sup.astype("object")
            X_quali_dummies = pd.concat((pd.get_dummies(X_quali_sup[col]) for col in X_quali_sup.columns.tolist()),axis=1)

            # Correlation Ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,
                                                      n_workers=n_workers) for col in X_quali_sup.columns.tolist()),axis=0)
            
            # # Coordinates of supplementary categories - corrected
            quali_sup_coord = mapply(X_quali_dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(ind_coord)
            quali_sup_coord = mapply(quali_sup_coord,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)

            #####################################################################################################################################
            ###### Distance à l'origine
            #####################################################################################################################################
            quali_sup_p_k = X_quali_dummies.mean(axis=0)
            Z_quali_sup = pd.concat((X_quali_dummies.loc[:,k]*(1/quali_sup_p_k[k])-1 for k  in X_quali_dummies.columns.tolist()),axis=1)
            quali_sup_dist2 = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            quali_sup_dist2.name = "dist"

            ################################## Cos2
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
            
            ################################## v-test
            quali_sup_n_k = X_quali_dummies.sum(axis=0)
            quali_sup_vtest = pd.concat(((quali_sup_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*quali_sup_n_k[k])/(X.shape[0] - quali_sup_n_k[k]))).to_frame(name=k).T for k in quali_sup_n_k.index.tolist()),axis=0)

            self.quali_sup_ = {"coord" : quali_sup_coord,
                               "cos2"  : quali_sup_cos2,
                               "dist"  : np.sqrt(quali_sup_dist2),
                               "vtest" : quali_sup_vtest,
                               "eta2"  : quali_sup_eta2}

            #################################### Summary supplementary qualitatives variables ##################################
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            summary_quali_sup.insert(0,"group","sup")
            # Concatenate with activate summary
            self.summary_quali_.insert(0,"group","active")
            self.summary_quali_ = pd.concat((self.summary_quali_,summary_quali_sup),axis=0,ignore_index=True)

            ################################### Chi2 statistic test ####################################
            chi2_test2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in np.arange(X_quali_sup.shape[1]):
                for j in np.arange(X.shape[1]):
                    tab = pd.crosstab(X_quali_sup.iloc[:,i],X.iloc[:,j])
                    chi = st.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                            "variable2" : X.columns.tolist()[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[idx])
                    chi2_test2 = pd.concat((chi2_test2,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            # Transform to int
            chi2_test2["dof"] = chi2_test2["dof"].astype("int")
            chi2_test2.insert(0,"group","sup")
            self.chi2_test_.insert(0,"group","active")
            self.chi2_test_ = pd.concat((self.chi2_test_,chi2_test2),axis=0,ignore_index=True)
            
            ################################### Chi2 statistics between each supplementary qualitatives columns ###################
            if X_quali_sup.shape[1]>1:
                chi2_test3 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for i in np.arange(X_quali_sup.shape[1]-1):
                    for j in np.arange(i+1,X_quali_sup.shape[1]):
                        tab = pd.crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j])
                        chi = st.chi2_contingency(tab,correction=False)
                        row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                                 "variable2" : X_quali_sup.columns.tolist()[j],
                                                 "statistic" : chi.statistic,
                                                 "dof"       : chi.dof,
                                                 "pvalue"    : chi.pvalue},index=[idx])
                        chi2_test3 = pd.concat((chi2_test3,row_chi2),axis=0,ignore_index=True)
                        idx = idx + 1
                # Transform to int
                chi2_test3["dof"] = chi2_test3["dof"].astype("int")
                chi2_test3.insert(0,"group","sup")
                self.chi2_test_ = pd.concat((self.chi2_test_,chi2_test3),axis=0,ignore_index=True)

        if self.quanti_sup:
            X_quanti_sup = Xtot.iloc[:,quanti_sup]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])

            ##############################################################################################################################
            X_quanti_sup = X_quanti_sup.astype("float")
            
            ############# Compute average mean and standard deviation
            d1 = DescrStatsW(X_quanti_sup.values,weights=ind_weights,ddof=0)

            # Z = (X - mu)/sigma
            Z_quanti_sup = (X_quanti_sup -  d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)

            ####### Compute row coord
            quanti_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            quanti_sup_coord = quanti_sup_coord.T.dot(svd["U"])
            quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(quanti_sup_coord.shape[1])]

            ############# Supplementary cos2 ###########################################
            quanti_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_sup_cor)
            quanti_sup_co2 = mapply(quanti_sup_coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)

            #############################################################################################################
            ##################### Compute statistics
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            
            self.quanti_sup_ = {"coord" : quanti_sup_coord,"cos2" : quanti_sup_co2}
            self.summary_quanti_ = summary_quanti_sup

        self.model_ = "mca"

        return self

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------
        
        X is projected on
        the first axes previous extracted from a training set.

        Parameters
        ----------
        X : array of string, int or float, shape (n_rows_sup, n_vars)
            New data, where n_rows_sup is the number of supplementary
            row points and n_vars is the number of variables.
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.
            X rows correspond to supplementary row points that are
            projected onto the axes.

        y : None
            y is ignored.
        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points onto the axes.
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
        
        # Set parallelize option
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        Y = np.zeros((X.shape[0],self.call_["dummies"].shape[1]))
        for i in np.arange(0,X.shape[0],1):
            values = [X.iloc[i,k] for k in np.arange(0,self.call_["X"].shape[1])]
            for j in np.arange(0,self.call_["dummies"].shape[1],1):
                if self.call_["dummies"].columns.tolist()[j] in values:
                    Y[i,j] = 1
        ind_sup_dummies = pd.DataFrame(Y,columns=self.call_["dummies"].columns.tolist(),index=X.index.tolist())

        ################################ Supplementary row coordinates ##################################################
        ind_sup_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],self.ind_["coord"].shape[1])),index=X.index.tolist(),
                                     columns=self.ind_["coord"].columns.tolist()).astype("float")
        for col in self.call_["X"].columns.tolist():
            modalite = np.unique(self.call_["X"][col]).tolist()
            data1 = ind_sup_dummies.loc[:,modalite]
            data2 = self.var_["coord"].loc[modalite,:]
            coord = (self.call_["var_weights"][col])*data1.dot(data2)
            ind_sup_coord = ind_sup_coord.add(coord)
        # Apply correction
        ind_sup_coord = mapply(ind_sup_coord,lambda x : x/np.sqrt(self.eig_.iloc[:,0][:self.call_["n_components"]]),axis=1,progressbar=False,n_workers=n_workers)
        return ind_sup_coord

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
            y is ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]
    

#############################################################################################
#               FACTOR ANALYSIS OF MIXED DATA (FAMD)
#############################################################################################

class FAMD(BaseEstimator,TransformerMixin):
    """
    Factor Analysis of Mixed Data (FAMD)
    ------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Factor Analysis of Mixed Data (FAMD) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    FAMD is a principal component method dedicated to explore data with both continuous and categorical variables. 
    It can be seen roughly as a mixed between PCA and MCA. More precisely, the continuous variables are scaled 
    to unit variance and the categorical variables are transformed into a disjunctive data table (crisp coding) 
    and then scaled using the specific scaling of MCA. This ensures to balance the influence of both continous and 
    categorical variables in the analysis. It means that both variables are on a equal foot to determine the dimensions 
    of variability. This method allows one to study the similarities between individuals taking into account mixed 
    variables and to study the relationships between all the variables.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); 
                    the weights are given only for the active individuals
    
    quanti_weights : an optional quantitatives variables weights (by default, a list/tuple of 1 for uniform quantitative variables weights), 
                        the weights are given only for the active quantitative variables
    
    quali_weights : an optional qualitatives variables weights (by default, a list/tuple of 1/(number of active qualitative variable) for uniform qualitative variables weights), 
                        the weights are given only for the active qualitative variables
    
    ind_sup : a list/tuple indicating the indexes of the supplementary individuals

    quanti_sup : a list/tuple indicating the indexes of the quantitative supplementary variables

    quali_sup : a list/tuple indicating the indexes of the categorical supplementary variables

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

    quali_var_ : a dictionary of pandas dataframe with all the results for the categorical variables (coordinates, square cosine, contributions, v.test)
    
    quanti_var_ : a dictionary of pandas datafrme with all the results for the quantitative variables (coordinates, correlation, square cosine, contributions)

    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'mca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2008), Analyses Factorielles Simples et Multiples.4ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Pages J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_famd_ind, get_famd_var, get_famd, summaryFAMD, dimdesc

    Examples
    --------
    > X = wine # from FactoMineR R package

    > res_famd = FAMD(parallelize=True)

    > res_famd.fit(X)

    > summaryFAMD(res_famd)
    """
    def __init__(self,
                 n_components = None,
                 ind_weights = None,
                 quanti_weights = None,
                 quali_weights = None,
                 ind_sup=None,
                 quanti_sup=None,
                 quali_sup=None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.quanti_weights = quanti_weights
        self.quali_weights = quali_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X, y=None):
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

        # Chack if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize option
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ###### Checks if categoricals variables re in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns.tolist():
                X[col] = X[col].astype("object")
        else:
            raise TypeError("Error : No qualitatives columns in data. Please use PCA function instead.")
        
        ##### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        else:
            raise TypeError("Error : No quantitatives columns in data. Please use MCA function instead.")

        ############################
        # Check is quali sup
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]

        #  Check if quanti sup
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        ####################################### Fill NA in quantitatives columns wih mean
        if is_quanti.isnull().any().any():
            col_list = is_quanti.columns.tolist()
            X[col_list] = mapply(X[col_list], lambda x : x.fillna(x.mean(),inplace=True),axis=0,progressbar=False,n_workers=n_workers)
            raise Warning("Missing values are imputed by the mean of the variable.")

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X

        ####################################### Drop supplementary qualitative columns ########################################
        if self.quali_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quali_sup])
        
        ######################################## Drop supplementary quantitatives columns #######################################
        if self.quanti_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quanti_sup])
        
        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ############################ Split X in quantitatives and qualitatives
        # Compute statistics
        X_quant = X.select_dtypes(include=np.number)
        X_qual = X.select_dtypes(include=["object","category"])

        # Check if NULL
        if X_quant.empty and not X_qual.empty:
            raise ValueError("Error : There is no continuous variables in X. Please use MCA function.")
        elif X_qual.empty and not X_quant.empty:
            raise ValueError("Error : There is no categoricals variables in X. Please use PCA function.")

        ############################################## Summary
        ################## Summary quantitatives variables ####################
        summary_quanti = X_quant.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti

        ################# Summary categoricals variables ##########################
        #########################################################################################################
        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X_qual.columns.tolist():
            eff = X_qual[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")
        self.summary_quali_ = summary_quali
        
        ################################### Chi2 statistic test ####################################
        if X_qual.shape[1]>1:
            chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in np.arange(X_qual.shape[1]-1):
                for j in np.arange(i+1,X_qual.shape[1]):
                    tab = pd.crosstab(X_qual.iloc[:,i],X_qual.iloc[:,j])
                    chi = st.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_qual.columns.tolist()[i],
                                            "variable2" : X_qual.columns.tolist()[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[idx])
                    chi2_test = pd.concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            # Transform to int
            chi2_test["dof"] = chi2_test["dof"].astype("int")
            self.chi2_test_ = chi2_test

        ###########################################################################################
        ########### Set row weight and quanti weight
        ###########################################################################################

        # Set row weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'row_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        ####################################################################################################
        ################################## Treatment of continues variables ################################
        ####################################################################################################
        # Set columns weight
        if self.quanti_weights is None:
            quanti_weights = np.ones(X_quant.shape[1])
        elif not isinstance(self.quanti_weights,list):
            raise ValueError("Error : 'quanti_weights' must be a list of quantitatives weights")
        elif len(self.quanti_weights) != X_quant.shape[1]:
            raise ValueError(f"Error : 'quanti_weights' must be a list with length {X_quant.shape[1]}.")
        else:
            quanti_weights = np.array(self.quanti_weights)
        
        ###########################################################################
        # Weighted Pearson correlation between continuous variables
        col_corr = weightedcorrcoef(x=X_quant,w=ind_weights)

        ############# Compute weighted average mean and standard deviation
        d1 = DescrStatsW(X_quant,weights=ind_weights,ddof=0)
        means = d1.mean.reshape(1,-1)
        std = d1.std.reshape(1,-1)
        Z1 = (X_quant - means)/std

        ###############################################################################################
        ##################################### Treatment of qualitatives variables #####################
        ###############################################################################################

        ################### Set variables weights ##################################################
        quali_weights = pd.Series(index=X_qual.columns.tolist(),name="weight").astype("float")
        if self.quali_weights is None:
            for col in X_qual.columns.tolist():
                quali_weights[col] = 1/X_qual.shape[1]
        elif not isinstance(self.quali_weights,dict):
            raise ValueError("Error : 'quali_weights' must be a dictionary where keys are qualitatives variables names and values are qualitatives variables weights.")
        elif len(self.quali_weights.keys()) != X_qual.shape[1]:
            raise ValueError(f"Error : 'quali_weights' must be a dictionary with length {X_qual.shape[1]}.")
        else:
            for col in X_qual.columns.tolist():
                quali_weights[col] = self.quali_weights[col]/sum(self.quali_weights)
        
        ###################### Set categories weights
        # Normalisation des variables qualitatives
        dummies = pd.concat((pd.get_dummies(X_qual[col]) for col in X_qual.columns.tolist()),axis=1)

        ###### Define mod weights
        mod_weights = pd.Series().astype("float")
        for col in X_qual.columns.tolist():
            data = pd.get_dummies(X_qual[col])
            weights = data.mean(axis=0)*quali_weights[col]
            mod_weights = pd.concat((mod_weights,weights),axis=0)
        
        ############################ Compute weighted mean and weighted standards 
        # Normalize Z2
        p_k = mapply(dummies,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        mean_k = np.average(dummies,axis=0,weights=ind_weights).reshape(1,-1)
        prop = p_k.values.reshape(1,-1)

        #####
        Z2 = (dummies - mean_k)/np.sqrt(prop)

        # Concatenate the 2 dataframe
        Z = pd.concat([Z1,Z2],axis=1)

        #################### Set number of components
        if self.n_components is None:
            n_components = min(X.shape[0]-1, Z.shape[1]-X_qual.shape[1])
        elif not isinstance(self.n_components,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("Error : 'n_components' must be greater or equal than 1.")
        else:
            n_components = min(self.n_components, X.shape[0]-1, Z.shape[1]-X_qual.shape[1])

         #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "quanti" : X_quant,
                      "quali" : X_qual,
                      "dummies" : dummies,
                      "Z" : Z,
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "mod_weights" : pd.Series(1/p_k,index=dummies.columns.tolist(),name="weight"),
                      "means" : pd.Series(means[0],index=X_quant.columns.tolist(),name="average"),
                      "std" : pd.Series(std[0],index=X_quant.columns.tolist(),name="scale"),
                      "means_k" : pd.Series(mean_k[0],index=dummies.columns.tolist(),name="means"),
                      "prop" : pd.Series(prop[0],index=dummies.columns.tolist(),name="prop"),
                      "n_components" : n_components}

        ########################################################################################################################
        #################### Informations about individuals #################################################################### 
        ########################################################################################################################
        # Distance between individuals and inertia center
        ind_dist2 = (mapply(Z1,lambda x : (x**2)*quanti_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)+
                     mapply(Z2,lambda x:  (x - np.sqrt(p_k))**2,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1))
        ind_dist2.name = "dist"
        # Individuals inertia
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        # Save all informations
        ind_infos = pd.concat((np.sqrt(ind_dist2),ind_inertia),axis=1)
        ind_infos.insert(1,"weight",ind_weights)

        ########################################################################################################################
        ################################  Informations about categories ########################################################
        ########################################################################################################################
        # Distance between ctegories
        dummies_weight = (dummies/prop)-1
        # Distance à l'origine
        quali_dist2 = mapply(dummies_weight,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        quali_dist2.name = "dist"
        # Inertie des lignes
        quali_inertia = quali_dist2*mod_weights
        quali_inertia.name = "inertia"
        # Save all informations
        quali_infos = pd.concat((np.sqrt(quali_dist2),quali_inertia),axis=1)
        quali_infos.insert(1,"weight",mod_weights)

        #########################################################################################################
        global_pca = PCA(standardize=False,n_components=n_components).fit(Z)

        ###########################################################################################################
        #                            Compute supplementary individuals informations
        ##########################################################################################################
        if self.ind_sup is not None:
            ##### Prepare supplementary columns
            X_ind_sup_quant = X_ind_sup[X_quant.columns.tolist()]
            X_ind_sup_qual = X_ind_sup[X_qual.columns.tolist()]
            #######
            Z1_ind_sup = (X_ind_sup_quant - means)/std

            Y = np.zeros((X_ind_sup.shape[0],dummies.shape[1]))
            for i in np.arange(0,X_ind_sup.shape[0],1):
                values = [str(X_ind_sup_qual.iloc[i,k]) for k in np.arange(0,X_qual.shape[1])]
                for j in np.arange(0,dummies.shape[1],1):
                    if dummies.columns.tolist()[j] in values:
                        Y[i,j] = 1
            row_sup_dummies = pd.DataFrame(Y,columns=dummies.columns.tolist(),index=X_ind_sup.index.tolist())
            
            Z2_ind_sup = (row_sup_dummies - mean_k)/np.sqrt(prop)
            Z_ind_sup = pd.concat((Z1_ind_sup,Z2_ind_sup),axis=1)
            # Concatenate
            Z_ind_sup = pd.concat((Z,Z_ind_sup),axis=0)
            global_pca = PCA(standardize=False,n_components=n_components,ind_sup=self.ind_sup).fit(Z_ind_sup)
            self.ind_sup_ = global_pca.ind_sup_
        
        ##########################################################################################################
        #                         Compute supplementary quantitatives variables statistics
        ###########################################################################################################
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.iloc[:,quanti_sup]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            ##################################################################################################"
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            self.summary_quanti_.insert(0,"group","active")
            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)
            
            # Standardize
            d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = (X_quanti_sup - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
            Z_quanti_sup = pd.concat((Z,Z_quanti_sup),axis=1)
            # Find supplementary quantitatives columns index
            index = [Z_quanti_sup.columns.tolist().index(x) for x in X_quanti_sup.columns.tolist()]
            # Update PCA
            global_pca = PCA(standardize=False,n_components=n_components,ind_sup=None,quanti_sup=index).fit(Z_quanti_sup)
            self.quanti_sup_ = global_pca.quanti_sup_
        
        ##########################################################################################################
        #                         Compute supplementary qualitatives variables statistics
        ###########################################################################################################
        if self.quali_sup is not None:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            # Chi-squared test between new categorie
            if X_quali_sup.shape[1] > 1:
                chi_sup_stats = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                cpt = 0
                for i in range(X_quali_sup.shpe[1]-1):
                    for j in range(i+1,X_quali_sup.shape[1]):
                        tab = pd.crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j])
                        chi = st.chi2_contingency(tab,correction=False)
                        row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                    "variable2" : X_quali_sup.columns.tolist()[j],
                                    "statistic" : chi.statistic,
                                    "dof"       : chi.dof,
                                    "pvalue"    : chi.pvalue},index=[cpt])
                        chi_sup_stats = pd.concat([chi_sup_stats,row_chi2],axis=0)
                        cpt = cpt + 1
            
            # Chi-squared between old and new qualitatives variables
            chi_sup_stats2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
            cpt = 0
            for i in range(X_quali_sup.shape[1]):
                for j in range(X_qual.shape[1]):
                    tab = pd.crosstab(X_quali_sup.iloc[:,i],X_qual.iloc[:,j])
                    chi = st.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                            "variable2" : X_qual.columns.tolist()[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[cpt])
                    chi_sup_stats2 = pd.concat([chi_sup_stats2,row_chi2],axis=0,ignore_index=True)
                    cpt = cpt + 1
            
            ###### Add 
            if X_quali_sup.shape[1] > 1 :
                chi_sup_stats = pd.concat([chi_sup_stats,chi_sup_stats2],axos=0,ignore_index=True)
            else:
                chi_sup_stats = chi_sup_stats2
            
            #################################### Summary quali
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            summary_quali_sup.insert(0,"group","sup")

            #########
            self.summary_quali_.insert(0,"group","active")
            self.summary_quali_ = pd.concat([self.summary_quali_,summary_quali_sup],axis=0,ignore_index=True)

            ##########################################################################################################################
            #
            #########################################################################################################################
            Z_quali_sup = pd.concat((Z,X_quali_sup),axis=1)
            # Find supplementary quantitatives columns index
            index = [Z_quali_sup.columns.tolist().index(x) for x in X_quali_sup.columns.tolist()]
            # Update PCA
            global_pca = PCA(standardize=False,n_components=n_components,ind_sup=None,quali_sup=index).fit(Z_quali_sup)
            self.quali_sup_ = global_pca.quali_sup_
        
        # Store Singular Value Decomposition
        self.svd_ = global_pca.svd_
        
        # Eigen - values
        eigen_values = global_pca.svd_["vs"][:min(X.shape[0]-1, Z.shape[1]-X_qual.shape[1])]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        ########################### Row informations #################################################################
        self.ind_ = global_pca.ind_

        ############################ Quantitatives columns ###########################################################
        quanti_coord =  global_pca.var_["coord"].loc[X_quant.columns.tolist(),:]
        quanti_contrib = global_pca.var_["contrib"].loc[X_quant.columns.tolist(),:]
        quanti_cos2 = global_pca.var_["cos2"].loc[X_quant.columns.tolist(),:]
        self.quanti_var_ = {"coord" : quanti_coord, "contrib" : quanti_contrib,"cor":quanti_coord,"cos2" : quanti_cos2,"corr" : col_corr}
        
        # Extract categories coordinates form PCA
        pca_coord_mod = global_pca.var_["coord"].loc[dummies.columns.tolist(),:]
        ### Apply correction to have categoricals coordinates
        quali_coord = mapply(pca_coord_mod,lambda x : x*np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
        quali_coord = (quali_coord.T/np.sqrt(prop)).T
        quali_contrib = global_pca.var_["contrib"].loc[dummies.columns.tolist(),:]
        quali_cos2 = mapply(quali_coord,lambda x : (x**2)/quali_dist2,axis=0,progressbar=False,n_workers=n_workers)
        I_k = dummies.sum(axis=0)
        quali_vtest = pd.concat(((quali_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*I_k[k])/(X.shape[0]-I_k[k]))).to_frame(k).T for k in dummies.columns.tolist()),axis=0)
        quali_vtest = mapply(quali_vtest,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
        self.quali_var_ = {"coord" : quali_coord, "contrib" : quali_contrib, "cos2" : quali_cos2, "infos" : quali_infos,"vtest":quali_vtest}

        ####################################   Add elements ###############################################
        #### Qualitatives eta2
        quali_var_eta2 = pd.concat((function_eta2(X=X_qual,lab=col,x=global_pca.ind_["coord"].values,weights=ind_weights,
                                                  n_workers=n_workers) for col in X_qual.columns.tolist()),axis=0)
        # Contributions des variables qualitatives
        quali_var_contrib = mapply(quali_var_eta2,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        # Cosinus carrés des variables qualitatives
        quali_var_cos2 = pd.concat((((quali_var_eta2.loc[col,:]**2)/(len(np.unique(X_qual[[col]]))-1)).to_frame(name=col).T for col in X_qual.columns.tolist()),axis=0)

        var_coord = pd.concat((quanti_cos2,quali_var_eta2),axis=0)
        var_contrib = pd.concat((quanti_contrib,quali_var_contrib),axis=0)
        var_cos2 = pd.concat((quanti_cos2**2,quali_var_cos2),axis=0)
        self.var_ = {"coord" : var_coord,"contrib" : var_contrib,"cos2" : var_cos2}

        if self.quanti_sup is not None and self.quali_sup is not None:
            var_sup_coord = pd.concat((self.quanti_sup_["cos2"],self.quali_sup_["eta2"]),axis=0)
            var_sup_cos2 = pd.concat((self.quanti_sup_["cos2"]**2,self.quali_sup_["cos2"]),axis=0)
            self.var_sup_ = {"coord" : var_sup_coord, "cos2" : var_sup_cos2}
        elif self.quanti_sup is not None:
            var_sup_coord = self.quanti_sup_["cos2"]
            var_sup_cos2 = self.quanti_sup_["cos2"]**2
            self.var_sup_ = {"coord" : var_sup_coord, "cos2" : var_sup_cos2}
        elif self.quali_sup is not None:
            var_sup_coord = self.quali_sup_["eta2"]
            var_sup_cos2 = self.quali_sup_["cos2"]
            self.var_sup_ = {"coord" : var_sup_coord, "cos2" : var_sup_cos2}

        self.model_ = "famd"

        return self

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame, shape (n_rows_sup, n_columns)
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

        # check if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Store continuous and categorical variables
        X_sup_quant = X[self.call_["quanti"].columns.tolist()]
        X_sup_qual = X[self.call_["quali"].columns.tolist()]

        # Standardscaler numerical variable
        Z1 = (X_sup_quant - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)

        # Standardscaler categorical Variable
        Y = np.zeros((X.shape[0],self.call_["dummies"].shape[1]))
        for i in np.arange(0,X.shape[0],1):
            values = [str(X_sup_qual.iloc[i,k]) for k in np.arange(0,X_sup_qual.shape[1])]
            for j in np.arange(0,self.call_["dummies"].shape[1],1):
                if self.call_["dummies"].columns.tolist()[j] in values:
                    Y[i,j] = 1
        Y = pd.DataFrame(Y,index=X.index.tolist(),columns=self.call_["dummies"].columns.tolist())
        # New normalized data
        Z2 = mapply(Y,lambda x : (x - self.call_["means_k"].values)/np.sqrt(self.call_["prop"].values),axis=1,progressbar=False,n_workers=n_workers)
        # Supplementary individuals coordinates
        coord = pd.concat((Z1,Z2),axis=1).dot(self.svd_["V"])
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return  coord

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        ----------------------------------------------------------------

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        y : None
            y is ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        self.fit(X)
        return self.ind_["coord"]

##########################################################################################
#           PARTIAL PRINCIPAL COMPONENT ANALYSIS (PPCA)
##########################################################################################

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
            raise TypeError("Error :  'partial' must be assigned.")
        
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
            raise ValueError("Error : 'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
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
            raise ValueError("Error : 'var_weights' must be a list of variables weights.")
        elif len(self.var_weights) != X.shape[1]:
            raise ValueError(f"Error : 'var_weights' must be a list with length {X.shape[1]}.")
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
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("Error : 'n_components' must be equal or greater than 1.")
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

##############################################################################################
#       EXPLORATORY FACTOR ANALYSIS (EFA)
###############################################################################################

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

#####################################################################################################################
#   MULTIPLE FACTOR ANALYSIS (MFA)
#####################################################################################################################

# https://husson.github.io/MOOC_AnaDo/AFM.html
# https://math.institut-agro-rennes-angers.fr/fr/ouvrages/analyse-factorielle-multiple-avec-r
# https://eudml.org/subject/MSC/62H25
class MFA(BaseEstimator,TransformerMixin):
    """
    Mutiple Factor Analysis (MFA)
    -----------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Factor Analysis in the sense of Escofier-Pages with supplementary individuals 
    and supplementary groups of variables. Active groups of variables must be quantitative. Supplementary groups 
    can be quantitative or categorical

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    group : a list or a tuple with the number of variables in each group

    name_group : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)

    group_type : the type of variables in each group; three possibilities : 
                    - "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
                    - "n" for categorical variables
                    - "m" for mixed variables (quantitative and qualitative variables)
    
    num_group_sup : the indexes of the illustrative groups (by default, None and no group are illustrative)

    ind_sup : an integer, a list or a tuple of the supplementary individuals

    ind_weights : an optional individuals weights (by default, a list of 1/(number of individuals) for uniform weights), the weights 
                    are given only for the active individuals
    
    var_weights_mfa : an optional quantitatives variables weights (by default, a list of 1 for uniform weights), the weights
                        are given only for active quantitatives variables
    
    Parallelize : bool, default = False. Adding multi-core methods to PandasObject.

    Return
    ------
    summary_quali_ : a summary of the results for the categorical variables

    summary_quanti_ : a summary of the results for the quantitative variables

    separate_analyses_ : the results for the separate analyses

    svd_ : a dictionary of matrices containing all the results of the singular value decomposition

    eig_ : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the
            cumulative percentge of variance
    
    group_ : a dictionary of pandas dataframe containing all the results for the groups (Lg and RV coefficients, coordinates, square cosine,
                contributions, distance to the origin, the correlations between each group and each factor)
    
    inertia_ratio_ : inertia ratio

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,
            contributions)
    
    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates,
                square cosine)
    
    quanti_var_ : a dictionary of pandas dataframe containing all the results for the quantitatives variables (coordinates,
                    correlation between variables and axes, contribution, cos2)
    
    quanti_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates,
                        correlation between variables and axes, cos2)
    
    quali_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of 
                        each categories of each variables, cos2 and vtest which is a criterion with a normal distribution)
    
    partial_axes_ : a dictionary of pandas dataframe containing all the results for the partial axes (coordinates, correlation between variables
                        and axes, correlation between partial axes)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfa'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    Escofier B, Pagès J (1984), l'Analyse factorielle multiple, Cahiers du Bureau universitaire de recherche opérationnelle. Série Recherche, tome 42 (1984), p. 3-68
    Escofier B, Pagès J (1983), Méthode pour l'analyse de plusieurs groupes de variables. Application à la caractérisation de vins rouges du Val de Loire. Revue de statistique appliquée, tome 31, n°2 (1983), p. 43-59
    """
    def __init__(self,
                 n_components=5,
                 group = None,
                 name_group = None,
                 group_type = None,
                 num_group_sup = None,
                 ind_sup = None,
                 ind_weights = None,
                 var_weights_mfa = None,
                 parallelize=False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.group_type = group_type
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.var_weights_mfa = var_weights_mfa
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : pandas/polars DataFrame of shape (n_rows, n_columns)

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
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
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
            
        ###### Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns.tolist():
                X[col] = X[col].astype("object")
        
        ###### Transform all quantitatives columns to float
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")
        
        ########################################################################################################################
        #   check if two categoricals variables have same categories
        ######################################################################################################################
        X = revaluate_cat_variable(X)

        #########################################################################################################################
        #   Check if group is None
        #########################################################################################################################
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("Error : 'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]

        ##########################################################################################################################
        # Remove supplementary group
        if self.num_group_sup is not None:
            # Set default values to None
            self.quali_var_sup_ = None
            self.quanti_var_sup_ = None
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]

        #########################################################################################################################
        #   Check if group type in not None
        #########################################################################################################################
        if self.group_type is None:
            raise ValueError("Error : 'group_type' must be assigned.")
        
        #######################################################################################################################
        if len(self.group) != len(self.group_type):
            raise TypeError("Error : Not convenient group definition")
        
        ############################################################################################################################
        #  Assigned group name
        ###########################################################################################################################
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("Error : 'group_name' must be a list or a tuple of group name")
        else:
            group_name = [x for x in self.name_group]
        
        ##############################################################################################################################
        # check if group name is an integer
        #############################################################################################################################
        for i in range(len(group_name)):
            if isinstance(group_name[i],int) or isinstance(group_name[i],float):
                group_name[i] = "Gr"+str(i+1)
        
        ##############################################################################################################################
        #   Assigned group name to label
        #############################################################################################################################
        group_active_dict = {}
        group_sup_dict = {}
        debut = 0
        for i in range(len(nb_elt_group)):
            X_group = X.iloc[:,(debut):(debut+nb_elt_group[i])]
            if self.num_group_sup is not None:
                if i in num_group_sup:
                    new_elt = {group_name[i]:X_group.columns.tolist()}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
            debut = debut + nb_elt_group[i]

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

       ######################################## Drop supplementary groups columns #######################################
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))
        
        ######################################## Drop supplementary individuals  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            # Drop supplementary individuals
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"Error : {grp} group should have at least two columns")
        
        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        if not all_num:
            raise TypeError("Error : All actives columns must be numeric")
        
        ####################################### Multiple Factor Analysis (MFA) ##################################################

        ################## Summary quantitatives variables ####################
        summary_quanti = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            summary = X[cols].describe().T.reset_index().rename(columns={"index" : "variable"})
            summary["count"] = summary["count"].astype("int")
            summary.insert(0,"group",group_name.index(grp))
            summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)
        self.summary_quanti_ = summary_quanti

        ########### Set individuals weight and variables weight
        # Set row weight
        if self.ind_weights is None:
            ind_weights = (np.ones(X.shape[0])/X.shape[0]).tolist()
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of individuals weights")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = [x/np.sum(self.ind_weights) for x in self.ind_weights]
        
        ############################# Set columns weight MFA
        var_weights_mfa = {}
        if self.var_weights_mfa is None:
            for grp, cols in group_active_dict.items():
                var_weights_mfa[grp] = np.ones(len(cols)).tolist()
        elif not isinstance(self.var_weights_mfa,dict):
            raise ValueError("Error : 'var_weights_mfa' must be a dictionary where keys are groups names and values are list of variables weights in group.")
        else:
            for grp, cols in group_active_dict.items():
                var_weights_mfa[grp] = np.array(self.var_weights_mfa[grp]).tolist()
        
        # Run a Factor Analysis in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if self.group_type[group_name.index(grp)]=="c":
                # Center Principal Components Anlysis (PCA)
                fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,var_weights=var_weights_mfa[grp],ind_sup=None,parallelize=self.parallelize)
            elif self.group_type[group_name.index(grp)]=="s":
                # Scale Principal Components Anlysis (PCA)
                fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,var_weights=var_weights_mfa[grp],ind_sup=None,parallelize=self.parallelize)
            else:
                raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
            model[grp] = fa.fit(X[cols])

            ##### Add supplementary individuals
            if self.ind_sup is not None:
                # Select 
                X_ind_sup = X_ind_sup.astype("float")

                if self.group_type[group_name.index(grp)]=="c":
                    # Center Principal Components Anlysis (PCA)
                    fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,var_weights=var_weights_mfa[grp],ind_sup=ind_sup,parallelize=self.parallelize)
                elif self.group_type[group_name.index(grp)]=="s":
                    # Scale Principal Components Anlysis (PCA)
                    fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,var_weights=var_weights_mfa[grp],ind_sup=ind_sup,parallelize=self.parallelize)
                else:
                    raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        ############################################### Separate  Factor Analysis for supplementary groups ######################################""
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            ## Factor Analysis
            for grp, cols in group_sup_dict.items():
                # Instnce the FA model
                if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
                elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        fa = MCA(n_components=None,parallelize=self.parallelize,benzecri=False,greenacre=False)
                    else:
                        raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        fa = FAMD(n_components=None,ind_weights=ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for mixed group 'group_type' should be 'm'")
                # Fit the model
                model[grp] = fa.fit(X_group_sup[cols])

        ##################### Compute group disto
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="dist2")

        ##### Compute group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="dist2")

        ##### Store separate analysis
        self.separate_analyses_ = model

        ################################################# Standardize Data ##########################################################
        means = {}
        std = {}
        base        = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        for grp,cols in group_active_dict.items():
            ############################### Compute Mean and Standard deviation #################################
            d1 = DescrStatsW(X[cols],weights=ind_weights,ddof=0)
            ########################### Standardize #################################################################################
            Z = (X[cols] - d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)
            ###################" Concatenate
            base = pd.concat([base,Z],axis=1)
            ##################################"
            means[grp] = d1.mean.reshape(1,-1)
            std[grp] = d1.std.reshape(1,-1)
            ################################ variables weights
            weights = pd.Series(np.repeat(a=1/model[grp].eig_.iloc[0,0],repeats=len(cols)),index=cols)
            # Ajout de la pondération de la variable
            weights = weights*np.array(var_weights_mfa[grp])
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        # Number of components
        if self.n_components is None:
            n_components = min(base.shape[0]-1,base.shape[1])
        else:
            n_components = min(self.n_components,base.shape[0]-1,base.shape[1])

        # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "var_weights" : var_weights,
                      "means" : means,
                      "std" : std,
                      "group" : group_active_dict,
                      "group_name" : group_name}
        
        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        # Global PCA without supplementary element
        global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),parallelize = self.parallelize).fit(base)

        ###############################################################################################
        #### Add supplementary individuals
        ###############################################################################################
        if self.ind_sup is not None:
            X_ind_sup = X_ind_sup.astype("float")
            # Concatenate
            Z_ind_sup = pd.concat((base,X_ind_sup),axis=0)
            # Apply PCA
            global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),ind_sup=ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
            self.ind_sup_ = global_pca.ind_sup_.copy()
        
        #######################################""
        ###### Add supplementary group
        ################################################################################################################################""
        if self.num_group_sup is not None:
            X_sup_quanti = X_group_sup.select_dtypes(exclude=["object","category"])
            X_sup_quali = X_group_sup.select_dtypes(include=["object","category"])
            if X_sup_quanti.shape[1]>0:
                ##################################################################################################"
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
                summary_quanti_sup.insert(0,"group",group_name.index(grp))
                self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

                ####### Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
                ### Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns.tolist()]
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            if X_sup_quali.shape[1]>1:
                # Concatenate
                Z_quali_sup = pd.concat((base,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_.copy()
                summary_quali_var_sup.insert(0,"group",group_name.index(grp))
                
                # Append 
                self.summary_quali_ = summary_quali_var_sup

        ##########################################
        self.global_pca_ = global_pca
        ####################################################################################################
        #  Eigenvalues
        ####################################################################################################
        self.eig_ = global_pca.eig_.copy()

        ####################################################################################################
        #   Singular Values Decomposition (SVD)
        ####################################################################################################
        self.svd_ = global_pca.svd_.copy()

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################
        ind = global_pca.ind_.copy()

        ####################################################################################################
        #   Variables informations : coordinates, cos2 and contrib
        ####################################################################################################
        # Correlation between variables en axis
        quanti_var_coord = weightedcorrcoef(x=X,y=ind["coord"],w=None)[:X.shape[1],X.shape[1]:]
        quanti_var_coord = pd.DataFrame(quanti_var_coord,index=X.columns.tolist(),columns=["Dim."+str(x+1) for x in range(quanti_var_coord.shape[1])])
        # Contribution
        quanti_var_contrib = global_pca.var_["contrib"].copy()
        # Cos2
        quanti_var_cos2 = global_pca.var_["cos2"].copy()
        ### Store all informations
        self.quanti_var_ = {"coord" : quanti_var_coord,"cor" : quanti_var_coord,"contrib":quanti_var_contrib,"cos2":quanti_var_cos2}

        ########################################################################################################### 
        # Partiel coordinates for individuals
        ###########################################################################################################
        ##### Add individuals partiels coordinaates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Standardisze data
            Z = (X[cols] - means[grp])/std[grp]
            # Partial coordinates
            coord_partial = mapply(Z.dot(quanti_var_coord.loc[cols,:]),lambda x : x/self.separate_analyses_[grp].eig_.iloc[0,0],axis=0,progressbar=False,n_workers=n_workers)
            coord_partial = len(list(group_active_dict.keys()))*mapply(coord_partial,lambda x : x/np.sqrt(self.eig_.iloc[:,0].values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
            coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
            ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
        
        ind["coord_partiel"] = ind_coord_partiel

        ##########################################################################################################
        #   Partiel coordinates for supplementary qualitatives columns
        ###########################################################################################################
        if self.num_group_sup is not None:
            quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp_sup, cols_sup in group_sup_dict.items():
                # If all columns in group are categoricals
                if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        ############################################################################################################################
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup[col]),axis=1).groupby(col).mean()for col in cols_sup),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
                # If at least one columns is categoricals
                elif any(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        ############################################################################################################################
                        X_group_sup_quali = X_group_sup[cols_sup].select_dtypes(include=['object'])
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup_quali[col]),axis=1).groupby(col).mean()for col in X_group_sup_quali.columns.tolist()),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
            # Store
            self.quali_var_sup_["coord_partiel"] = quali_var_sup_coord_partiel

        ################################################################################################"
        #    Inertia Ratios
        ################################################################################################
        #### "Between" inertia on axis s
        between_inertia = len(list(group_active_dict.keys()))*mapply(ind["coord"],lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        between_inertia.name = "between_inertia"

        ### Total inertial on axis s
        total_inertia = pd.Series(name="total_inertia").astype("float")
        for dim in ind["coord"].columns.tolist():
            value = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum().sum()
            inertia = pd.Series([value],index=[dim],name="total_inertia")
            total_inertia = pd.concat((total_inertia,inertia),axis=0)

        ### Inertia ratio
        inertia_ratio = between_inertia/total_inertia
        inertia_ratio.name = "inertia_ratio"
        self.inertia_ratio_ = inertia_ratio

        ##############################################################################################################
        #   Individuals Within inertia
        ##############################################################################################################
        ############################### Within inertia ################################################################
        ind_within_inertia = pd.DataFrame(index=X.index.tolist(),columns=ind["coord"].columns.tolist()).astype("float")
        for dim in ind["coord"].columns.tolist():
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        ######################################## Within partial inertia ################################################
        data = pd.DataFrame().astype("float")
        for dim in ind["coord"].columns.tolist():
            data1 = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers)
            data1 = 100*data1/data1.sum().sum()
            data = pd.concat([data,data1],axis=1)

        ######## Rorder inertia by group
        ind_within_partial_inertia = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            partial_inertia = data[grp]
            partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
            ind_within_partial_inertia = pd.concat([ind_within_partial_inertia,partial_inertia],axis=1)
        ind["within_partial_inertia"] = ind_within_partial_inertia

        #################"" Store 
        self.ind_ = ind

        ##################################################################################################
        #   Partial axes informations
        #################################################################################################
        ########################################### Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            ######### Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        ############################################## Partial axes cos2
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        #########" Partial correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        #### Add 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
                all_coord = pd.concat([all_coord,data],axis=1)
            # Reorder
            all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
        
        #################################### Partial axes contrib ################################################"
        axes_contrib = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
            eig = self.separate_analyses_[grp].eig_.iloc[:nbcol,0].values/self.separate_analyses_[grp].eig_.iloc[0,0]
            contrib = mapply(partial_axes_coord[grp].iloc[:,:nbcol],lambda x : (x**2)*eig,axis=1,progressbar=False,n_workers=n_workers)
            contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
            axes_contrib  = pd.concat([axes_contrib,contrib],axis=1)
        
        partial_axes_contrib = mapply(axes_contrib,lambda x : 100*x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)

        #### Add a null dataframe
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
                contrib = pd.DataFrame(np.zeros(shape=(n_components,nbcol)),index=["Dim."+str(x+1) for x in range(n_components)],columns=["Dim."+str(x+1) for x in range(nbcol)])
                contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
                partial_axes_contrib = pd.concat([partial_axes_contrib,contrib],axis=1)
            ## Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])
                
        ###############
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : all_coord.corr()}
        
        #################################################################################################################
        # Group informations : coord
        #################################################################################################################
        # group coordinates
        group_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].call_["Z"]
            coord =  (weightedcorrcoef(data,self.ind_["coord"],w=None)[:data.shape[1],data.shape[1]:]**2).sum(axis=0)/self.separate_analyses_[grp].eig_.iloc[0,0]
            coord  = pd.DataFrame(coord.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_coord = pd.concat((group_coord,coord),axis=0)
        
        ########################################### Group contributions ############################################
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)

        ######################################## group cos2 ################################################################
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_dist2.loc[grp]).to_frame(grp).T for grp in group_coord.index.tolist()),axis=0)

        ########################################### Group correlations ###############################################
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=self.ind_["coord"],w=None)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_correlation = pd.concat((group_correlation,correl),axis=0)

        #################################################################################################################
        # Measuring how similar groups
        #################################################################################################################
        Lg = pd.DataFrame().astype("float")
        for grp1,cols1 in group_active_dict.items():
            for grp2,cols2 in group_active_dict.items():
                # Sum of square coefficient of correlation
                sum_corr2 = np.array([(weightedcorrcoef(x=X[col1],y=X[col2],w=None)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                # Weighted the sum using the eigenvalues of each group
                weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                Lg.loc[grp1,grp2] = weighted_corr2
        
        # Reorder using active group name
        Lg = Lg.loc[list(group_active_dict.keys()),list(group_active_dict.keys())]

        # Add supplementary Lg elements
        if self.num_group_sup is not None:
            Lg_sup = pd.DataFrame().astype("float")
            for grp1, cols1 in group_sup_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    if (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square coefficient of correlation
                        sum_corr2 = np.array([(weightedcorrcoef(x=X_group_sup[col1],y=X_group_sup[col2],w=None)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using the eigenvalues of each group
                        weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                        Lg_sup.loc[grp1,grp2] = weighted_corr2
                    elif ((pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X_group_sup[col1],X_group_sup[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg_sup.loc[grp1,grp2] = weighted_chi2
                    elif (all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col1],X_group_sup[col2],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(len(cols1)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
                    elif (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col2],X_group_sup[col1],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
            
            ####### Concatenate
            Lg = pd.concat((Lg,Lg_sup),axis=0)
            # Fill NA with 0.0
            Lg = Lg.fillna(0.0)

            ####
            for grp1,cols1 in group_active_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    X1, X2 = X[cols1], X_group_sup[cols2]
                    if all(pd.api.types.is_numeric_dtype(X2[col]) for col in cols2):
                        # Sum of square coefficient of correlation
                        sum_corr2 = np.array([(np.corrcoef(X1[col1],X2[col2],rowvar=False)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using the eigenvalues of each group
                        weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                        Lg.loc[grp1,grp2] = weighted_corr2
                        Lg.loc[grp2,grp1] = weighted_corr2
                    elif all(pd.api.types.is_string_dtype(X2[col]) for col in cols2):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X2[col2],X1[col1],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0][0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                        Lg.loc[grp1,grp2] = weighted_eta2
                        Lg.loc[grp2,grp1] = weighted_eta2
            
            ##################
            Lg = Lg.loc[group_name,group_name]
        
        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        ## RV Coefficient
        RV = pd.DataFrame().astype("float")
        for grp1 in Lg.index:
            for grp2 in Lg.columns:
                RV.loc[grp1,grp2] = Lg.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1])*np.sqrt(Lg.loc[grp2,grp2]))
        
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}

        ##### Add supplementary elements
        if self.num_group_sup is not None:
            group_sup_coord = pd.DataFrame().astype("float")
            for grp, cols in group_sup_dict.items():
                Xg = X_group_sup[cols]
                if all(pd.api.types.is_numeric_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    correl = np.sum((weightedcorrcoef(self.separate_analyses_[grp].call_["Z"],self.ind_["coord"],w=None)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/self.separate_analyses_[grp].eig_.iloc[0,0]
                    coord = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns = ["Dim."+str(x+1) for x in range(len(correl))])
                    group_sup_coord = pd.concat((group_sup_coord,coord),axis=0)

                elif all(pd.api.types.is_string_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    data = self.quali_var_sup_["eta2"].loc[cols,:]
                    coord = (data.sum(axis=0)/(Xg.shape[1]*self.separate_analyses_[grp].eig_.iloc[0,0]))
                    group_sup_coord = pd.concat((group_sup_coord,coord.to_frame(grp).T),axis=0)
                else:
                    raise TypeError("Error : All columns should have the same type.")
            
            #################################### group sup cos2 ###########################################################
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index.tolist()),axis=0)
            
            # Append two dictionnaries
            self.group_ = {**self.group_,**{"coord_sup" : group_sup_coord, "dist2_sup" : group_sup_dist2,"cos2_sup" : group_sup_cos2}}
            
        self.model_ = "mfa"
        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters:
        ----------
        X : pandas/polars DataFrame of shape (n_rows_,n_cols_)

        y : None
            y is ignored
        
        Return
        ------
        X_new : pandas dataframe of shape (n_rows, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ----------------------------------------

        X is projected on the first axes previous extracted from a
        training set

        Parameters
        ----------
        X : pandas/polars DataFrame of shape (n_rows_sup, n_cols_)
            New data, where n_rows_sup is the number of supplementary
            row points and n_cols_ is the number of columns.
            X rows correspond to supplementary row points that are projected
            on the axes.
        
        y : None
            y is ignored
        
        Return
        ------
        X_new : array of float, shape (n_row_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("Error : DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ######################################""
        row_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],self.call_["n_components"])),index=X.index.tolist(),
                                 columns=["Dim."+str(x+1) for x in range(self.call_["n_components"])])
        for grp, cols in self.call_["group"].items():
            # Standardize the data using 
            Z = (X[cols] - self.call_["means"][grp])/self.call_["std"][grp]
            # Partial coordinates
            coord = mapply(Z.dot(self.quanti_var_["coord"].loc[Z.columns.tolist(),:]),lambda x : x/self.separate_analyses_[grp].eig_.iloc[0,0],axis=0,progressbar=False,n_workers=n_workers)
            row_coord = row_coord + coord
        ################################# Divide by eigenvalues ###########################
        row_coord = mapply(row_coord, lambda x : x/np.sqrt(self.eig_.iloc[:,0][:self.call_["n_components"]]),axis=1,progressbar=False,n_workers=n_workers)
        
        return row_coord
        
####################################################################################################################
#  MULTIPLE FACTOR ANALYSIS FOR QUALITATIVES VARIABLES
####################################################################################################################
class MFAQUAL(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis for Qualitatives Variables (MFAQUAL)
    -------------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Factor Analysis for Qualitatives Variables in the sense of Pagès J. (2002) with supplementary individuals
    and supplementary groups of variables. Active groups of variables must be qualitatives. Supplementary groups of variables 
    can be quantitative or categorical
 
    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    group : a list or a tuple with the number of variables in each group

    name_group : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)

    group_type : the type of variables in each group; three possibilities : 
                    - "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
                    - "n" for categorical variables
                    - "m" for mixed variables (quantitative and qualitative variables)
    
    num_group_sup : the indexes of the illustrative groups (by default, None and no group re illustrative)

    ind_sup : an integer, a list or a tuple of the supplementary individuals

    ind_weights : an optional individuals weights (by default, a list of 1/(number of individuals) for uniform weights), the weights 
                    are given only for the active individuals
    
    var_weights_mfa : an optional qualitatives variables weights (by defaut, a list of 1/(number of categoricals variables in the group)),
                            the weights are given only for qualitatives variables
    
    Parallelize : bool, default = False. Adding multi-core methods to PandasObject.

    Return
    ------
    summary_quali_ : a summary of the results for the categorical variables

    summary_quanti_ : a summary of the results for the quantitative variables

    separate_analyses_ : the results for the separate analyses

    svd_ : a dictionary of matrices containing all the results of the singular value decomposition

    eig_ : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the
            cumulative percentge of variance
    
    group_ : a dictionary of pandas dataframe containing all the results for the groups (Lg and RV coefficients, coordinates, square cosine,
                contributions, distance to the origin, the correlations between each group and each factor)
    
    inertia_ratio_ : inertia ratio

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,
            contributions)
    
    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates,
                square cosine)
    
    quali_var_ : a dictionary of pandas dataframe containing all the results for the categorical variables (coordinates of each categories
                    of each variables, contribution and vtest which is a criterion with a normal distribution)
    
    quanti_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates,
                        correlation between variables and axes, cos2)
    
    quali_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of 
                        each categories of each variables, cos2 and vtest which is a criterion with a normal distribution)
    
    partial_axes_ : a dictionary of pandas dataframe containing all the results for the partial axes (coordinates, correlation between variables
                        and axes, correlation between partial axes)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfaqual'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    Pagès J (2002), Analyse factorielle multiple appliquée aux variables qualitatives et aux données mixtes, Revue de statistique appliquée, tome 50, n°4(2002), p. 5-37
    """
    def __init__(self,
                 n_components=5,
                 group = None,
                 name_group = None,
                 group_type = None,
                 num_group_sup = None,
                 ind_sup = None,
                 ind_weights = None,
                 var_weights_mfa = None,
                 parallelize=False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.group_type = group_type
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.var_weights_mfa = var_weights_mfa
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
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
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
        
        ###### Transform all categoricals variables to object
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns.tolist():
            X[col] = X[col].astype("object")

        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        
        ########################################################################################################################
        #   check if two categoricals variables have same categories
        ######################################################################################################################
        X = revaluate_cat_variable(X)

        #########################################################################################################################
        #   Check if group is None
        #########################################################################################################################
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("Error : 'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]
        
        #######################################################################################################################
        # Remove supplementary group
        if self.num_group_sup is not None:
            # Set default values to None
            self.quali_var_sup_ = None
            self.quanti_var_sup_ = None
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        ####################################################################
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        #########################################################################################################################
        #   Check if group type in not None
        #########################################################################################################################
        if self.group_type is None:
            raise ValueError("Error : 'group_type' must be assigned.")
        
        #######################################################################################################################
        if len(self.group) != len(self.group_type):
            raise TypeError("Error : Not convenient group definition")

        ############################################################################################################################
        #  Assigned group name
        ###########################################################################################################################
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("Error : 'group_name' must be a list or a tuple of group name")
        else:
            group_name = [x for x in self.name_group]
        
        ##############################################################################################################################
        # check if group name is an integer
        #############################################################################################################################
        for i in range(len(group_name)):
            if isinstance(group_name[i],int) or isinstance(group_name[i],float):
                group_name[i] = "Gr"+str(i+1)
        
        ##############################################################################################################################
        #   Assigned group name to label
        #############################################################################################################################
        group_active_dict = {}
        group_sup_dict = {}
        debut = 0
        for i in range(len(nb_elt_group)):
            X_group = X.iloc[:,(debut):(debut+nb_elt_group[i])]
            if self.num_group_sup is not None:
                if i in num_group_sup:
                    new_elt = {group_name[i]:X_group.columns.tolist()}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
            debut = debut + nb_elt_group[i]

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

        ######################################## Drop supplementary groups columns #######################################
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))

        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
         ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"Error : {grp} group should have at least two columns")
        
        # Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist())
        if not all_cat:
            raise TypeError("Error : All actives columns must be categoricals")

        ####################################### Multiple Factor Analysis for Qualitatives Variables (MFAQUAL) ##################################################

        ########################################## Summary qualitatives variables ###############################################
        # Compute statisiques
        summary_quali = pd.DataFrame()
        for grp, cols in group_active_dict.items():
            for col in cols:
                eff = X[col].value_counts().to_frame("effectif").reset_index().rename(columns={"index" : "modalite"})
                eff.insert(0,"variable",col)
                eff.insert(0,"group",group_name.index(grp))
                summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["effectif"] = summary_quali["effectif"].astype("int")
        self.summary_quali_ = summary_quali
        
        ########### Set row weight and columns weight
        # Set row weight
        if self.ind_weights is None:
            ind_weights = (np.ones(X.shape[0])/X.shape[0]).tolist()
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of individuals weights")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = [x/np.sum(self.ind_weights) for x in self.ind_weights]
        
        ############################# Set variables weight MFA #######################################
        var_weights_mfa = {}
        if self.var_weights_mfa is None:
            for grp, cols in group_active_dict.items():
                var_weights = pd.Series(index=cols,name="weights").astype("float")
                for col in cols:
                    var_weights[col] = 1/len(cols)
                var_weights_mfa[grp] = var_weights
        elif not isinstance(self.var_weights_mfa,dict):
            raise ValueError("Error : 'var_weights_mfa' must be a dictionary where keys are groups names and values are pandas series of variables weights in group.")
        else:
            for grp, cols in group_active_dict.items():
                var_weights = pd.Series(index=cols,name="weights").astype("float")
                for col in cols:
                    var_weights[col] = self.var_weights_mfa[grp][col]/self.var_weights_mfa[grp].values.sum()
                var_weights_mfa[grp] = var_weights
            
        # Run a Factor Analysis in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if self.group_type[group_name.index(grp)]=="n":
                # Multiple Correspondence Analysis (MCA)
                fa = MCA(n_components=None,ind_weights=ind_weights,ind_sup=None,var_weights=var_weights_mfa[grp],benzecri=False,greenacre=False,parallelize=self.parallelize)
            else:
                raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
            model[grp] = fa.fit(X[cols])

            ##### Add supplementary individuals
            if self.ind_sup is not None:
                # Select 

                X_ind_sup = X_ind_sup.astype("float")
                if self.group_type[group_name.index(grp)]=="c":
                    fa = MCA(n_components=None,ind_weights=ind_weights,ind_sup=ind_sup,var_weights=var_weights_mfa[grp],benzecri=False,greenacre=False,parallelize=self.parallelize)
                else:
                    raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        ############################################### Separate  Factor Analysis for supplementary groups ######################################""
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            for grp, cols in group_sup_dict.items():
                # Instance the FA model
                if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
                elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        fa = MCA(n_components=None,ind_weights=ind_weights,benzecri=False,greenacre=False,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        fa = FAMD(n_components=None,ind_weights=ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for mixed group 'group_type' should be 'm'")
                model[grp] = fa.fit(X_group_sup[cols])

        ##################### Compute group disto
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="dist2")

        ##### Compute group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="dist2")

        ##### Store separate analysis
        self.separate_analyses_ = model

        ################################################# Standardize Data ##########################################################
        base        = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        for grp,cols in group_active_dict.items():
            # Compute dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
            # Effectif par categories
            I_k = dummies.sum(axis=0)
            # Apply standardize
            Z = pd.concat((dummies.loc[:,[k]]*(X.shape[0]/I_k[k])-1 for k  in dummies.columns.tolist()),axis=1)
            # Concatenate
            base = pd.concat((base,Z),axis=1)
            ###### Define weights of categories
            weights = pd.Series(name="weight").astype("float")
            for col in cols:
                data = pd.get_dummies(X[col])
                m_k = (data.mean(axis=0)*var_weights_mfa[grp][col])/model[grp].eig_.iloc[0,0]
                weights = pd.concat((weights,m_k),axis=0)
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        ####### Update number of components
        if self.n_components is None:
            n_components = base.shape[1] - X.shape[1]
        else:
            n_components = min(self.n_components,base.shape[1] - X.shape[1])
        
        # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "var_weights" : var_weights,
                      "group" : group_active_dict,
                      "group_name" : group_name}
        
        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        # Add original data to full base and global PCA without supplementary element
        D = pd.concat((base,X),axis=1)
        index = [D.columns.tolist().index(x) for x in D.columns.tolist() if x not in base.columns.tolist()]
        global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(D)
        quali_var = global_pca.quali_sup_.copy()
        quali_var["contrib"] = global_pca.var_["contrib"]

        #### Add supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup = X_ind_sup.astype("float")
            # Create a copy
            Z_ind_sup = pd.concat((base,X_ind_sup),axis=0)
            # Apply PCA
            global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),ind_sup=ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
            self.ind_sup_ = global_pca.ind_sup_.copy()

        #######################################""
        ###### Add supplementary group
        ################################################################################################################################""
        if self.num_group_sup is not None:
            X_sup_quanti = X_group_sup.select_dtypes(exclude=["object","category"])
            X_sup_quali = X_group_sup.select_dtypes(include=["object","category"])
            if X_sup_quanti.shape[1]>0:
                ##################################################################################################"
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
                summary_quanti_sup.insert(0,"group",group_name.index(grp))
                
                ####### Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
                ### Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns.tolist()]
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            if X_sup_quali.shape[1]>1:
                # Concatenate
                Z_quali_sup = pd.concat((base,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_.copy()
                summary_quali_var_sup.insert(0,"group",group_name.index(grp))
                
                # Append 
                self.summary_quanti_ = pd.concat((self.summary_quali_,summary_quali_var_sup),axis=0,ignore_index=True)

        ##########################################
        self.global_pca_ = global_pca

        ############################################# Removing duplicate value in cumulative percent #######################"
        cumulative = sorted(list(set(global_pca.eig_.iloc[:,3])))

        ##################################################################################################################
        #   Eigenvalues
        ##################################################################################################################
        self.eig_ = global_pca.eig_.iloc[:len(cumulative),:]

        ####### Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:len(cumulative)],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################
        ind = global_pca.ind_.copy()

        ########################################################################################################### 
        # Partiel coordinates for individuals
        ###########################################################################################################
        ##### Add individuals partiels coordinaates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
            # Partial coordinates
            coord_partial = mapply(dummies.dot(quali_var["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),axis=0,progressbar=False,n_workers=n_workers)
            coord_partial = len(list(group_active_dict.keys()))*mapply(coord_partial,lambda x : x/self.eig_.iloc[:,0].values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
            coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
            ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
        # Assign
        ind["coord_partiel"] = ind_coord_partiel
        
        ###########################################################################################################
        #   Partiel coordinates for qualitatives columns
        ###########################################################################################################
        quali_var_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            ############################################################################################################################
            # Compute categories coordinates
            quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X[col]),axis=1).groupby(col).mean()for col in X.columns.tolist()),axis=0)
            quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_coord_partiel.columns.tolist()])
            quali_var_coord_partiel = pd.concat([quali_var_coord_partiel,quali_coord_partiel],axis=1)
                
        quali_var["coord_partiel"] = quali_var_coord_partiel
        # Store informations
        self.quali_var_  = quali_var

        ##########################################################################################################
        #   Partiel coordinates for supplementary qualitatives columns
        ###########################################################################################################
        if self.num_group_sup is not None:
            quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp_sup, cols_sup in group_sup_dict.items():
                if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        ############################################################################################################################
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup[col]),axis=1).groupby(col).mean()for col in cols_sup),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
            self.quali_var_sup_["coord_partiel"] = quali_var_sup_coord_partiel
        
        ################################################################################################"
        #    Inertia Ratios
        ################################################################################################
        #### "Between" inertia on axis s
        between_inertia = len(list(group_active_dict.keys()))*mapply(ind["coord"],lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        between_inertia.name = "between_inertia"

        ### Total inertial on axis s
        total_inertia = pd.Series(name="total_inertia").astype("float")
        for dim in ind["coord"].columns.tolist():
            value = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum().sum()
            inertia = pd.Series([value],index=[dim],name="total_inertia")
            total_inertia = pd.concat((total_inertia,inertia),axis=0)

        ### Inertia ratio
        inertia_ratio = between_inertia/total_inertia
        inertia_ratio.name = "inertia_ratio"
        self.inertia_ratio_ = inertia_ratio

        ##############################################################################################################
        #   Individuals Within inertia
        ##############################################################################################################
        ############################### Within inertia ################################################################
        ind_within_inertia = pd.DataFrame(index=X.index.tolist(),columns=ind["coord"].columns.tolist()).astype("float")
        for dim in ind["coord"].columns.tolist():
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        ######################################## Within partial inertia ################################################
        data = pd.DataFrame().astype("float")
        for dim in ind["coord"].columns.tolist():
            data1 = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers)
            data1 = 100*data1/data1.sum().sum()
            data = pd.concat([data,data1],axis=1)

        ######## Rorder inertia by group
        ind_within_partial_inertia = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            partial_inertia = data[grp]
            partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
            ind_within_partial_inertia = pd.concat([ind_within_partial_inertia,partial_inertia],axis=1)
        ind["within_partial_inertia"] = ind_within_partial_inertia

        #################
        self.ind_ = ind

        ##################################################################################################
        #   Partial axes informations
        #################################################################################################
        ########################################### Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            ######### Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        ############################################## Partial axes cos2
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        #########" Partial correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        #### Add 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
                all_coord = pd.concat([all_coord,data],axis=1)
            # Reorder
            all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
        
        #################################### Partial axes contrib ################################################"
        axes_contrib = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
            eig = self.separate_analyses_[grp].eig_.iloc[:nbcol,0].values/self.separate_analyses_[grp].eig_.iloc[0,0]
            contrib = mapply(partial_axes_coord[grp].iloc[:,:nbcol],lambda x : (x**2)*eig,axis=1,progressbar=False,n_workers=n_workers)
            contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
            axes_contrib  = pd.concat([axes_contrib,contrib],axis=1)
        
        partial_axes_contrib = mapply(axes_contrib,lambda x : 100*x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)

        #### Add a null dataframe
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
                contrib = pd.DataFrame(np.zeros(shape=(n_components,nbcol)),index=["Dim."+str(x+1) for x in range(n_components)],columns=["Dim."+str(x+1) for x in range(nbcol)])
                contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
                partial_axes_contrib = pd.concat([partial_axes_contrib,contrib],axis=1)
            ## Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])
                
        ###############
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : all_coord.corr()}

        #################################################################################################################
        # Group informations : coord
        #################################################################################################################
        # group coordinates
        group_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = pd.concat((function_eta2(X=X[cols],lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in cols),axis=0)
            coord = data.sum(axis=0)/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0])
            coord  = pd.DataFrame(coord.values.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_coord = pd.concat((group_coord,coord),axis=0)
        
        ########################################### Group contributions ############################################
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)

        ######################################## group cos2 ################################################################
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_dist2.loc[grp]).to_frame(grp).T for grp in group_coord.index.tolist()),axis=0)

        ########################################### Group correlations ###############################################
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=self.ind_["coord"],w=None)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_correlation = pd.concat((group_correlation,correl),axis=0)

        #################################################################################################################
        # Measuring how similar groups
        #################################################################################################################
        Lg = pd.DataFrame().astype("float")
        for grp1,cols1 in group_active_dict.items():
            for grp2,cols2 in group_active_dict .items():
                # Sum of chi-squared
                sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X[col1],X[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                Lg.loc[grp1,grp2] = weighted_chi2
        
        # Reorder using active group name
        Lg = Lg.loc[list(group_active_dict.keys()),list(group_active_dict.keys())]

        if self.num_group_sup is not None:
            Lg_sup = pd.DataFrame().astype("float")
            for grp1, cols1 in group_sup_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    if (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square coefficient of correlation
                        sum_corr2 = np.array([(weightedcorrcoef(x=X_group_sup[col1],y=X_group_sup[col2],w=None)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using the eigenvalues of each group
                        weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                        Lg_sup.loc[grp1,grp2] = weighted_corr2
                    elif ((pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X_group_sup[col1],X_group_sup[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg_sup.loc[grp1,grp2] = weighted_chi2
                    elif (all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col1],X_group_sup[col2],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(len(cols1)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
                    elif (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col2],X_group_sup[col1],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
            
            ####### Concatenate
            Lg = pd.concat((Lg,Lg_sup),axis=0)
            # Fill NA with 0.0
            Lg = Lg.fillna(0.0)

            ####
            for grp1,cols1 in group_active_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    X1, X2 = X[cols1], X_group_sup[cols2]
                    if all(pd.api.types.is_string_dtype(X2[col]) for col in cols2):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg.loc[grp1,grp2] = weighted_chi2
                        Lg.loc[grp2,grp1] = weighted_chi2
                    elif all(pd.api.types.is_numeric_dtype(X2[col]) for col in cols2):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols1)))*sum_eta2
                        Lg.loc[grp1,grp2] = weighted_eta2
                        Lg.loc[grp2,grp1] = weighted_eta2
            
            ##################
            Lg = Lg.loc[group_name,group_name]
        
        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        ## RV Coefficient
        RV = pd.DataFrame().astype("float")
        for grp1 in Lg.index:
            for grp2 in Lg.columns:
                RV.loc[grp1,grp2] = Lg.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1])*np.sqrt(Lg.loc[grp2,grp2]))
        
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}

        ##### Add supplementary elements
        if self.num_group_sup is not None:
            group_sup_coord = pd.DataFrame().astype("float")
            for grp, cols in group_sup_dict.items():
                Xg = X_group_sup[cols]
                if all(pd.api.types.is_numeric_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    correl = np.sum((weightedcorrcoef(self.separate_analyses_[grp].call_["Z"],self.ind_["coord"],w=None)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/self.separate_analyses_[grp].eig_.iloc[0,0]
                    coord = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns = ["Dim."+str(x+1) for x in range(len(correl))])
                    group_sup_coord = pd.concat((group_sup_coord,coord),axis=0)
                elif all(pd.api.types.is_string_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    data = self.quali_var_sup_["eta2"].loc[cols,:]
                    coord = (data.sum(axis=0)/(Xg.shape[1]*self.separate_analyses_[grp].eig_.iloc[0,0]))
                    group_sup_coord = pd.concat((group_sup_coord,coord.to_frame(grp).T),axis=0)
                else:
                    raise TypeError("Error : All columns should have the same type.")
            
            #################################### group sup cos2 ###########################################################
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index.tolist()),axis=0)
            
            # Append two dictionnaries
            self.group_ = {**self.group_,**{"coord_sup" : group_sup_coord, "dist2_sup" : group_sup_dist2,"cos2_sup" : group_sup_cos2}}
            
        # Model name
        self.model_ = "mfaqual"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters:
        ----------
        X : pandas/polars DataFrame of shape (n_rows_,n_cols_)

        y : None
            y is ignored
        
        Return
        ------
        X_new : pandas dataframe of shape (n_rows, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ----------------------------------------

        X is projected on the first axes previous extracted from a
        training set

        Parameters
        ----------
        X : pandas/polars DataFrame of shape (n_rows_sup, n_cols_)
            New data, where n_rows_sup is the number of supplementary
            row points and n_cols_ is the number of columns.
            X rows correspond to supplementary row points that are projected
            on the axes.
        
        y : None
            y is ignored
        
        Return
        ------
        X_new : array of float, shape (n_row_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        # Check if X is an instance of pd.DataFrame class
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("Error : DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Revaluate categoricals variables
        X = revaluate_cat_variable(X=X)
        
        #### Apply transition relations
        row_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],self.quali_var_["coord"].shape[1])),index=X.index.tolist(),columns=self.quali_var_["coord"].columns.tolist())
        for grp, cols in self.call_["group"].items():
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
            # Apply
            coord = mapply(dummies.dot(self.quali_var_["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),axis=0,progressbar=False,n_workers=n_workers)
            row_coord = row_coord + coord
        # Weighted by the eigenvalue
        ind_coord = mapply(row_coord ,lambda x : x/self.eig_.iloc[:,0][:self.call_["n_components"]],axis=1,progressbar=False,n_workers=n_workers)
        return ind_coord

#####################################################################################################################
#   MULTIPLE FACTOR ANALYSIS FOR MIXED DATA (MIXED GROUP)
#####################################################################################################################
class MFAMIX(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis for Mixed Data (MFAMIX)
    ------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Factor Analysis for Mixed Data in the sense of Pagès J. (2002) with supplementary individuals
    and supplementary groups of variables. Groups of variables can be quantitative, categorical.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    group : a list or a tuple with the number of variables in each group

    name_group : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)

    group_type : the type of variables in each group; three possibilities : 
                    - "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
                    - "n" for categorical variables
                    - "m" for mixed variables (quantitative and qualitative variables)
    
    num_group_sup : the indexes of the illustrative groups (by default, None and no group re illustrative)

    ind_sup : an integer, a list or a tuple of the supplementary individuals

    ind_weights : an optional individuals weights (by default, a list of 1/(number of individuals) for uniform weights), the weights 
                    are given only for the active individuals
    
    quanti_var_weights_mfa : an optional quantitatives variables weights (by default, a list of 1 for uniform weights), the weights
                                are given only for active quantitatives variables
    
    quali_var_weights_mfa : an optional qualitatives variables weights (by defaut, a list of 1/(number of categoricals variables in the group)),
                            the weights are given only for qualitatives variables
    
    Parallelize : bool, default = False. Adding multi-core methods to PandasObject.

    Return
    ------
    summary_quali_ : a summary of the results for the categorical variables

    summary_quanti_ : a summary of the results for the quantitative variables

    separate_analyses_ : the results for the separate analyses

    svd_ : a dictionary of matrices containing all the results of the singular value decomposition

    eig_ : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the
            cumulative percentge of variance
    
    group_ : a dictionary of pandas dataframe containing all the results for the groups (Lg and RV coefficients, coordinates, square cosine,
                contributions, distance to the origin, the correlations between each group and each factor)
    
    inertia_ratio_ : inertia ratio

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,
            contributions)
    
    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates,
                square cosine)
    
    quanti_var_ : a dictionary of pandas dataframe containing all the results for the quantitatives variables (coordinates,
                    correlation between variables and axes, contribution, cos2)
    
    quali_var_ : a dictionary of pandas dataframe containing all the results for the categorical variables (coordinates of each categories
                    of each variables, contribution and vtest which is a criterion with a normal distribution)
    
    quanti_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates,
                        correlation between variables and axes, cos2)
    
    quali_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of 
                        each categories of each variables, cos2 and vtest which is a criterion with a normal distribution)
    
    partial_axes_ : a dictionary of pandas dataframe containing all the results for the partial axes (coordinates, correlation between variables
                        and axes, correlation between partial axes)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfamix'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    """
    def __init__(self,
                n_components=5,
                group = None,
                name_group = None,
                group_type = None,
                num_group_sup = None,
                ind_sup = None,
                ind_weights=None,
                quanti_var_weights_mfa = None,
                quali_var_weights_mfa = None,
                parallelize = False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.group_type = group_type
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.quanti_var_weights_mfa = quanti_var_weights_mfa
        self.quali_var_weights_mfa = quali_var_weights_mfa
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : pandas DataFrame, shape (n_rows, n_columns)

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
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
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
        
        # Drop mixed group
        if "m" in self.group_type:
            raise TypeError("Error: 'group_type' should be one of 'c', 's', 'n'")
        
        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")

        ###### Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns.tolist():
            X[col] = X[col].astype("object")
        
        ########################################################################################################################
        #   check if two categoricals variables have same categories
        ######################################################################################################################
        X = revaluate_cat_variable(X)
        
        #########################################################################################################################
        #   Check if group is None
        #########################################################################################################################
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("Error : 'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]
        
        #######################################################################################################################
        # Remove supplementary group
        if self.num_group_sup is not None:
            # Set default values to None
            self.quali_var_sup_ = None
            self.quanti_var_sup_ = None
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        ##################################################################
        # Check if supplementary individuals
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        #########################################################################################################################
        #   Check if group type in not None
        #########################################################################################################################
        if self.group_type is None:
            raise ValueError("Error : 'group_type' must be assigned.")
        
        #######################################################################################################################
        if len(self.group) != len(self.group_type):
            raise TypeError("Error : Not convenient group definition")

        ############################################################################################################################
        #  Assigned group name
        ###########################################################################################################################
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("Error : 'group_name' must be a list or a tuple of group name")
        else:
            group_name = [x for x in self.name_group]
        
        ##############################################################################################################################
        # check if group name is an integer
        #############################################################################################################################
        for i in range(len(group_name)):
            if isinstance(group_name[i],int) or isinstance(group_name[i],float):
                group_name[i] = "Gr"+str(i+1)
        
        ##############################################################################################################################
        #   Assigned group name to label
        #############################################################################################################################
        group_active_dict = {}
        group_sup_dict = {}
        debut = 0
        for i in range(len(nb_elt_group)):
            X_group = X.iloc[:,(debut):(debut+nb_elt_group[i])]
            if self.num_group_sup is not None:
                if i in num_group_sup:
                    new_elt = {group_name[i]:X_group.columns.tolist()}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
            debut = debut + nb_elt_group[i]

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

       ######################################## Drop supplementary groups columns #######################################
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))
        
        ######################################## Drop supplementary individuals  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            # Drop supplementary individuals
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"Error : {grp} group should have at least two columns")

        # Extract qualitatives and quantitatives groups
        all_nums = {}
        all_cats = {}
        for grp, cols in group_active_dict.items():
            all_nums[grp] = all(pd.api.types.is_numeric_dtype(X[col]) for col in cols)
            all_cats[grp]= all(pd.api.types.is_string_dtype(X[col]) for col in cols)
        
        ########################################## Summary qualitatives variables ###############################################
        # Compute statisiques
        summary_quanti = pd.DataFrame()
        summary_quali = pd.DataFrame()
        for grp, cols in group_active_dict.items():
            if all_cats[grp]:
                for col in cols:
                    eff = X[col].value_counts().to_frame("effectif").reset_index().rename(columns={"index" : "modalite"})
                    eff.insert(0,"variable",col)
                    eff.insert(0,"group",group_name.index(grp))
                    summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
            elif all_nums[grp]:
                summary = X[cols].describe().T.reset_index().rename(columns={"index" : "variable"})
                summary.insert(0,"group",group_name.index(grp))
                summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)
            else:
                # Split X into two group
                X_quali = X[cols].select_dtypes(include=["object"])
                X_quanti = X[cols].select_dtypes(exclude=["object"])

                ###### Summary for qualitatives variables
                for col in X_quali.columns.tolist():
                    eff = X_quali[col].value_counts().to_frame("effectif").reset_index().rename(columns={"index" : "modalite"})
                    eff.insert(0,"variable",col)
                    eff.insert(0,"group",group_name.index(grp))
                    summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
                
                ####### Summary of quantitatives variables
                summary = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary.insert(0,"group",group_name.index(grp))
                summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)
                
        # Convert effectif and count to int
        summary_quali["effectif"] = summary_quali["effectif"].astype("int")
        summary_quanti["count"] = summary_quanti["count"].astype("int")

        ### Store summary
        self.summary_quanti_ = summary_quanti
        self.summary_quali_ = summary_quali
        
        ###############################################################################################################################
        #   Set weights : Individuals weights, quantitatives variables weights, qualitatives variables weights
        ###############################################################################################################################

        ########### Set indiviuduals weights
        if self.ind_weights is None:
            ind_weights = (np.ones(X.shape[0])/X.shape[0]).tolist()
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of individuals weights")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = [x/np.sum(self.ind_weights) for x in self.ind_weights]
        
        ########## Set quantitatives variables weights
        quanti_var_weights_mfa = {}
        if self.quanti_var_weights_mfa is None:
            for grp, cols in group_active_dict.items():
                if all_nums[grp]:
                    quanti_var_weights_mfa[grp] = np.ones(len(cols)).tolist()
        elif not isinstance(self.quanti_var_weights_mfa,dict):
            raise ValueError("Error : 'quanti_var_weights_mfa' must be a dictionary where keys are quantitatives groups names and values are list of quantitatives variables weights in group.")
        else:
            for grp, cols in group_active_dict.items():
                if all_nums[grp]:
                    quanti_var_weights_mfa[grp] = np.array(self.quanti_var_weights_mfa[grp]).tolist()

        ########### Set qualitatives variables weights
        quali_var_weights_mfa = {}
        if self.quali_var_weights_mfa is None:
            for grp, cols in group_active_dict.items():
                if all_cats[grp]:
                    var_weights = pd.Series(index=cols,name="weights").astype("float")
                    for col in X[cols].columns.tolist():
                        var_weights[col] = 1/X[cols].shape[1]
                    quali_var_weights_mfa[grp] = var_weights
        elif not isinstance(self.quali_var_weights_mfa,dict):
            raise ValueError("Error : 'var_weights_mfa' must be a dictionary where keys are qualitatives groups names and values are pandas series of qualitatives variables weights in group.")
        else:
            for grp, cols in group_active_dict.items():
                if all_cats[grp]:
                    var_weights = pd.Series(index=cols,name="weights").astype("float")
                    for col in X[cols].columns.tolist():
                        var_weights[col] = self.quali_var_weights_mfa[grp][col]/self.quali_var_weights_mfa[grp].values.sum()
                    quali_var_weights_mfa[grp] = var_weights
        
        ############################ Run a Factor Analysis in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if all_nums[grp]:
                if self.group_type[group_name.index(grp)]=="c":
                    # Center Principal Components Anlysis (PCA) 
                    fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,var_weights=quanti_var_weights_mfa[grp],ind_sup=None,parallelize=self.parallelize)
                elif self.group_type[group_name.index(grp)]=="s":
                    # Scale Principal Components Anlysis (PCA)
                    fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,var_weights=quanti_var_weights_mfa[grp],ind_sup=None,parallelize=self.parallelize)
                else:
                    raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
            elif all_cats[grp]:
                if self.group_type[group_name.index(grp)]=="n":
                    # Multiple Correspondence Analysis (MCA)
                    fa = MCA(n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize,benzecri=False,greenacre=False)
                else:
                    raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
            else:
                if self.group_type[group_name.index(grp)]=="m":
                    # Factor Analysis of Mixed Data (FAMD)
                    fa = FAMD(n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                else:
                    raise TypeError("Error : for mixed group 'group_type' should be 'm'")
            model[grp] = fa.fit(X[cols])

            ########################## Add supp
            if self.ind_sup is not None:
                # Select 
                X_ind_sup = X_ind_sup.astype("float")
                if all_nums[grp]:
                    if self.group_type[group_name.index(grp)]=="c":
                        # Center Principal Components Analysis (PCA)
                        fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,var_weights=quanti_var_weights_mfa[grp],ind_sup=ind_sup,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        # Scale Principal Components Analysis (PCA)
                        fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,var_weights=quanti_var_weights_mfa[grp],ind_sup=ind_sup,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
                elif all_cats[grp]:
                    if self.group_type[group_name.index(grp)]=="n":
                        # Multiple Correspondence Analysis (MCA)
                        fa = MCA(n_components=None,ind_weights=ind_weights,ind_sup=ind_sup,parallelize=self.parallelize,benzecri=False,greenacre=False)
                    else:
                        raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        # Factor Analysis of Mixed Data (FAMD)
                        fa = FAMD(n_components=None,ind_weights=ind_weights,ind_sup=ind_sup,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for mixed group 'group_type' should be 'm'")
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        ############################################### Separate  Factor Analysis for supplementary groups ######################################""
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            ####### Find columns for supplementary group
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            # Factor Analysis
            for grp, cols in group_sup_dict.items():
                # Instance the FA model
                if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
                elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        fa = MCA(n_components=None,parallelize=self.parallelize,benzecri=False,greenacre=False)
                    else:
                        raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        fa = FAMD(n_components=None,ind_weights=ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for mixed group 'group_type' should be 'm'")
                    
                # Fit the model
                model[grp] = fa.fit(X_group_sup[cols])
        
        ##################### Compute group disto
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="dist2")

        ##### Compute group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="dist2")

        ##### Store separate analysis
        self.separate_analyses_ = model

        ################################################# Standardize Data ##########################################################
        means = {}
        std = {}
        base        = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        for grp, cols in group_active_dict.items():
            if all_nums[grp]:
                ############################### Compute Mean and Standard deviation #################################
                d1 = DescrStatsW(X[cols],weights=ind_weights,ddof=0)
                ########################### Standardize #################################################################################
                Z = (X[cols] - d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)
                ###################" Concatenate
                base = pd.concat([base,Z],axis=1)
                ##################################"
                means[grp] = d1.mean.reshape(1,-1)
                std[grp] = d1.std.reshape(1,-1)
                ################################ variables weights
                weights = pd.Series(np.repeat(a=1/model[grp].eig_.iloc[0,0],repeats=len(cols)),index=cols)
                # Ajout de la pondération de la variable
                weights = weights*np.array(quanti_var_weights_mfa[grp])
                var_weights = pd.concat((var_weights,weights),axis=0)
            elif all_cats[grp]:
                # Compute dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
                # Effectif par categories
                I_k = dummies.sum(axis=0)
                # Apply standardize
                Z = pd.concat((dummies.loc[:,[k]]*(X.shape[0]/I_k[k])-1 for k  in dummies.columns.tolist()),axis=1)
                # Concatenate
                base = pd.concat([base,Z],axis=1)
                ###### Define weights of categories
                weights = pd.Series(name="weight").astype("float")
                for col in X[cols].columns.tolist():
                    data = pd.get_dummies(X[cols][col])   
                    m_k = (data.mean(axis=0)*quali_var_weights_mfa[grp][col])/model[grp].eig_.iloc[0,0]
                    weights = pd.concat([weights,m_k],axis=0)
                var_weights = pd.concat([var_weights,weights],axis=0)
            else:
                ########################### Split X into two group
                X_quali = X[cols].select_dtypes(include=["object"])
                X_quanti = X[cols].select_dtypes(exclude=["object"])

                ##############################################################################################################
                #   Apply weighted for quantitatives variables
                #############################################################################################################
                # Compute Mean and Standard deviation
                d1 = DescrStatsW(X_quanti,weights=ind_weights,ddof=0)
                # Standardize
                Z1 = (X_quanti - d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)
                # Assign means and stand
                means[grp] = d1.mean.reshape(1,-1)
                std[grp] = d1.std.reshape(1,-1)
                ################################ variables weights
                weights1 = pd.Series(np.repeat(a=1/model[grp].eig_.iloc[0,0],repeats=len(X_quanti.columns)),index=X_quanti.columns.tolist())
               
                ##################################################################################################################3
                # Apply weighted for categoricals variables
                ####################################################################################################################
                # Compute dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X_quali[col]) for col in X_quali.columns.tolist()),axis=1)
                # Effectif par categories
                I_k = dummies.sum(axis=0)
                # Apply standardize
                Z2 = pd.concat((dummies.loc[:,[k]]*(X.shape[0]/I_k[k])-1 for k  in dummies.columns.tolist()),axis=1)
                # categories weights
                weights2 = dummies.mean(axis=0)*(1/X_quali.shape[1])
                ######################################################################################################################
                # Concatenate base
                Z = pd.concat((Z1,Z2),axis=1)
                base = pd.concat([base,Z],axis=1)
                # Concatenate variables weights
                weights = pd.concat((weights1,weights2),axis=0)
                var_weights = pd.concat((var_weights,weights),axis=0)

        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        # Add original data to full base and global PCA without supplementary element
        D = base.copy()
        for col in X.columns.tolist():
            if X[col].dtype in ["object"]:
                D = pd.concat((D,X[col]),axis=1)
        index = [D.columns.tolist().index(x) for x in D.columns.tolist() if x not in base.columns.tolist()]
        global_pca = PCA(standardize = False,n_components = None,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(D)

        if self.ind_sup is not None:
            ########
            D = pd.concat((D,X_ind_sup),axsi=0)

        ############################################# Removing duplicate value in cumulative percent #######################"
        cumulative = sorted(list(set(global_pca.eig_.iloc[:,3])))

        # Number of components
        if self.n_components is None:
            n_components = len(cumulative)
        else:
            n_components = min(self.n_components,len(cumulative))
        
         # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "var_weights" : var_weights,
                      "means" : means,
                      "std" : std,
                      "group" : group_active_dict,
                      "group_name" : group_name}

        # Store global PCA

        ## Extract actives qualitatives informations
        quali_var = {"coord" : global_pca.quali_sup_["coord"].iloc[:,:n_components],"cos2" : global_pca.quali_sup_["cos2"].iloc[:,:n_components],"dist":global_pca.quali_sup_["dist"],
                     "vtest" : global_pca.quali_sup_["vtest"].iloc[:,:n_components],"eta2" : global_pca.quali_sup_["eta2"].iloc[:,:n_components]}
        

        #### Add supplementary individuals
        # if self.ind_sup is not None:
        #     X_ind_sup = X_ind_sup.astype("float")
        #     # Create a copy
        #     Z_ind_sup = base.copy()
        #     for grp, cols in group.items():
        #         Z_ind_sup = pd.concat((Z_ind_sup,X_ind_sup[grp]),axis=0)
        #     # Apply PCA
        #     global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),ind_sup=ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
        #     self.ind_sup_ = global_pca.ind_sup_.copy()

        ###### Add supplementary group
        if self.num_group_sup is not None:
            X_sup_quanti = X_group_sup.select_dtypes(exclude=["object","category"])
            X_sup_quali = X_group_sup.select_dtypes(include=["object","category"])
            if X_sup_quanti.shape[1]>0:
                ##################################################################################################"
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
                summary_quanti_sup.insert(0,"group",group_name.index(grp))
                self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

                ####### Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti- d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
                ### Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns.tolist()]
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            if X_sup_quali.shape[1]>1:
                # Concatenate
                Z_quali_sup = pd.concat((base,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_.copy()
                summary_quali_var_sup.insert(0,"group",group_name.index(grp))
                
                # Append 
                self.summary_quali_ = pd.concat((self.summary_quali_,summary_quali_var_sup),axis=0,ignore_index=True)

        ########################################## Store global PCA
        self.global_pca_ = global_pca

        ##################################################################################################################
        #   Eigenvalues
        ##################################################################################################################
        self.eig_ = global_pca.eig_.iloc[:len(cumulative),:]

        ####### Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:len(cumulative)],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################
        ind = {"coord" : global_pca.ind_["coord"].iloc[:,:n_components],"contrib" : global_pca.ind_["contrib"].iloc[:,:n_components],"cos2" : global_pca.ind_["cos2"].iloc[:,:n_components],"infos":global_pca.ind_["infos"]}

        ####################################################################################################
        #   Quantitatiaves variables informations : coordinates, cos2 and contrib
        ####################################################################################################
        # Correlation between variables en axis
        quanti_var_coord = pd.DataFrame().astype("float")
        quanti_var_contrib = pd.DataFrame().astype("float")
        quanti_var_cos2 = pd.DataFrame().astype("float")
        # qualitative contrib
        quali_var_contrib =  pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            if all_nums[grp]:
                ########## Correlation between variables en axis
                quanti_coord = weightedcorrcoef(x=X[cols],y=ind["coord"],w=None)[:X[cols].shape[1],X[cols].shape[1]:]
                quanti_coord = pd.DataFrame(quanti_coord,index=cols,columns=["Dim."+str(x+1) for x in range(quanti_coord.shape[1])])
                quanti_var_coord = pd.concat([quanti_var_coord,quanti_coord],axis=0)

                ########### Contributions
                quanti_contrib = global_pca.var_["contrib"].loc[cols,:].copy()
                quanti_var_contrib = pd.concat([quanti_var_contrib,quanti_contrib],axis=0)

                ########### Cos2
                quanti_cos2 = global_pca.var_["cos2"].loc[cols,:].copy()
                quanti_var_cos2 = pd.concat([quanti_var_cos2,quanti_cos2],axis=0)
            elif all_cats[grp]:
                modalite = []
                for col in cols:
                    modalite = modalite + np.unique(X[cols][col]).tolist()
                ########### contribution
                quali_contrib = global_pca.var_["contrib"].loc[modalite,:].copy()
                quali_var_contrib = pd.concat([quali_var_contrib,quali_contrib],axis=0)
            else:
                #########################################################################################################
                X_quanti = X[cols].select_dtypes(exclude=["object"])

                ########## Correlation between variables en axis
                quanti_coord = weightedcorrcoef(x=X_quanti,y=ind["coord"],w=None)[:X_quanti.shape[1],X_quanti.shape[1]:]
                quanti_coord = pd.DataFrame(quanti_coord,index=X_quanti.columns.tolist(),columns=["Dim."+str(x+1) for x in range(quanti_coord.shape[1])])
                quanti_var_coord = pd.concat([quanti_var_coord,quanti_coord],axis=0)

                ########### Contributions
                quanti_contrib = global_pca.var_["contrib"].loc[X_quanti.columns.tolist(),:].copy()
                quanti_var_contrib = pd.concat([quanti_var_contrib,quanti_contrib],axis=0)

                ########### Cos2
                quanti_cos2 = global_pca.var_["cos2"].loc[X_quanti.columns.tolist(),:].copy()
                quanti_var_cos2 = pd.concat([quanti_var_cos2,quanti_cos2],axis=0)

                #########################################################################################################
                X_quali = X[cols].select_dtypes(include=["object"])

                modalite = []
                for col in X_quali.columns.tolist():
                    modalite = modalite + np.unique(X_quali[col]).tolist()
                ########### contribution
                quali_contrib = global_pca.var_["contrib"].loc[modalite,:].copy()
                quali_var_contrib = pd.concat([quali_var_contrib,quali_contrib],axis=0)

        
        ### Store all informations
        self.quanti_var_ = {"coord" : quanti_var_coord,"cor" : quanti_var_coord,"contrib":quanti_var_contrib,"cos2":quanti_var_cos2}

        ##### Set qualitatives contributions
        quali_var["contrib"] = quali_var_contrib

        ########################################################################################################### 
        # Partiel coordinates for individuals
        ###########################################################################################################
        ##### Add individuals partiels coordinaates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            if all_nums[grp]:
                # Standardisze data
                Z = (X[cols] - means[grp])/std[grp]
                # Partial coordinates
                coord_partial = mapply(Z.dot(quanti_var_coord.loc[cols,:]),lambda x : x/self.separate_analyses_[grp].eig_.iloc[0,0],axis=0,progressbar=False,n_workers=n_workers)
                coord_partial = len(nb_elt_group)*mapply(coord_partial,lambda x : x/np.sqrt(self.eig_.iloc[:,0].values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
                coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
            elif all_cats[grp]:
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X[cols][col]) for col in cols),axis=1)
                # Partial coordinates
                coord_partial = mapply(dummies.dot(quali_var["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),axis=0,progressbar=False,n_workers=n_workers)
                coord_partial = len(nb_elt_group)*mapply(coord_partial,lambda x : x/self.eig_.iloc[:,0].values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
            else:
                #########################################################################################################
                # Apply weights for quantitatives variables
                ##########################################################################################################
                X_quanti = X[cols].select_dtypes(exclude=["object"])
                # Standardisze data
                Z = (X_quanti - means[grp])/std[grp]
                # Partial coordinates
                coord_partial1 = mapply(Z.dot(quanti_var_coord.loc[X_quanti.columns.tolist(),:]),lambda x : x/self.separate_analyses_[grp].eig_.iloc[0,0],
                                        axis=0,progressbar=False,n_workers=n_workers)

                #########################################################################################################
                #   Apply weights for categoricals variables
                #########################################################################################################
                X_quali = X[cols].select_dtypes(include=["object"])
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X_quali[col]) for col in X_quali.columns.tolist()),axis=1)
                coord_partial2 = mapply(dummies.dot(quali_var["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(X_quali.shape[1]*self.separate_analyses_[grp].eig_.iloc[0,0]),
                                       axis=0,progressbar=False,n_workers=n_workers)
                # Add the two
                coord_partial = coord_partial1 + coord_partial2
                coord_partial = len(nb_elt_group)*mapply(coord_partial,lambda x : x/self.eig_.iloc[:,0].values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)

        # Assign
        ind["coord_partiel"] = ind_coord_partiel

        ###########################################################################################################
        #   Partiel coordinates for qualitatives columns
        ###########################################################################################################
        quali_var_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            quali_grp_coord_partiel = pd.DataFrame()
            if all_cats[grp]:
                ############################################################################################################################
                # Compute categories coordinates
                for grp2, cols2 in group_active_dict.items():
                    quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp2],X[col]),axis=1).groupby(col).mean()for col in cols),axis=0)
                    quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp2,col) for col in quali_coord_partiel.columns.tolist()])
                    # cbind.dataframe
                    quali_grp_coord_partiel = pd.concat([quali_grp_coord_partiel,quali_coord_partiel],axis=1)
            elif any(pd.api.types.is_string_dtype(X[col]) for col in cols):
                X_quali = X[cols].select_dtypes(include=["object"])
                ###########################################################################################################################
                # Compute categories coordinates
                for grp2, cols2 in group_active_dict.items():
                    quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp2],X_quali[col]),axis=1).groupby(col).mean()for col in X_quali.columns.tolist()),axis=0)
                    quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp2,col) for col in quali_coord_partiel.columns.tolist()])
                    # cbind.dataframe
                    quali_grp_coord_partiel = pd.concat([quali_grp_coord_partiel,quali_coord_partiel],axis=1)
            # rbind.dataframe
            quali_var_coord_partiel = pd.concat((quali_var_coord_partiel,quali_grp_coord_partiel),axis=0)
                    
        quali_var["coord_partiel"] = quali_var_coord_partiel
        # Store informations
        self.quali_var_  = quali_var

        ##########################################################################################################
        #   Partiel coordinates for supplementary qualitatives columns
        ###########################################################################################################
        if self.num_group_sup is not None:
            quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp_sup, cols_sup in group_sup_dict.items():
                if all(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        ############################################################################################################################
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup[col]),axis=1).groupby(col).mean()for col in cols_sup),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
                elif any(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        ############################################################################################################################
                        X_group_sup_quali = X_group_sup[cols_sup].select_dtypes(include=['object'])
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup_quali[col]),axis=1).groupby(col).mean()for col in X_group_sup_quali.columns.tolist()),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
            self.quali_var_sup_["coord_partiel"] = quali_var_sup_coord_partiel
        
        ################################################################################################"
        #    Inertia Ratios
        ################################################################################################
        #### "Between" inertia on axis s
        between_inertia = len(nb_elt_group)*mapply(ind["coord"],lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        between_inertia.name = "between_inertia"

        ### Total inertial on axis s
        total_inertia = pd.Series(name="total_inertia").astype("float")
        for dim in ind["coord"].columns.tolist():
            value = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum().sum()
            inertia = pd.Series([value],index=[dim],name="total_inertia")
            total_inertia = pd.concat((total_inertia,inertia),axis=0)

        ### Inertia ratio
        inertia_ratio = between_inertia/total_inertia
        inertia_ratio.name = "inertia_ratio"
        self.inertia_ratio_ = inertia_ratio

        ##############################################################################################################
        #   Individuals Within inertia
        ##############################################################################################################
        ############################### Within inertia ################################################################
        ind_within_inertia = pd.DataFrame(index=X.index.tolist(),columns=ind["coord"].columns.tolist()).astype("float")
        for dim in ind["coord"].columns.tolist():
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        ######################################## Within partial inertia ################################################
        data = pd.DataFrame().astype("float")
        for dim in ind["coord"].columns.tolist():
            data1 = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers)
            data1 = 100*data1/data1.sum().sum()
            data = pd.concat([data,data1],axis=1)

        ######## Rorder inertia by group
        ind_within_partial_inertia = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            partial_inertia = data[grp]
            partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
            ind_within_partial_inertia = pd.concat([ind_within_partial_inertia,partial_inertia],axis=1)
        ind["within_partial_inertia"] = ind_within_partial_inertia

        #################"" Store 
        self.ind_ = ind

        ##################################################################################################
        #   Partial axes informations
        #################################################################################################
        ########################################### Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            ######### Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        ############################################## Partial axes cos2
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        #########" Partial individuals coordinates
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        
        #### Add 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
                all_coord = pd.concat([all_coord,data],axis=1)
            # Reorder
            all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
        
        #################################### Partial axes contrib ################################################"
        axes_contrib = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
            eig = self.separate_analyses_[grp].eig_.iloc[:nbcol,0].values/self.separate_analyses_[grp].eig_.iloc[0,0]
            contrib = mapply(partial_axes_coord[grp].iloc[:,:nbcol],lambda x : (x**2)*eig,axis=1,progressbar=False,n_workers=n_workers)
            contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
            axes_contrib  = pd.concat([axes_contrib,contrib],axis=1)
        
        partial_axes_contrib = mapply(axes_contrib,lambda x : 100*x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)
    
        #### Add a null dataframe
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
                contrib = pd.DataFrame(np.zeros(shape=(n_components,nbcol)),index=["Dim."+str(x+1) for x in range(n_components)],columns=["Dim."+str(x+1) for x in range(nbcol)])
                contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
                partial_axes_contrib = pd.concat([partial_axes_contrib,contrib],axis=1)
            ## Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])
                
        ###############
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : all_coord.corr()}

        #################################################################################################################
        # Group informations : coord
        #################################################################################################################
        # group coordinates
        group_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            if all_cats[grp]:
                data = pd.concat((function_eta2(X=X[cols],lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in cols),axis=0)
                coord = data.sum(axis=0)/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0])
                coord  = pd.DataFrame(coord.values.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
                group_coord = pd.concat((group_coord,coord),axis=0)
            elif all_nums[grp]:
                data = self.separate_analyses_[grp].call_["Z"]
                coord =  (weightedcorrcoef(data,self.ind_["coord"],w=None)[:data.shape[1],data.shape[1]:]**2).sum(axis=0)/self.separate_analyses_[grp].eig_.iloc[0,0]
                coord  = pd.DataFrame(coord.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
                group_coord = pd.concat((group_coord,coord),axis=0)
        
        ########################################### Group contributions ############################################
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)

        ######################################## group cos2 ################################################################
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_dist2.loc[grp]).to_frame(grp).T for grp in group_coord.index.tolist()),axis=0)

        ########################################### Group correlations ###############################################
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=self.ind_["coord"],w=None)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_correlation = pd.concat((group_correlation,correl),axis=0)
        
        #################################################################################################################
        # Measuring how similar groups
        #################################################################################################################
        Lg = pd.DataFrame().astype("float")
        for grp1,cols1 in group_active_dict.items():
            for grp2,cols2 in group_active_dict.items():
                X1, X2 = X[cols1], X[cols2]
                if (all_nums[grp1] and all_nums[grp2]):
                    # Sum of square coefficient of correlation
                    sum_corr2 = np.array([(weightedcorrcoef(x=X1[col1],y=X2[col2],w=ind_weights)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                    # Weighted the sum using the eigenvalues of each group
                    weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                    Lg.loc[grp1,grp2] = weighted_corr2
                elif (all_cats[grp1] and all_cats[grp2]):
                    # Sum of chi-squared
                    sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                    # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                    weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                    Lg.loc[grp1,grp2] = weighted_chi2
                elif (all_nums[grp1] and all_cats[grp2]):
                    # Sum of square correlation ratio
                    sum_eta2 = np.array([eta2(X2[col2],X1[col1],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                    Lg.loc[grp1,grp2] = weighted_eta2
                    Lg.loc[grp2,grp1] = weighted_eta2
                elif (all_cats[grp1] and all_nums[grp2]):
                    # Sum of square correlation ratio
                    sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*len(cols1)*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_eta2
                    Lg.loc[grp1,grp2] = weighted_eta2
                    Lg.loc[grp2,grp1] = weighted_eta2
        
        # Reorder using active group name
        Lg = Lg.loc[list(group_active_dict.keys()),list(group_active_dict.keys())]

        # Add supplementary Lg elements
        if self.num_group_sup is not None:
            Lg_sup = pd.DataFrame().astype("float")
            for grp1, cols1 in group_sup_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    if (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square coefficient of correlation
                        sum_corr2 = np.array([(weightedcorrcoef(x=X_group_sup[col1],y=X_group_sup[col2],w=ind_weights)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using the eigenvalues of each group
                        weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                        Lg_sup.loc[grp1,grp2] = weighted_corr2
                    elif ((pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X_group_sup[col1],X_group_sup[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg_sup.loc[grp1,grp2] = weighted_chi2
                    elif (all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col1],X_group_sup[col2],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(len(cols1)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
                    elif (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col2],X_group_sup[col1],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
            
            ####### Concatenate
            Lg = pd.concat((Lg,Lg_sup),axis=0)
            # Fill NA with 0.0
            Lg = Lg.fillna(0.0)
            
            ####
            for grp1,cols1 in group_active_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    X1, X2 = X[cols1], X_group_sup[cols2]
                    if all_nums[grp1] and all(pd.api.types.is_numeric_dtype(X2[col]) for col in cols2):
                        # Sum of square coefficient of correlation
                        sum_corr2 = np.array([(weightedcorrcoef(x=X1[col1],y=X2[col2],w=ind_weights)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using the eigenvalues of each group
                        weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                        Lg.loc[grp1,grp2] = weighted_corr2
                        Lg.loc[grp2,grp1] = weighted_corr2
                    elif all_nums[grp1] and all(pd.api.types.is_string_dtype(X2[col]) for col in cols2):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X2[col2],X1[col1],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0][0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                        Lg.loc[grp1,grp2] = weighted_eta2
                        Lg.loc[grp2,grp1] = weighted_eta2
                    elif all_cats[grp1] and all(pd.api.types.is_numeric_dtype(X2[col]) for col in cols2):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["correlation ratio"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols1)))*sum_eta2
                        Lg.loc[grp1,grp2] = weighted_eta2
                        Lg.loc[grp2,grp1] = weighted_eta2
                    elif all_cats[grp1] and all(pd.api.types.is_string_dtype(X2[col]) for col in cols2):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg.loc[grp1,grp2] = weighted_chi2
                        Lg.loc[grp2,grp1] = weighted_chi2
            
            ##################
            Lg = Lg.loc[group_name,group_name]

        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        ## RV Coefficient
        RV = pd.DataFrame().astype("float")
        for grp1 in Lg.index:
            for grp2 in Lg.columns:
                RV.loc[grp1,grp2] = Lg.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1])*np.sqrt(Lg.loc[grp2,grp2]))
        
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}

        ##### Add supplementary elements
        if self.num_group_sup is not None:
            group_sup_coord = pd.DataFrame().astype("float")
            for grp, cols in group_sup_dict.items():
                Xg = X_group_sup[cols]
                if all(pd.api.types.is_numeric_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    correl = np.sum((weightedcorrcoef(self.separate_analyses_[grp].call_["Z"],self.ind_["coord"],w=None)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/self.separate_analyses_[grp].eig_.iloc[0,0]
                    coord = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns = ["Dim."+str(x+1) for x in range(len(correl))])
                    group_sup_coord = pd.concat((group_sup_coord,coord),axis=0)
                elif all(pd.api.types.is_string_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    data = self.quali_var_sup_["eta2"].loc[cols,:]
                    coord = (data.sum(axis=0)/(Xg.shape[1]*self.separate_analyses_[grp].eig_.iloc[0,0]))
                    group_sup_coord = pd.concat((group_sup_coord,coord.to_frame(grp).T),axis=0)
                else:
                    raise TypeError("Error : All columns should have the same type.")
            
            #################################### group sup cos2 ###########################################################
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index.tolist()),axis=0)
            
            # Append two dictionnaries
            self.group_ = {**self.group_,**{"coord_sup" : group_sup_coord, "dist2_sup" : group_sup_dist2,"cos2_sup" : group_sup_cos2}}

        # Name of model
        self.model_ = "mfamix"

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit to data, then transform it.

        Parameters:
        ----------
        X : pandas DataFrame of shape (n_rows_,n_cols_)

        y : None
            y is ignored

        """
        self.fit(X)
        return self.ind_["coord"]
    
    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ----------------------------------------

        X is projected on the first axes previous extracted from a
        training set

        Parameters
        ----------
        X : pandas DataFrame of shape (n_rows_sup, n_cols_)
            New data, where n_rows_sup is the number of supplementary
            row points and n_cols_ is the number of columns.
            X rows correspond to supplementary row points that are projected
            on the axes.
        
        y : None
            y is ignored
        
        Return
        ------
        X_new : array of float, shape (n_row_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ############### Apply revaluate function
        X = revaluate_cat_variable(X=X)
        
        # Check New Data has same group
        row_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],self.call_["n_components"])),index=X.index.tolist(),
                                 columns=["Dim."+str(x+1) for x in range(self.call_["n_components"])])
        for grp, cols in self.call_["group"].items():
            num_row_partial = pd.DataFrame(np.zeros(shape=(X.shape[0],self.call_["n_components"])),index=X.index.tolist(),
                                           columns=["Dim."+str(x+1) for x in range(self.call_["n_components"])])
            cat_row_partial = pd.DataFrame(np.zeros(shape=(X.shape[0],self.call_["n_components"])),index=X.index.tolist(),
                                           columns=["Dim."+str(x+1) for x in range(self.call_["n_components"])])
            if all(pd.api.types.is_numeric_dtype(X[col]) for col in cols):
                # Standardize the Data
                Z = (X[cols] - self.call_["means"][grp])/self.call_["std"][grp]
                # Partiel coordinates
                coord = mapply(Z.dot(self.quanti_var_["coord"].loc[Z.columns.tolist(),:]),lambda x : x/self.separate_analyses_[grp].eig_.iloc[0,0],
                                       axis=0,progressbar=False,n_workers=n_workers)
                num_coord = len(self.call_["group"].keys())*mapply(coord,lambda x : x/np.sqrt(self.eig_.iloc[:,0][:self.call_["n_components"]]),
                                                                   axis=1,progressbar=False,n_workers=n_workers)
                num_row_partial = num_row_partial + num_coord
            # If all variables in group are categoricals
            elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
                # Partiel coordinates
                coord = mapply(dummies.dot(self.quali_var_["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),
                               axis=0,progressbar=False,n_workers=n_workers)
                cat_coord = len(self.call_["group"].keys())*mapply(coord,lambda x : x/self.eig_.iloc[:,0][:self.call_["n_components"]],
                                                                   axis=1,progressbar=False,n_workers=n_workers)
                cat_row_partial = cat_row_partial + cat_coord
            row_coord = row_coord + (1/len(self.call_["group"].keys()))*(num_row_partial + cat_row_partial)
        return row_coord

######################################################################################################################
#   MULTIPLE FACTOR ANALYSIS FOR CONTINGENCY TABLES (MFACT)
######################################################################################################################

class MFACT(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis For Contingency Tables (MFACT)
    -------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Factor Analysis for Contingency Tables in the sense of Pagès J. (2002) with supplementary individuals
    and supplementary groups of variables. Groups of variables can be quantitative, categorical.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    group : a list or a tuple with the number of variables in each group

    name_group : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)
    
    num_group_sup : the indexes of the illustrative groups (by default, None and no group re illustrative)

    row_sup : an integer, a list or a tuple of the supplementary rows
    
    Parallelize : bool, default = False. Adding multi-core methods to PandasObject.

    Return
    ------
    separate_analyses_ : the results for the separate analyses

    svd_ : a dictionary of matrices containing all the results of the singular value decomposition

    eig_ : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the
            cumulative percentge of variance

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,
            contributions)
    
    freq_ : a dictionary of pandas dataframe containing all the results for the frequencies variables (coordinates, contribution, cos2)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfact'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    """

    def __init__(self,
                 n_components = 5,
                 group = None,
                 name_group = None,
                 num_group_sup = None,
                 row_sup = None,
                 parallelize= False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.num_group_sup = num_group_sup
        self.row_sup = row_sup
        self.parallelize = parallelize
    
    def fit(self, X, y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : pandas DataFrame, shape (n_rows, n_columns)

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
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
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

        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")
        
        ########################################################################################################################
        #   check if two categoricals variables have same categories
        ######################################################################################################################
        X = revaluate_cat_variable(X)

        #########################################################################################################################
        #   Check if group is None
        #########################################################################################################################
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("Error : 'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]
        
        #######################################################################################################################
        # Remove supplementary group
        if self.num_group_sup is not None:
            # Set default values to None
            self.freq_sup_ = None
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        ##################################################################
        # Check if supplementary rows
        if self.row_sup is not None:
            if (isinstance(self.row_sup,int) or isinstance(self.row_sup,float)):
                row_sup = [int(self.row_sup)]
            elif ((isinstance(self.row_sup,list) or isinstance(self.row_sup,tuple)) and len(self.row_sup)>=1):
                row_sup = [int(x) for x in self.row_sup]

        ############################################################################################################################
        #  Assigned group name
        ###########################################################################################################################
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("Error : 'group_name' must be a list or a tuple of group name")
        else:
            group_name = [x for x in self.name_group]
        
        ##############################################################################################################################
        # check if group name is an integer
        #############################################################################################################################
        for i in range(len(group_name)):
            if isinstance(group_name[i],int) or isinstance(group_name[i],float):
                group_name[i] = "Gr"+str(i+1)
        
        ##############################################################################################################################
        #   Assigned group name to label
        #############################################################################################################################
        group_active_dict = {}
        group_sup_dict = {}
        debut = 0
        for i in range(len(nb_elt_group)):
            X_group = X.iloc[:,(debut):(debut+nb_elt_group[i])]
            if self.num_group_sup is not None:
                if i in num_group_sup:
                    new_elt = {group_name[i]:X_group.columns.tolist()}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
            debut = debut + nb_elt_group[i]

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

       ######################################## Drop supplementary groups columns #######################################
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))
        
        ######################################## Drop supplementary individuals  ##############################################
        if self.row_sup is not None:
            # Extract supplementary individuals
            X_row_sup = X.iloc[self.row_sup,:]
            # Drop supplementary individuals
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in row_sup])
        
        ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"Error : {grp} group should have at least two columns")
        
        ######################################## Compute Frequency in all table #######################################
        F = X/X.sum().sum()

        ########################################################################################################################"
        ##  Row margins and columns margins
        #######################################################################################################################
        F_jt = {}
        Fi_t = {}
        for grp, cols in group_active_dict.items():
            F_jt[grp] = F[cols].sum(axis=0)
            Fi_t[grp] = F[cols].sum(axis=1)
        
        #### Set row margin and columns margin
        row_margin = F.sum(axis=1)
        col_margin = F.sum(axis=0)

        ####################################### Sum of frequency by group #############################################
        sum_term_grp = pd.Series().astype("float")
        for grp, cols in group_active_dict.items():
            sum_term_grp.loc[grp] = F[cols].sum().sum()
        
        ########################################### Construction of table Z #############################################"
        X1 = mapply(F,lambda x : x/col_margin.values,axis=1,progressbar=False,n_workers=n_workers)
        X2 = pd.DataFrame(columns=list(group_active_dict.keys()),index=X.index.tolist()).astype("float")
        for grp, cols in group_active_dict.items():
            X2[grp] = F[cols].sum(axis=1)/sum_term_grp[grp]

        ##########
        base = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            Zb = mapply(X1[cols],lambda x : x - X2[grp].values,axis=0,progressbar=False,n_workers=n_workers)
            Zb = mapply(Zb,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
            base = pd.concat((base,Zb),axis=1)
        
        # Run a Principal Component Analysis (PCA) in each group
        model = {}
        for grp, cols in group_active_dict.items():
            fa = PCA(standardize=False,ind_sup=None,ind_weights=row_margin.values.tolist(),var_weights=F_jt[grp].values.tolist())
            model[grp] = fa.fit(base[cols])
        
        ##################### Compute group disto
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="dist2")

        ##### Compute group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="dist2")

        ##### Store separate analysis
        self.separate_analyses_ = model

        #################################################
        ####### columns weights
        var_weights = pd.Series(name="weight").astype("float")
        for grp, cols in group_active_dict.items():
            weights = F_jt[grp]/model[grp].eig_.iloc[0,0]
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        ######################################################## Global
        global_pca = PCA(standardize=False,n_components=None,ind_weights=row_margin.values.tolist(),var_weights=var_weights.values.tolist(),parallelize=self.parallelize)
        global_pca.fit(base)

        ############################################# Removing duplicate value in cumulative percent #######################"
        cumulative = sorted(list(set(global_pca.eig_.iloc[:,3])))

        # Number of components
        if self.n_components is None:
            n_components = len(cumulative)
        else:
            n_components = min(self.n_components,len(cumulative))
        
         # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : row_margin,
                      "var_weights" : var_weights,
                      "group" : group_active_dict,
                      "group_name" : group_name}

        ########################################## Store global PCA
        self.global_pca_ = global_pca

        ##################################################################################################################
        #   Eigenvalues
        ##################################################################################################################
        self.eig_ = global_pca.eig_.iloc[:len(cumulative),:]

        ####### Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:len(cumulative)],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################
        ind = {"coord" : global_pca.ind_["coord"].iloc[:,:n_components],"contrib" : global_pca.ind_["contrib"].iloc[:,:n_components],"cos2" : global_pca.ind_["cos2"].iloc[:,:n_components],"infos":global_pca.ind_["infos"]}
        self.ind_ = ind

        ######################################################################################################
        #    Columns informations
        #######################################################################################################
        freq = {"coord" : global_pca.var_["coord"].iloc[:,:n_components],"contrib" : global_pca.var_["contrib"].iloc[:,:n_components],"cos2" : global_pca.var_["cos2"].iloc[:,:n_components]} 
        self.freq_ = freq


        ############################### group result
        self.group_ = {"dist2" : group_dist2}

        # Name of model
        self.model_ = "mfact"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit to data, then transform it.

        Parameters:
        ----------
        X : pandas DataFrame of shape (n_rows_,n_cols_)

        y : None
            y is ignored

        """
        self.fit(X)
        return self.ind_["coord"]
    
    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ----------------------------------------

        X is projected on the first axes previous extracted from a
        training set

        Parameters
        ----------
        X : pandas/polars DataFrame of shape (n_rows_sup, n_cols_)
            New data, where n_rows_sup is the number of supplementary
            row points and n_cols_ is the number of columns.
            X rows correspond to supplementary row points that are projected
            on the axes.
        
        y : None
            y is ignored
        
        Return
        ------
        X_new : array of float, shape (n_row_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("Error : DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        raise NotImplementedError("Error : This method is not yet implemented")