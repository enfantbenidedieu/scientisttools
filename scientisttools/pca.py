# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin

from .function_eta2 import function_eta2
from .weightedcorrcoef import weightedcorrcoef
from .revaluate_cat_variable import revaluate_cat_variable
from .svd_triplet import svd_triplet

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
    
    ind_sup : an integer or a list/tuple indicating the indexes of the supplementary individuals

    quanti_sup : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    quali_sup : an integer or a list/tuple indicating the indexes of the categorical supplementary variables

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
            raise TypeError("PCA does not support sparse input.")
        
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
        
        ####################################### Check if missing values
        if X.isnull().any().any():
            if self.quali_sup is None:
                X = mapply(X, lambda x : x.fillna(x.mean(),inplace=True),axis=0,progressbar=False,n_workers=n_workers)
            else:
                col_list = [x for x in list(range(X.shape[1])) if x not in quali_sup]
                for i in col_list:
                    if X.iloc[:,i].isnull().any():
                        X.iloc[:,i] = X.iloc[:,i].fillna(X.iloc[:,i].mean())
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
            X_ind_sup = X.iloc[ind_sup,:]
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
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,X.shape[0]-1,X.shape[1])
        
        ################################################################################################
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
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
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
        bs_pvalue = 1-sp.stats.chi2.cdf(bartlett_stats,df=bs_dof)
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
            raise ValueError("'columns' aren't aligned")

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