# -*- coding: utf-8 -*-

# https://kiwidamien.github.io/making-a-python-package.html
##################################### Chargement des librairies
from functools import reduce
import numpy as np
import pandas as pd
from mapply.mapply import mapply
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
from scipy.spatial.distance import pdist,squareform
from scipy.sparse import issparse
import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from scientisttools.utils import (
    orthonormalize,
    random_orthonormal,
    weighted_mean,
    solve_weighted,
    check_array_with_weights,
    global_kmo_index,
    per_item_kmo_index,
    eta2,
    from_dummies,
    svd_triplet,
    function_eta2,
    weightedcorrcoef)

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
    bases on the Singular Value Decomposition

    Performs Principal Component Analysis (PCA) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Missing values are replaced by the column mean.

    Usage
    -----
    PCA(normalize=True,
        n_components=None,
        row_labels=None,
        col_labels=None,
        row_sup_labels =None,
        quanti_sup_labels = None,
        quali_sup_labels = None,
        parallelize=False).fit(X)

    where X a data frame with n_rows (individuals) and p columns (numeric variables).

    Parameters
    ----------
    normalize : bool, default = True
        - If True : the data are scaled to unit variance.
        - If False : the data are not scaled to unit variance.

    n_components : int or None, default = 5

    row_sup_labels : array of strings or None, defulat = None
        This array provides the supplementary individuals labels

    quanti_sup_labels : arrays of strings or None, default = None
        This array provides the quantitative supplementary variables labels

    quali_sup_labels : array of strings or None, default = None
        This array provides the categorical supplementary variables labels

    parallelize : bool, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.

    row_sup_labels_ : array of strings or None
        Labels of supplementary individuals labels

    quanti_sup_labels_ : arrays of strings or None
        Labels of quantitative supplementary variables

    quali_sup_labels_ : arrays of strings or None

    mod_sup_labels_ : list of strings
                        labels for the categories supplementary

    short_sup_labels_ : list of strings
                        Short labels for the categories supplementary

    eig_ : array of float
        A 4 x n_components_ matrix containing all the eigenvalues
        (1st row), difference (2nd row) the percentage of variance (3rd row) and the
        cumulative percentage of variance (4th row).

    
    

    model_ : string
        The model fitted = 'pca'
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
        """Fit the model to X

        Parameters
        ----------
        X : pandas DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored

        Returns:
        --------
        self : object
                Returns the instance itself
        """

        # Return data


        # Check if sparse matrix
        if issparse(X):
            raise TypeError("PCA does not support sparse input.")
        # Check if X is an instance of pd.DataFrame class
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # set parallelize
        if self.parallelize:
            self.n_workers = -1
        else:
            self.n_workers = 1

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
                quanti_sup = list(int(self.quanti_sup))
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = list(int(self.ind_sup))
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        ####################################### Check NA
        if X.isnull().any().any():
            if self.quali_sup is None:
                X = mapply(X, lambda x : x.fillna(x.mean(),inplace=True),axis=0,progressbar=False,n_workers=self.n_workers_)
            else:
                col_list = [x for x in list(range(X.shape[0])) if x not in quali_sup]
                X.iloc[:,col_list] = X.iloc[:,col_list].fillna(X[:,col_list].mean())
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
        
        ####################################### Principal Components Analysis (PCA) ##################################################

        ################## Summary quantitatives variables ####################
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")

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
        ind_dist2 = mapply(Z,lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=self.n_workers).sum(axis=1)
        ind_dist2.name = "dist"
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        ind_infos = pd.concat([np.sqrt(ind_dist2),ind_inertia],axis=1)
        ind_infos.insert(1,"weight",ind_weights)

        ################################ Columns informations ##################################################
        var_dist2 = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=self.n_workers).sum(axis=0)
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
        ind_contrib = mapply(ind_coord,lambda x : 100*(x**2)*ind_weights,axis=0,progressbar=False,n_workers=self.n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=self.n_workers)

        # Variables contributions
        var_contrib = mapply(var_coord,lambda x : 100*(x**2)*var_weights,axis=0,progressbar=False,n_workers=self.n_workers)
        var_contrib = mapply(var_contrib, lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=self.n_workers)
    
        ####################################### Cos2 ###########################################
        # Individuals Cos2
        ind_cos2 = mapply(ind_coord,lambda x : x**2/ind_dist2,axis=0,progressbar=False,n_workers=self.n_workers)

        # Variables Cos2
        cor_var  = mapply(var_coord,lambda x : x/np.sqrt(var_dist2),axis=0,progressbar=False,n_workers=self.n_workers)
        var_cos2 = mapply(cor_var,  lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers)

        #### Weighted Pearson correlation
        weighted_corr = weightedcorrcoef(X,w=ind_weights)
        weighted_corr = pd.DataFrame(weighted_corr,index=X.columns.tolist(),columns=X.columns.tolist())

        #################################### Store result #############################################
        self.ind_ = {"coord":ind_coord,"cos2":ind_cos2,"contrib":ind_contrib,"dist":np.sqrt(ind_dist2),"infos" : ind_infos}
        self.var_ = {"coord":var_coord,"cor":cor_var,"cos2":var_cos2,"contrib":var_contrib,"weighted_corr":weighted_corr,"infos" : var_infos,"summary_quanti" : summary_quanti}

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
            ind_sup_coord = mapply(Z_ind_sup,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=self.n_workers)
            ind_sup_coord = np.dot(ind_sup_coord,svd["V"])
            ind_sup_coord = pd.DataFrame(ind_sup_coord,index=X_ind_sup.index.tolist(),columns=["Dim."+str(x+1) for x in range(ind_sup_coord.shape[1])])

            ###### Distance to origin
            ind_sup_dist2 = mapply(Z_ind_sup,lambda  x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=self.n_workers).sum(axis=1)
            ind_sup_dist2.name = "dist"

            ######## Compute cos2
            ind_sup_cos2 = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_dist2,axis=0,progressbar=False,n_workers=self.n_workers)

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
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")

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
            var_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=self.n_workers)
            var_sup_coord = np.dot(var_sup_coord.T,svd["U"])
            var_sup_coord = pd.DataFrame(var_sup_coord,index=X_quanti_sup.columns.tolist(),columns = ["Dim."+str(x+1) for x in range(var_sup_coord.shape[1])])

            ############# Supplementary quantitatives variables Cos2
            var_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=self.n_workers)
            var_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),var_sup_cor)
            var_sup_cos2 = mapply(var_sup_coord,lambda x : (x**2)/np.sqrt(var_sup_dist2),axis=0,progressbar=False,n_workers=self.n_workers)

            # Weighted correlation between supplementary quantitatives variables and actives quantitatives
            var_sup_corr = weightedcorrcoef(x=X_quanti_sup,y=X,w=ind_weights)[:X_quanti_sup.shape[1],:]
            var_sup_corr = pd.DataFrame(var_sup_corr,columns=X_quanti_sup.columns.tolist()+X.columns.tolist(),index=X_quanti_sup.columns.tolist())        

            # Store supplementary quantitatives informations
            self.quanti_sup_ =  {"coord":var_sup_coord,
                                 "cor" : var_sup_coord,
                                 "cos2" : var_sup_cos2,
                                 "weighted_corr" : var_sup_corr,
                                 "summary_quanti" : summary_quanti_sup}

        #################################################################################################################################################
        # Compute supplementary qualitatives variables statistics
        ###############################################################################################################################################
        if self.quali_sup is not None:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
            
            ######################################## Barycentre of DataFrame ########################################
            X_quali_sup = X_quali_sup.astype("object")
            ####################################" Correlation ratio #####################################################
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,
                                                      n_workers=self.n_workers) for col in X_quali_sup.columns.tolist()),axis=0)

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
            quali_sup_dist2  = mapply(bary, lambda x : x**2*var_weights,axis=1,progressbar=False,n_workers=self.n_workers).sum(axis=1)
            quali_sup_dist2.name = "dist"

            ################################### Barycentrique coordinates #############################################
            quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=self.n_workers)
            quali_sup_coord = quali_sup_coord.dot(svd["V"])
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(quali_sup_coord.shape[1])]

            ################################## Cos2
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=self.n_workers)
            
            ################################## v-test
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=self.n_workers)
            quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((X.shape[0]-n_k[k])/((X.shape[0]-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

            #################################### Summary quali
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")

            # Supplementary categories informations
            self.quali_sup_ = {"coord" : quali_sup_coord,
                               "cos2" : quali_sup_cos2,
                               "vtest" : quali_sup_vtest,
                               "dist" : np.sqrt(quali_sup_dist2),
                               "eta2" : quali_sup_eta2,
                               "barycentre" : barycentre,
                               "summary_quali" : summary_quali_sup}
            
        ########################################################################################################
        # store model name
        self.model_ = "pca"

        return self

    def transform(self,X,y=None):
        """Apply the dimensionality reduction on X

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
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        X = X.astype("float")

        ######### check if X.shape[1] = ncols
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("Error : 'columns' aren't aligned")

        # Apply transition relation
        Z = (X.values - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)

        ###### Multiply by columns weight & Apply transition relation
        coord = np.apply_along_axis(func1d=lambda x : x*self.call_["var_weights"],axis=1,arr=Z).dot(self.svd_["V"])
        return coord

    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

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
        return self.ind_["coord"].values

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

    Usage
    -----
    CA(n_components=None,
       row_labels=None,
       col_labels=None,
       row_sup_labels=None,
       col_sup_labels=None,
       parallelize = False).fit(X)

    where X a data frame or a table with n rows and p columns, i.e. a contingency table.

    Parameters
    ----------
    n_components : int, float or None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
          that the amount of variance that needs to be explained is
          greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
              equal to n_components.
            - If n_components is float, select the higher number of
              components lower than n_components.

    row_labels : list of strings or None
        - If row_labels is a list of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.

    col_labels : list of strings or None
        - If col_labels is a list of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.

    row_sup_labels : list of strings or None
        - If row_sup_labels is a list of strings : this array provides the
          supplementary row labels.

    col_sup_labels :  list of strings or None
        - If col_sup_labels is a list of strings : this array provides the
          supplementary columns labels.

    The sum of the absolute frequencies in the X array.

    model_ : string
        The model fitted = 'ca'
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
        """ Fit the model to X
        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
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
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set type to int
        X = X.astype("int")
        row_sum = X.sum(axis=1)
        X = np.apply_along_axis(func1d=lambda x : x/row_sum,axis=0,arr=X)
        coord = np.dot(X,self.svd_["V"])
        
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
        return self.row_["coord"].values

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

    This class performs Multiple Correspondence Analysis (MCA) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Usage
    ----

    Parameters
    ----------

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
                 na_method = "drop",
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.benzecri = benzecri
        self.greenacre = greenacre
        self.ind_sup = ind_sup
        self.quali_sup = quali_sup
        self.quanti_sup = quanti_sup
        self.na_method = na_method
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """

        """
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        ######
        if self.na_method not in ["drop","include"]:
            raise ValueError("Error : 'na_method' should be one of 'drop', 'include'")

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
        
        ####################################### Check NA in quantitatives variables
        if X.isnull().any().any():
            if self.quanti_sup is not None:
                X.iloc[:,quanti_sup] = mapply(X.iloc[:,quanti_sup], lambda x : x.fillna(x.mean(),inplace=True),axis=0,progressbar=False,n_workers=n_workers)
            else:
                col_list = [x for x in list(range(X.shape[0])) if x not in quali_sup]
                X.iloc[:,col_list] = X.iloc[:,col_list].fillna(X[:,col_list].mean())
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
        
        ####################################### Multiple Correspondence Anlysis (MCA) ##################################################
            
        #########################################################################################################
        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X.columns.tolist():
            eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")

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
        var_weights = {}
        if self.var_weights is None:
            for col in X.columns.tolist():
                var_weights[col] = 1/X.shape[1]
        elif not isinstance(self.var_weights,dict):
            raise ValueError("Error : 'var_weights' must be a dictionary where keys are variables names and values are variables weights.")
        else:
            for col in self.var_labels_:
                var_weights[col] = self.var_weights[col]/sum(self.var_weights.values())

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
            # Concatenate with activate summary
            summary_quali = pd.concat((summary_quali,summary_quali_sup),axis=0,ignore_index=True)

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
            chi2_test = pd.concat((chi2_test,chi2_test2),axis=0,ignore_index=True)
            
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
                chi2_test = pd.concat((chi2_test,chi2_test3),axis=0,ignore_index=True)

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
        
        # Set chi2
        self.chi2_test_ = chi2_test 
        # Set summary quali
        self.summary_quali_ = summary_quali

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

        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #self._compute_row_sup_stats(X)
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
        ind_sup_coord = np.apply_along_axis(func1d=lambda x : x/np.sqrt(self.eig_.iloc[:,0][:self.call_["n_components"]]),axis=1,arr=ind_sup_coord)
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
        return self.ind_["coord"].values


##########################################################################################
#           PARTIAL PRINCIPAL COMPONENTS ANALYSIS (PPCA)
##########################################################################################

class PartialPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Components Analysis (PartialPCA)
    --------------------------------------------------

    Description
    -----------


    Parameters:
    -----------




    Returns:
    --------
    """
    def __init__(self,
                 n_components=None,
                 normalize=True,
                 row_labels=None,
                 col_labels=None,
                 partial_labels=None,
                 parallelize = False):
        self.n_components = n_components
        self.normalize = normalize
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.partial_labels = partial_labels
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize option
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        self.n_rows_, self.n_cols_ = X.shape
        self.data_ = X

        self._compute_stats(X)
        self._compute_svds(X)

        return self


    def _compute_stats(self,X,y=None):
        """


        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        global_kmo = global_kmo_index(X)
        per_var_kmo = per_item_kmo_index(X)
        corr = X.corr(method="pearson")
        pcorr = X.pcorr()

        self.global_kmo_index_ = global_kmo
        self.partial_kmo_index_ = per_var_kmo
        self.pearson_correlation_ = corr
        self.partial_correlation_ = pcorr

    def _compute_svds(self,X,y=None):
        """

        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")


        self.partial_labels_ = self.partial_labels
        X = X.drop(columns = self.partial_labels_)

        # Extract coefficients and intercept
        coef = pd.DataFrame(np.zeros((len(self.partial_labels_)+1,X.shape[1])),
                            index = ["intercept"]+self.partial_labels_,columns=X.columns)
        rsquared = pd.DataFrame(np.zeros((1,X.shape[1])),index = ["R carré"],columns=X.columns)
        rmse = pd.DataFrame(np.zeros((1,X.shape[1])),index = ["RMSE"],columns=X.columns)
        E = pd.DataFrame(np.zeros((self.n_rows_,X.shape[1])),index=X.index,columns=X.columns) # Résidu de régression

        for lab in X.columns:
            res = smf.ols(formula="{}~{}".format(lab,"+".join(self.partial_labels_)), data=self.data_).fit()
            coef.loc[:,lab] = res.params.values
            rsquared.loc[:,lab] = res.rsquared
            rmse.loc[:,lab] = mean_squared_error(self.data_[lab],res.fittedvalues,squared=False)
            E.loc[:,lab] = res.resid

        # Coefficients normalisés
        normalized_data = mapply(self.data_,lambda x : (x - x.mean())/x.std(),axis=0,progressbar=False,n_workers=self.n_workers_)
        normalized_coef = pd.DataFrame(np.zeros((len(self.partial_labels_),X.shape[1])),
                                       index = self.partial_labels_,columns=X.columns)
        for lab in X.columns:
            normalized_coef.loc[:,lab] = smf.ols(formula="{}~{}".format(lab,"+".join(self.partial_labels_)),data=normalized_data).fit().params[1:]

        # Matrice des corrélations partielles vers y
        resid_corr = E.corr(method="pearson")

        # Matrice des corrélations brutes
        R = X.corr(method="pearson")

        # ACP sur les résidus
        self.row_labels_ = self.row_labels
        my_pca = PCA(normalize=self.normalize,n_components=self.n_components,row_labels=self.row_labels_,col_labels=E.columns).fit(E)

        self.resid_corr_ = resid_corr

        self.n_components_ = my_pca.n_components_

        self.eig_ = my_pca.eig_
        self.svd_ = my_pca.svd_
        self.inertia_ = my_pca.inertia_
        self.dim_index_ =  my_pca.dim_index_

        self.row_coord_ = my_pca.row_coord_
        self.row_contrib_ = my_pca.row_contrib_
        self.row_cos2_ = my_pca.row_cos2_
        self.row_infos_ = my_pca.row_infos_

        self.col_coord_ = my_pca.col_coord_
        self.col_cor_ = my_pca.col_cor_
        self.col_ftest = my_pca.col_ftest_
        self.col_cos2_ = my_pca.col_cos2_
        self.col_contrib_ = my_pca.col_contrib_

        self.bartlett_sphericity_test_ = my_pca.bartlett_sphericity_test_
        self.kaiser_proportion_threshold_ = my_pca.kaiser_proportion_threshold_
        self.kaiser_threshold_ = my_pca.kaiser_threshold_
        self.broken_stick_threshold_ = my_pca.broken_stick_threshold_
        self.kss_threshold_ = my_pca.kss_threshold_
        self.col_labels_ = my_pca.col_labels_

        self.rsquared_ = rsquared
        self.rmse_ = rmse
        self.coef_ = coef
        self.normalized_coef_ = normalized_coef
        self.normalized_data_ = normalized_data
        self.resid_ = E
        self.R_ = R

        self.model_ = "ppca"

    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

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
        return self.row_coord_

    def transform(self,X,y=None):
        """Apply the Partial Principal Components Analysis reduction on X

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
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        raise NotImplementedError("Error : This method is not implemented yet.")

##############################################################################################
#       EXPLORATORY FACTOR ANALYSIS (EFA)
###############################################################################################

class EFA(BaseEstimator,TransformerMixin):
    """Exploratory Factor Analysis

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    EFA performs a Exploratory Factor Analysis, given a table of
    numeric variables; shape = n_rows x n_columns

    Parameters
    ----------
    normalize : bool
        - If true : the data are scaled to unit variance
        - If False : the data are not scaled to unit variance

    n_components: int or None
        number of components to keep

    row_labels : list of string or None
        The list provides the row labels

    col_labels : list of strings or None
        The list provides the columns labels

    method : {"principal","harris"}
        - If method = "principal" : performs Exploratory Factor Analyis using principal approach
        - If method = "harris" : performs Exploratory Factor Analysis using Harris approach

    row_sup_labels : list of strings or None
        The list provides the supplementary row labels

    quanti_sup_labels : list of strings or None
        The list provides the supplementary continuous columns

    quali_sup_labels : list of strings or None
        The list provides the supplementary categorical variables

    graph : bool or None
        - If True : return graph

    figsize = tuple of int or None

    Returns:
    --------

    """
    def __init__(self,
                normalize=True,
                n_components = None,
                row_labels = None,
                col_labels = None,
                method = "principal",
                row_sup_labels = None,
                quanti_sup_labels = None,
                quali_sup_labels = None):
        self.normalize = normalize
        self.n_components =n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.method = method
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels

    def fit(self,X,y=None):
        """Fit the model to X

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

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary numeric or categorical columns
        self.quanti_sup_labels_ = self.quanti_sup_labels
        self.quali_sup_labels_ = self.quali_sup_labels
        if ((self.quali_sup_labels_ is not None) and (self.quanti_sup_labels_ is not None)):
            X_ = _X.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)
        elif self.quali_sup_labels_ is not None:
            X_= _X.drop(columns = self.quali_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_)
        elif self.quanti_sup_labels_ is not None:
            X_ = _X.drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup  = row_sup.drop(columns = self.quanti_sup_labels_)
        else:
            X_ = _X

        # Save dataframe
        self.data_ = X
        self.active_data_ = X_

        # Dimension
        self.n_rows_, self.n_cols_ = X_.shape

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]

        # Set col labels
        self.col_labels_ = self.col_labels
        if ((self.col_labels_ is None) or (len(self.col_labels_) != self.n_cols_)):
            self.col_labels_ = ["col_" + str(k+1) for k in np.arange(0,self.n_cols_)]

        # Initialisation
        self.uniqueness_    = None
        self.row_sup_coord_ = None
        self.col_sup_coord_ = None

        #
        self.estimated_communality_ = None
        self.col_coord_             = None
        self.col_contrib_           = None
        self.explained_variance_    = None
        self.percentage_variance_   = None
        self.factor_score_          = None
        self.factor_fidelity_       = None
        self.row_coord_             = None

        # Correlation Matrix
        self.correlation_matrix_ = X_.corr(method= "pearson")

        # Rsquared
        self.initial_communality_ =  np.array([1 - (1/x) for x in np.diag(np.linalg.inv(self.correlation_matrix_))])
        # Total inertia
        self.inertia_ = np.sum(self.initial_communality_)

        # Scale - data
        self.means_ = np.mean(X_.values, axis=0).reshape(1,-1)
        if self.normalize:
            self.std_ = np.std(X_.values,axis=0,ddof=0).reshape(1,-1)
            Z = (X_ - self.means_)/self.std_
        else:
            Z = X_ - self.means_

        self.normalized_data_ = Z

        if self.method == "principal":
            self._compute_principal(X_)
        elif self.method == "harris":
            self._compute_harris(X_)

        # Compute supplementrary rows statistics
        if self.row_sup_labels_ is not None:
            self._compute_row_sup_stats(X=row_sup)

        self.model_ = "efa"

        return self

    def _compute_eig(self,X):
        """Compute eigen decomposition

        """

        # Eigen decomposition
        eigenvalue, eigenvector = np.linalg.eigh(X)

        # Sort eigenvalue
        eigen_values = np.flip(eigenvalue)
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = (eigenvalue > 0).sum()

        self.eig_ = np.array([eigen_values[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])

        self.eigen_vectors_ = eigenvector
        return eigenvalue, eigenvector

    def _compute_principal(self,X):
        """Compute EFA using principal approach


        """
        # Compute Pearson correlation matrix
        corr_prim = X.corr(method="pearson")

        # Fill diagonal with nitial communality
        np.fill_diagonal(corr_prim.values,self.initial_communality_)

        # eigen decomposition
        eigen_value,eigen_vector = self._compute_eig(corr_prim)
        eigen_value = np.flip(eigen_value)
        eigen_vector = np.fliplr(eigen_vector)

        # Compute columns coordinates
        col_coord = eigen_vector*np.sqrt(eigen_value)
        self.col_coord_ = col_coord[:,:self.n_components_]

        # Variance restituées
        explained_variance = np.sum(np.square(self.col_coord_),axis=0)

        # Communalité estimée
        estimated_communality = np.sum(np.square(self.col_coord_),axis=1)

        # Pourcentage expliquée par variables
        percentage_variance = estimated_communality/self.initial_communality_

        # F - scores
        factor_score = np.dot(np.linalg.inv(X.corr(method="pearson")),self.col_coord_)

        # Contribution des variances
        col_contrib = np.square(factor_score)/np.sum(np.square(factor_score),axis=0)

        # Fidélité des facteurs
        factor_fidelity = np.sum(factor_score*self.col_coord_,axis=0)

        # Row coordinates
        row_coord = np.dot(self.normalized_data_,factor_score)

        # Broken stick threshold
        broken_stick_threshold = np.flip(np.cumsum(1/np.arange(self.n_cols_,0,-1)))

        # Karlis - Saporta - Spinaki threshold
        kss = 1 + 2*np.sqrt((self.n_rows_-1)/(self.n_rows_-1))

        # Store all result
        self.estimated_communality_ = estimated_communality

        self.col_contrib_ = col_contrib[:,:self.n_components_]
        self.explained_variance_ = explained_variance
        self.percentage_variance_ = percentage_variance
        self.factor_score_ = factor_score
        self.factor_fidelity_ = factor_fidelity
        self.row_coord_ = row_coord[:,:self.n_components_]
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        # Add eigenvalue threshold informations
        self.kaiser_threshold_ = 1.0
        self.kaiser_proportion_threshold_ = 100/self.inertia_
        self.kss_threshold_ = kss
        self.broken_stick_threshold_ = broken_stick_threshold[:self.n_components_]


    def _compute_harris(self,X):
        """Compute EFA using harris method

        """

        self.uniqueness_ = 1 - self.initial_communality_

        # Save
        corr_prim = X.corr(method="pearson")
        np.fill_diagonal(corr_prim.values,self.initial_communality_)

        #  New correlation matrix
        corr_snd = np.zeros((self.n_cols_,self.n_cols_))
        for k in np.arange(0,self.n_cols_,1):
            for l in np.arange(0,self.n_cols_,1):
                corr_snd[k,l] = corr_prim.iloc[k,l]/np.sqrt(self.uniqueness_[k]*self.uniqueness_[l])

        eigen_value,eigen_vector = self._compute_eig(corr_snd)

    def _compute_row_sup_stats(self,X,y=None):
        """Compute statistics supplementary row

        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if self.method == "principal":
            if self.normalize:
                Z = (X - self.means_)/self.std_
            else:
                Z = X - self.means_

            self.row_sup_coord_ = np.dot(Z,self.factor_score_)[:,:self.n_components_]
        else:
            raise NotImplementedError("Error : This method is not implemented yet.")

    def _compute_quanti_sup_stats(self,X,y=None):
        """Compute quantitative supplementary variables

        """
        raise NotImplementedError("Error : This method is not implemented yet.")

    def _compute_quali_sup_stats(self,X,y=None):
        """Compute qualitative supplementary variables

        """
        raise NotImplementedError("Error : This method is not implemented yet.")

    def transform(self,X,y=None):
        """Apply the dimensionality reduction on X

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
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if self.method == "principal":
            if self.normalize:
                Z = (X - self.means_)/self.std_
            else:
                Z = X - self.means_
            return np.dot(Z,self.factor_score_)[:,:self.n_components_]
        else:
            raise NotImplementedError("Error : This method is not implemented yet.")

    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

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
        return self.row_coord_

################################################################################################
#                                   CORRESPONDENCE ANALYSIS (CA)
################################################################################################

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)


#################################################################################################################
#   CANONICAL CORRESPONDENCE ANALYSIS (CCA)
#################################################################################################################
    
class CCA(BaseEstimator,TransformerMixin):
    """
    CANONICAL CORRESPONDENCE ANALYSIS (CCA)
    ---------------------------------------
    
    """
    def __init__(self):
        pass


################################################################################################################
#    PARTIEL CANONICAL CORRESPONDENCE ANALYSIS (PCCA)
###############################################################################################################

class PCCA(BaseEstimator,TransformerMixin):
    """
    PARTIEL CANONICAL CORRESPONDENCE ANALYSIS (PCCA)
    ------------------------------------------------

    Description
    -----------
    
    
    
    """
    def __init__(self):
        pass


################################################################################################################
#   PARTIAL LEAST SQUARE CANONICAL CORRESPONDENCE ANALYSIS (PLSCCA)
###############################################################################################################

class PLSCCA(BaseEstimator,TransformerMixin):
    """
    PARTIAL LEAST SQUARE CANONICAL CORRESPONDENCE ANALYSIS (PLSCCA)
    ---------------------------------------------------------------
    
    
    
    """
    def __init__(self):
        pass


#############################################################################################
#               FACTOR ANALYSIS OF MIXED DATA (FAMD)
#############################################################################################

class FAMD(BaseEstimator,TransformerMixin):
    """
    Factor Analysis of Mixed Data
    ------------------------------

    Performs Factor Analysis of Mixed Data (FAMD) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Parameters:
    -----------
    see scientisttools.decomposition.PCA and scientisttools.decomposition.MCA

    """
    def __init__(self,
                 normalize=True,
                 n_components = None,
                 row_labels = None,
                 row_weights = None,
                 quanti_labels = None,
                 quanti_weights = None,
                 quali_labels=None,
                 quali_weights = None,
                 row_sup_labels=None,
                 quanti_sup_labels=None,
                 quali_sup_labels=None,
                 parallelize = False):
        self.normalize =normalize
        self.n_components = n_components
        self.row_labels = row_labels
        self.row_weights = row_weights
        self.quanti_labels = quanti_labels
        self.quanti_weights = quanti_weights
        self.quali_labels = quali_labels
        self.quali_weights = quali_weights
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.parallelize = parallelize

    def fit(self,X, y=None):
        """


        """

        # Chack if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize option
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary numeric or categorical columns
        self.quali_sup_labels_ = self.quali_sup_labels
        self.quanti_sup_labels_ = self.quanti_sup_labels
        if ((self.quali_sup_labels_ is not None) and (self.quanti_sup_labels_ is not None)):
            col_to_drop = self.quali_sup_labels_ + self.quanti_sup_labels_
            X_ = _X.drop(columns=col_to_drop)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns=col_to_drop)
        elif self.quali_sup_labels_ is not None:
            X_= _X.drop(columns = self.quali_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_)
        elif self.quanti_sup_labels_ is not None:
            X_ = _X.drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup  = row_sup.drop(columns = self.quanti_sup_labels_)
        else:
            X_ = _X

        # Save initial data
        self.data_ = X
        self.active_data_ = X_

        # Apply Singular Value Decomposition (SVD)
        self._compute_svd(X=X_)

        # Compute supplementary individuals informations
        if self.row_sup_labels_ is not None:
            self._compute_row_sup_stats(X=row_sup)

        # Compute supplementary quantitatives informations
        if self.quanti_sup_labels_ is not None:
            self._compute_quanti_sup_stats(X=_X[self.quanti_sup_labels_])

        # Compute supplementary qualitatives informations
        if self.quali_sup_labels_ is not None:
            self._compute_quali_sup_stats(X=_X[self.quali_sup_labels_])

        return self
    
    def _compute_svd(self,X):
        """
        Singular Value Decomposition (SVD)
        ----------------------------------

        Parameters
        ----------
        X : pandas dataframe of shape (n_rows,n_cols)

        Return:
        ------
        None
        """

        # Compute statistics
        self.n_rows_ = X.shape[0]
        X_quant = X.select_dtypes(include=np.number)
        X_qual = X.select_dtypes(include=["object","category"])

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]

        # Check if NULL
        if X_quant.empty and not X_qual.empty:
            raise ValueError("Error : There is no continuous variables in X. Please use MCA function.")
        elif X_qual.empty and not X_quant.empty:
            raise ValueError("Error : There is no categoricals variables in X. Please use PCA function.")

        # set row labels
        if self.row_labels is None:
            self.row_labels_ = X.index.tolist()
        else:
            self.row_labels_ = self.row_labels

        # Set qualitatives labels
        if self.quali_labels is None:
            self.quali_labels_ = X_qual.columns.tolist()
        else:
            self.quali_labels_ = self.quali_labels

        # Set quantitatives labels
        if self.quanti_labels is None:
            self.quanti_labels_ = X_quant.columns.tolist()
        else:
            self.quanti_labels_ = self.quanti_labels

        ######################
        self.quanti_data_ = X_quant
        self.quali_data_ = X_qual

        ###### Set number of continuous variables and number of qualitatives variables
        self.n_cols_ = X_quant.shape[1]
        self.n_vars_ = X_qual.shape[1]

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
        chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
        idx = 0
        for i in np.arange(self.n_vars_-1):
            for j in np.arange(i+1,self.n_vars_):
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
        if self.row_weights is None:
            self.row_weights_ = np.ones(self.n_rows_)/self.n_rows_
        elif not isinstance(self.row_weights,list):
            raise ValueError("Error : 'row_weights' must be a list of row weight.")
        elif len(self.row_weights) != self.n_rows_:
            raise ValueError(f"Error : 'row_weights' must be a list with length {self.n_rows_}.")
        else:
            self.row_weights_ = np.array([x/np.sum(self.row_weights) for x in self.row_weights])
        
        ####################################################################################################
        ################################## Treatment of continues variables ################################
        ####################################################################################################
        # Set columns weight
        if self.quanti_weights is None:
            self.col_weights_ = np.ones(self.n_cols_)
        elif not isinstance(self.quanti_weights,list):
            raise ValueError("Error : 'quanti_weights' must be a list of columns weight.")
        elif len(self.quanti_weights) != self.n_cols_:
            raise ValueError(f"Error : 'quanti_weights' must be a list with length {self.n_cols_}.")
        else:
            self.col_weights_ = np.array(self.quanti_weights)
            
        # Weighted Pearson correlation between continuous variables
        self.col_corr_ = weightedcorrcoef(x=X_quant,w=self.row_weights_)

        # Partial correlation between continuous variables
        self.col_pcorr_ = np.array(X_quant.pcorr())

         ############# Compute weighted average mean and standard deviation
        d1 = DescrStatsW(X_quant,weights=self.row_weights_,ddof=0)

        # Initializations - scale data
        self.means_ = d1.mean.reshape(1,-1)
        if self.normalize:
            self.std_ = d1.std.reshape(1,-1)
        else:
            self.std_ = np.ones(self.n_cols_).reshape(1,-1)
        # Z = (X - mu)/sigma
        Z1 = (X_quant - self.means_)/self.std_

        ###############################################################################################
        ##################################### Treatment of qualitatives variables #####################
        ###############################################################################################

        ################### Set variables weights ##################################################
        self.var_weights_ = {}
        if self.quali_weights is None:
            for col in X_qual.columns.tolist():
                self.var_weights_[col] = 1/self.n_vars_
        elif not isinstance(self.quali_weights,dict):
            raise ValueError("Error : 'quali_weights' must be a dictionary where keys are qualitatives variables names and values are qualitatives variables weights.")
        elif len(self.quali_weights.keys()) != self.n_vars_:
            raise ValueError(f"Error : 'quali_weights' must be a dictionary with length {self.n_vars_}.")
        else:
            for col in X_qual.columns.tolist():
                self.var_weights_[col] = self.quali_weights[col]/sum(self.quali_weights.values())
        
        ###################### Set categories weights
        # Normalisation des variables qualitatives
        dummies = pd.concat((pd.get_dummies(X_qual[col],prefix=col,prefix_sep='_') for col in X_qual.columns.tolist()),axis=1)
        # Compute y_{ik}*(I/I_{k}) - 
        mod_stats = dummies.agg(func=[np.sum,np.mean]).T
        self.mod_stats_ = mod_stats

        ###### Define mod weights
        mod_weights = pd.Series().astype("float")
        for var in self.var_weights_:
            data = dummies.loc[:,dummies.columns.str.startswith(var)]
            weights = data.mean(axis=0)*self.var_weights_[var]
            mod_weights = pd.concat((mod_weights,weights),axis=0)
        self.mod_weights_ = mod_weights

        self.dummies_means_ = dummies.mean(axis=0)
        self.dummies_std_ = np.sqrt(self.dummies_means_)
        mod_stats = dummies.agg(func=[np.sum,np.mean]).T
        
        ############################ Compute weighted mean and weighted standards 
        # Normalize Z2
        Z2 = mapply(dummies,lambda x: x/np.sqrt(self.dummies_means_.values),axis = 1,progressbar=False,n_workers=self.n_workers_)

        # Concatenate the 2 dataframe
        Z = pd.concat([Z1,Z2],axis=1)

        #################### Set number of components
        if self.n_components is None:
            self.n_components_ = min(self.n_rows_-1, Z.shape[1]-self.n_vars_)
        elif not isinstance(self.n_components,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("Error : 'n_components' must be greater or equal than 1.")
        else:
            self.n_components_ = min(self.n_components, self.n_rows_-1, Z.shape[1]-self.n_vars_)

        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

         #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"X" : Z,
                      "row_weights" : self.row_weights_,
                      "mod_weights" : 1/self.dummies_means_,
                      "means" : self.means_,
                      "std" : self.std_}

        ########################################################################################################################
        #################### Informations about individuals #################################################################### 
        ########################################################################################################################
        # Distance between individuals
        row_dist = squareform(pdist(Z,metric='sqeuclidean'))
        # Distance between individuals and inertia center
        row_disto = (np.sum((Z1**2)*self.col_weights_,axis=1) +
                     mapply(dummies,lambda x:np.sum(1/self.dummies_means_.values*(x-self.dummies_means_.values)**2),
                            axis=1,progressbar=False,n_workers=self.n_workers_))
        # Individuals inertia
        row_inertie = row_disto*self.row_weights_
        # Save all informations
        row_infos = np.c_[np.sqrt(row_disto),self.row_weights_,row_inertie]

        ########################################################################################################################
        ################################  Informations about categories ########################################################
        ########################################################################################################################
        # Distance between ctegories
        n_k = dummies.sum(axis=0)
        dummies_weight = pd.concat((dummies.loc[:,k]/n_k[k] for k in dummies.columns.tolist()),axis=1)
        mod_dist = self.n_rows_*squareform(pdist(dummies_weight.T,metric="sqeuclidean"))
        # Distance à l'origine
        mod_disto = mapply(dummies_weight,lambda x : np.sum(self.n_rows_*(x-self.row_weights_)**2),axis=0,progressbar=False,n_workers=self.n_workers_)
        # Inertie des lignes
        mod_inertie = mod_disto*self.mod_weights_
        # Save all informations
        mod_infos = np.c_[np.sqrt(mod_disto), self.mod_weights_, mod_inertie]

        self.row_infos_ = row_infos
        self.mod_infos_ = mod_infos
        self.row_dist_ = row_dist
        self.mod_dist_ = mod_dist
        self.mod_stats_ = np.array(mod_stats)
        self.normalized_data_ = Z
        self.mod_labels_ = dummies.columns.tolist()
        self.short_labels_ = list([x.split("_",1)[-1] for x in dummies.columns])

        #########################################################################################################
        res = PCA(normalize=False,
                  n_components=self.n_components_,
                  row_labels=Z.index,
                  row_weights=self.row_weights_.tolist(),
                  col_labels=Z.columns).fit(Z)

        ########### Store all informations
        self.svd_ = res.svd_

        # Eigen - values
        eigen_values = res.svd_["vs"][:min(self.n_rows_-1, Z.shape[1]-self.n_vars_)]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        ###################################################################################
        # Store all informations 
        self.eig_ = np.array([eigen_values,difference,proportion,cumulative])
        
        #######
        self.dim_index_ = ["Dim."+str(x+1) for x in range(self.n_components_)]
        
        ################################################################################
        ################################# Coordinates ##################################
        ################################################################################
        # Row coordinates
        row_coord = res.row_coord_[:,:self.n_components_]
        # Coordinates for quantitatives columns
        var_mod_coord = pd.DataFrame(res.col_coord_[:,:self.n_components_],index=Z.columns.tolist(),columns=self.dim_index_)
        col_coord = var_mod_coord.loc[self.quanti_labels_,:]
        col_coord = np.array(col_coord)
        
        # Extract categories coordinates form PCA
        pca_coord_mod = var_mod_coord.loc[self.mod_labels_,:]
        
        # Compute True categories coordinates
        p_k = dummies.mean(axis=0)
        mod_coord = mapply(pca_coord_mod,lambda x : x*np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
        mod_coord = pd.concat(((mod_coord.loc[k,:]/np.sqrt(p_k[k])).to_frame(k).T for k in dummies.columns.tolist()),axis=0)
        
        ####################################### Contributions ##############################
        # Contributions of individuals
        row_contrib = np.apply_along_axis(func1d=lambda x : (x**2)*self.row_weights_,axis=0,arr=row_coord)
        row_contrib = np.apply_along_axis(func1d=lambda x : 100*x/self.eig_[0][:self.n_components_],axis=1,arr=row_contrib)
        
        # Contributions of variables
        col_contrib = np.apply_along_axis(func1d=lambda x : (x**2)*self.col_weights_,axis=0,arr=col_coord)
        col_contrib = np.apply_along_axis(func1d=lambda x : 100*x/self.eig_[0][:self.n_components_],axis=1,arr=col_contrib)

        # Contribution of categories
        mod_contrib = mapply(pca_coord_mod,lambda x : 100*(x**2)/self.eig_[0][:self.n_components_],axis = 1,progressbar=False,n_workers=self.n_workers_)

        ####################################### Cos2 ###########################################
        # Cos2 of individuals
        row_cos2 = np.apply_along_axis(func1d=lambda x : (x**2)/row_disto,axis=0,arr=row_coord)

        # Cos2 of continouos variables
        col_disto = np.apply_along_axis(func1d=lambda x : (x**2)*self.row_weights_,axis=0,arr=Z1).sum(axis=0)
        col_var  = np.apply_along_axis(func1d=lambda x : x/np.sqrt(col_disto),axis=0,arr=col_coord)
        col_cos2 = np.apply_along_axis(func1d=lambda x : x**2,axis=0,arr=col_var)

        # Cos2 of categories
        mod_cos2 = mapply(mod_coord,lambda x : (x**2)/mod_disto, axis=0,progressbar=False,n_workers=self.n_workers_)

        ####################################   Add elements ###############################################
        # Correlation between continouos variables and factors
        col_cor = weightedcorrcoef(x=X_quant,y=row_coord,w=self.row_weights_)[:self.n_cols_,self.n_cols_:]

        # Vtest of categories
        mod_vtest = pd.concat(((mod_coord.loc[k,:]*np.sqrt(((self.n_rows_-1)*n_k[k])/(self.n_rows_-n_k[k]))).to_frame(k).T for k in dummies.columns.tolist()),axis=0)
        mod_vtest = mapply(mod_vtest,lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)

        #### Qualitatives eta2
        quali_eta2 = pd.concat((function_eta2(X=X_qual,
                                              lab=col,
                                              x=row_coord,
                                              weights=self.row_weights_,
                                              n_workers=self.n_workers_) for col in X_qual.columns.tolist()),axis=0)
        
        # Contributions des variables qualitatives
        quali_contrib = mapply(quali_eta2,lambda x : 100*x/self.eig_[0][:self.n_components_],axis=1,progressbar=False,n_workers=self.n_workers_)

        # Cosinus carrés des variables qualitatives
        nb_mod = pd.Series([len(np.unique(X_qual[[col]])) for col in self.quali_labels_],index=self.quali_labels_,name="count")
        quali_cos2 = pd.concat(((quali_eta2.loc[cols,:]/nb_mod[cols]).to_frame(name=cols).T for cols in quali_eta2.index),axis=0)

        #######################################################################################################################################
        # Row informations
        self.row_coord_     = res.row_coord_
        self.row_contrib_   = row_contrib
        self.row_cos2_      = row_cos2
        self.res_row_dist_  = squareform(pdist(self.row_coord_,metric="sqeuclidean"))

        # Quantitatives informations
        self.col_coord_     = np.array(col_coord)
        self.col_cos2_      = np.array(col_cos2)
        self.col_contrib_   = np.array(col_contrib)
        self.col_cor_       = col_cor
        self.col_labels_    = self.quanti_labels_

        # Modality informations
        self.pca_coord_mod_ = np.array(pca_coord_mod)
        self.mod_coord_ = np.array(mod_coord)
        self.mod_cos2_ = np.array(mod_cos2)
        self.mod_contrib_ = np.array(mod_contrib)
        self.mod_vtest_ = np.array(mod_vtest)

        # Information sur les variables qualitatives
        self.var_mod_coord_ = np.array(var_mod_coord)
        self.var_mod_labels_ = Z.columns.tolist()
        self.quali_eta2_ = np.array(quali_eta2)
        self.quali_cos2_ = np.array(quali_cos2)
        self.quali_contrib_ = np.array(quali_contrib)

        self.model_ = "famd"

    def _compute_row_sup_stats(self,X):
        """Compute supplementary individuals coordinates

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

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        X_sup_quant = X[self.quanti_labels_]
        X_sup_qual = X[self.quali_labels_]

        # Standardize continuous data
        Z1 = (X_sup_quant - self.means_)/self.std_

        # Standardscale Categorical Variable
        n_rows = X_sup_qual.shape[0]
        n_cols = len(self.mod_labels_)
        Y = np.zeros((n_rows,n_cols))
        for i in np.arange(0,n_rows,1):
            values = [self.quali_labels_[k] +"_"+str(X_sup_qual.iloc[i,k]) for k in np.arange(0,len(self.quali_labels_))]
            for j in np.arange(0,n_cols,1):
                if self.mod_labels_[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index.tolist())

        # New normalized Data
        Z2 = mapply(row_sup_dummies,lambda x : (x - self.dummies_means_)/self.dummies_std_,axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary individuals coordinates
        row_sup_coord = np.dot(pd.concat([Z1,Z2],axis=1),self.svd_["V"])

        # Supplementary individuals distance to inertia
        row_sup_disto = (np.sum((Z1**2)*self.col_weights_,axis=1)+
                         mapply(row_sup_dummies,lambda x:np.sum(1/self.dummies_means_.values*(x-self.dummies_means_.values)**2),
                                axis=1,progressbar=False,n_workers=self.n_workers_))
        row_sup_disto.name = "dist"
        ############# Supplementary Cos2
        row_sup_cos2 = np.apply_along_axis(func1d=lambda x : (x**2)/(row_sup_disto),axis=0,arr=row_sup_coord)

        # Save
        self.row_sup_coord_ = row_sup_coord[:,:self.n_components_]
        self.row_sup_disto_ = np.sqrt(np.array(row_sup_disto))
        self.row_sup_cos2_ = row_sup_cos2[:,:self.n_components_]

        return {"coord" : pd.DataFrame(row_sup_coord[:,:self.n_components_],index=X.index.tolist(),columns=self.dim_index_),
                "dist"  : np.sqrt(np.array(row_sup_disto)),
                "cos2"  : pd.DataFrame(row_sup_cos2[:,:self.n_components_],index=X.index.tolist(),columns=self.dim_index_)}

    def _compute_quanti_sup_stats(self,X,y=None):
        """Comupute supplementary continuous variables statistics

        Parameters
        ----------
        self    :   An instance of class FAMD
        X       :   DataFrame (n_rows,n_columns)
        y : None
            y is ignored

        Return
        ------
        col_sup_corr_   : Pearson correlation between new continuous variables and old continuous variables
        col_sup_coord_  :   Supplementary continuous coordinates
        col_sup_cos2_   :   Supplementary continuous cosines
        col_sup_ftest_  :   Supplementary continuous Fisher - test
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        ##############################
        X = X.astype("float")
        ##################### Compute statistics
        self.summary_quanti_.insert(0,"group","active")
        summary_quanti_sup = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
        summary_quanti_sup.insert(0,"group","sup")
        self.summary_quanti_ = pd.concat([self.summary_quanti_,summary_quanti_sup],axis=0,ignore_index=True)

        n_sup_cols = X.shape[1]
        # Weighted Correlation between New continuous variables and old continuous variables
        col_sup_corr = weightedcorrcoef(x=X,y=self.quanti_data_,w=self.row_weights_)[:n_sup_cols,n_sup_cols:]

        # Supplementary weighted continuous coordinates
        col_sup_coord = weightedcorrcoef(x=X,y=self.row_coord_,w=self.row_weights_)[:n_sup_cols,n_sup_cols:]
       
        # Supplementary continuous cos2
        col_sup_cos2 = np.apply_along_axis(func1d=lambda x : x**2,arr = col_sup_coord,axis=0)

        # Store supplementary continuous informations
        self.col_sup_corr_  =   col_sup_corr
        self.col_sup_coord_ =   col_sup_coord[:,:self.n_components_]
        self.col_sup_cos2_  =   col_sup_cos2[:,:self.n_components_]

        # Self
        self.col_sup_labels_ = X.columns.tolist()

        return {"corr"     :   pd.DataFrame(col_sup_corr, index=X.columns,columns=self.col_labels_),
                "coord"    :   pd.DataFrame(col_sup_coord[:,:self.n_components_],index=X.columns.tolist(),columns=self.dim_index_),
                "cos2"     :   pd.DataFrame(col_sup_cos2[:,:self.n_components_], index = X.columns.tolist(),columns=self.dim_index_)}

    def _compute_quali_sup_stats(self,X,y=None):
        """Compute statistics supplementary categorical variables

        Parameters
        ----------
        self    :   An instance of class FAMD
        X       :   DataFrame (n_rows,n_columns)
        y : None
            y is ignored

        Return
        ------
        chi2_sup_test_  : chi-squared test
        mod_sup_coord_  : Supplementary categories coordinates
        mod_sup_cos2_   : Supplementary categories cosines
        mod_sup_disto_  : Supplementary categories distance
        mod_sup_stats_  : Statistic for supplementary categories (count and percentage)
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Chi-squared test between new categorie
        if X.shape[1] > 1:
            chi_sup_stats = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            cpt = 0
            for i in range(X.shpe[1]-1):
                for j in range(i+1,X.shape[1]):
                    tab = pd.crosstab(X.iloc[:,i],X.iloc[:,j])
                    chi = st.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X.columns.tolist()[i],
                                "variable2" : X.columns.tolist()[j],
                                "statistic" : chi.statistic,
                                "dof"       : chi.dof,
                                "pvalue"    : chi.pvalue},index=[cpt])
                    chi_sup_stats = pd.concat([chi_sup_stats,row_chi2],axis=0)
                    cpt = cpt + 1
        
        # Chi-squared between old and new qualitatives variables
        chi_sup_stats2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
        cpt = 0
        for i in range(X.shape[1]):
            for j in range(self.n_vars_):
                tab = pd.crosstab(X.iloc[:,i],self.quali_data_.iloc[:,j])
                chi = st.chi2_contingency(tab,correction=False)
                row_chi2 = pd.DataFrame({"variable1" : X.columns.tolist()[i],
                                         "variable2" : self.quali_labels_[j],
                                         "statistic" : chi.statistic,
                                         "dof"       : chi.dof,
                                         "pvalue"    : chi.pvalue},index=[cpt])
                chi_sup_stats2 = pd.concat([chi_sup_stats2,row_chi2],axis=0,ignore_index=True)
                cpt = cpt + 1
        
        ###### Add 
        if X.shape[1] > 1 :
            chi_sup_stats = pd.concat([chi_sup_stats,chi_sup_stats2],axos=0,ignore_index=True)
        else:
            chi_sup_stats = chi_sup_stats2
        

        #################################### Summary quali
        # Compute statistiques
        summary_quali_sup = pd.DataFrame()
        for col in X.columns.tolist():
            eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
        summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
        summary_quali_sup.insert(0,"group","sup")

        #########
        self.summary_quali_.insert(0,"group","active")
        self.summary_quali_ = pd.concat([self.summary_quali_,summary_quali_sup],axis=0,ignore_index=True)

        ###################################################################################################
        #### Qualitatives eta2
        quali_sup_eta2 = pd.concat((function_eta2(X=X,
                                              lab=col,
                                              x=self.row_coord_,
                                              weights=self.row_weights_,
                                              n_workers=self.n_workers_) for col in X.columns.tolist()),axis=0)
        # Cosinus carrés des variables qualitatives supplémentaires
        nb_mod = pd.Series([len(np.unique(X[[col]])) for col in X.columns],index=X.columns,name="count")
        quali_sup_cos2 = pd.concat(((quali_sup_eta2.loc[cols,:]/nb_mod[cols]).to_frame(name=cols).T for cols in quali_sup_eta2.index),axis=0)
        
        ##### Dummies
        dummies = pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols in X.columns),axis=1)
        mod_sup_stats = dummies.agg(func=[np.sum,np.mean]).T
        n_k = dummies.sum(axis=0)
        p_k = dummies.mean(axis=0)
        mod_sup_labels = dummies.columns.tolist()
        short_sup_labels = list([x.split("_",1)[-1] for x in mod_sup_labels])

        # Supplementary categories coordinates
        mod_sup_coord = pd.concat((pd.concat((pd.DataFrame(self.row_coord_,index=self.row_labels_,
                                                   columns=self.dim_index_),dummies[cols]),axis=1)
                                      .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)

        # Supplementary categories v-test
        mod_sup_vtest = pd.concat(((mod_sup_coord.loc[k,:]/np.sqrt((self.n_rows_-n_k[k])/((self.n_rows_-1)*n_k[k]))).to_frame(name=k).T for k in mod_sup_coord.index),axis=0)
        mod_sup_vtest = mapply(mod_sup_vtest,lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)

        # Moyennes conditionnelles sur la variable Z
        mz_g = pd.concat((pd.concat((self.normalized_data_,dummies[cols]),axis=1)
                                    .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)

        # Distance des modalités à  l'origine
        mod_sup_disto = mapply(mz_g,lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary categories cos2
        mod_sup_cos2 = mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_).div(mod_sup_disto,axis="index")

        # Supplementary categories informations
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_disto_     =   np.array(mod_sup_disto)
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)

        self.mod_sup_labels_    =   mod_sup_labels
        self.short_sup_labels_  =   short_sup_labels

        # Categorical variables
        self.quali_sup_eta2_    =   np.array(quali_sup_eta2)
        self.quali_sup_cos2_    =   np.array(quali_sup_cos2)

        return {"coord"    :   pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
                "cos2"     :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_),
                "dist"     :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=["dist"]),
                "eta2"     :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
                "vtest"    :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)}

    def transform(self,X,y=None):
        """Apply the dimensionality reduction on X

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
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Store continuous and categorical variables
        X_sup_quant = X[self.quanti_labels_]
        X_sup_qual = X[self.quali_labels_]

        # Standardscaler numerical variable
        Z1 = (X_sup_quant - self.means_)/self.std_

        # Standardscaler categorical Variable
        n_rows = X_sup_qual.shape[0]
        n_cols = len(self.mod_labels_)
        Y = np.zeros((n_rows,n_cols))
        for i in np.arange(0,n_rows,1):
            values = [self.quali_labels_[k] +"_"+str(X_sup_qual.iloc[i,k]) for k in np.arange(0,len(self.quali_labels_))]
            for j in np.arange(0,n_cols,1):
                if self.mod_labels_[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index.tolist())

        # New normalized data
        Z2 = mapply(row_sup_dummies,lambda x : (x - self.dummies_means_)/self.dummies_std_,axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary individuals coordinates
        row_sup_coord = np.dot(np.array(pd.concat([Z1,Z2],axis=1)),self.svd_["V"])

        return  row_sup_coord[:,:self.n_components_]

    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

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
        return self.row_coord_

######################################################################################################
#               Multiple Factor Analysis (MFA)
#####################################################################################################

# https://husson.github.io/MOOC_AnaDo/AFM.html
# https://math.institut-agro-rennes-angers.fr/fr/ouvrages/analyse-factorielle-multiple-avec-r
# https://eudml.org/subject/MSC/62H25

class MFA(BaseEstimator,TransformerMixin):
    """Multiple Factor Analysis (MFA)

    Performs Multiple Factor Analysis

    Parameters:
    ----------
    normalize :
    n_components :

    group : list of string

    group : list of string

    group_sup : list of string

    row_labels : list of string

    row_sup_labels : lits of string

    row_weight : list

    col_weight_mfa : dict

    parallelize : 

    Return
    ------



    """
    def __init__(self,
                 n_components=5,
                 group=None,
                 group_sup = None,
                 row_labels = None,
                 row_sup_labels = None,
                 row_weights = None,
                 col_weights_mfa = None,
                 parallelize=False):
        self.n_components = n_components
        self.group = group
        self.group_sup = group_sup
        self.row_labels = row_labels
        self.row_sup_labels = row_sup_labels
        self.row_weights = row_weights
        self.col_weights_mfa = col_weights_mfa
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """

        """

        # Check if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        ######## Check if columns is level 2
        if X.columns.nlevels != 2:
            raise ValueError("Error : X must have a MultiIndex columns with 2 levels.")

         # set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Check if groups is None
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")

        ##### Extract supplementary rows in dataframe
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Remove supplementary group
        if self.group_sup is not None:
            diff = [i for i in self.group + self.group_sup if i not in self.group or i not in self.group_sup]
            if len(diff)==0:
                raise ValueError("Error : ")
            else:
                Xsup = _X[self.group_sup]
                X_ = _X[self.group]
        else:
            X_ = _X

        # Save data
        self.data_ = X
        self.active_data_ = X_

        ############################## Initialise elements ################################
        self.mod_labels_ = None

        # Compute stats
        self._compute_stats(X_)

        # Compute columns supplementary coordinates
        if self.group_sup is not None:
            self._compute_groups_sup_coord(X=Xsup)


        return self

    def _compute_stats(self,X):
        """

        """

        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        # Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist())

        # Shape of X
        self.n_rows_, self.n_cols_= X.shape

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]
        
        ########### Set row weight and columns weight
        # Set row weight
        if self.row_weights is None:
            self.row_weights_ = np.ones(self.n_rows_)/self.n_rows_
        elif not isinstance(self.row_weights,list):
            raise ValueError("Error : 'row_weights' must be a list of row weight.")
        elif len(self.row_weights) != self.n_rows_:
            raise ValueError(f"Error : 'row_weights' must be a list with length {self.n_rows_}.")
        else:
            self.row_weights_ = np.array([x/np.sum(self.row_weights) for x in self.row_weights])

        # Checks groups are provided
        self.group_ = self._determine_groups(X=X,groups=self.group)

        # Check group types are consistent
        self.all_nums_ = {}
        self.all_cats_ = {}
        for grp, cols in self.group_.items():
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError(f"Not all columns in '{grp}' group are of the same type. Used HMFA function instead.")
            self.all_nums_[grp] = all_num
            self.all_cats_[grp] = all_cat
        
        ############################# Set columns weight MFA
        self.col_weights_mfa_ = self.col_weights_mfa
        if self.col_weights_mfa_ is None:
            self.col_weights_mfa_ = {}
            for grp, cols in self.group_.items():
                if self.all_nums_[grp]:
                    self.col_weights_mfa_[grp] = np.ones(len(cols))
        else:
            self.col_weights_mfa_ = {}
            for grp, cols in self.group_.items():
                if self.all_nums_[grp]:
                    self.col_weights_mfa_[grp] = np.array(self.col_weights_mfa[grp])
        
        # Number of components
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = min(self.n_rows_-1,self.n_cols_)
        else:
            self.n_components_ = min(self.n_components_,self.n_rows_-1,self.n_cols_)

        # Run a Factor Analysis in each group
        col_labels       = []
        col_group_labels = []
        var_labels       = []
        var_group_labels = []
        model            = {}
        for grp, cols in self.group_.items():
            Xg = X[grp]
            if self.all_nums_[grp]:
                # Principal Components Anlysis (PCA)
                fa = PCA(normalize=True,
                         n_components=self.n_components_,
                         row_labels=self.row_labels_,
                         row_weights=self.row_weights_.tolist(),
                         col_weights=self.col_weights_mfa_[grp].tolist(),
                         col_labels=Xg.columns.tolist(),
                         parallelize=self.parallelize)
                # Set col labels name
                col_labels = col_labels + Xg.columns.tolist()
                # Set group labels name
                col_group_labels = col_group_labels + [grp]*Xg.shape[1]
            elif self.all_cats_[grp]:
                # Multiple Correspondence Analysis (MCA)
                fa = MCA(n_components=self.n_components_,
                         row_labels=self.row_labels_,
                         var_labels=Xg.columns.tolist(),
                         parallelize=self.parallelize)
                # Set variables labels
                var_labels = var_labels + Xg.columns.tolist()
                # Set group variables labels
                var_group_labels = var_group_labels + [grp]*Xg.shape[1]
            # Fit the model
            model[grp] = fa.fit(Xg)
        
        ##################### Compute group disto
        group_disto = np.array([np.sum(model[grp].eig_[0]**2)/model[grp].eig_[0][0]**2 for grp, cols in self.group_.items()])
        group_disto = pd.Series(group_disto,index=[grp for grp,cols in self.group_.items()],name="dist2")

        ########
        self.separate_analyses_ = model
        self.col_labels_        = col_labels
        self.col_group_labels_  = col_group_labels
        self.var_labels_        = var_labels
        self.var_group_labels_  = var_group_labels

        # Normalize data
        means      = {}
        std        = {}
        base       = pd.DataFrame().astype("float")
        col_weights = pd.Series().astype("float")
        for grp,cols in self.group_.items():
            Xg = X[grp]
            # All variables in group are numericals
            if self.all_nums_[grp]:
                ############################### Compute Mean and Standard deviation #################################
                d1 = DescrStatsW(Xg.values,weights=self.row_weights_,ddof=0)
                # Compute Mean
                means_ = d1.mean.reshape(1,-1)
                # Compute Sandard Error
                std_ = d1.std.reshape(1,-1)
                ########################### Concatenate #################################################################################
                Z = (Xg - means_)/std_
                ###################" Concatenate
                base = pd.concat([base,Z],axis=1)
                ##################################"
                std[grp] = std_
                means[grp] = means_
                ################################ Col weight
                weights = pd.Series(np.repeat(a=1/model[grp].eig_[0][0],repeats=len(cols)),index=Xg.columns.tolist())
                # Ajout de la pondération de la variable
                weights = weights*self.col_weights_mfa_[grp]
                col_weights = pd.concat((col_weights,weights),axis=0)
            # All variables in group are categoricals
            elif self.all_cats_[grp]:
                # Dummies tables :  0/1
                dummies = pd.get_dummies(Xg)
                # Effectif par modalite
                I_k = dummies.sum(axis=0)
                Z = pd.concat((dummies.loc[:,k]*(self.n_rows_/I_k[k])-1 for k  in dummies.columns.tolist()),axis=1)
                # Weight of categories
                m_k = (1/(self.n_rows_*Xg.shape[1]*model[grp].eig_[0][0]))*I_k
                # Concatenate
                base = pd.concat([base,Z],axis=1)
                ############# Weighted of categories
                weights = pd.Series(m_k,index=dummies.columns.tolist())
                col_weights = pd.concat((col_weights,weights),axis=0)
            else:
                raise ValueError(f"Error : Mixed of variables in {grp} group. Used HMFA function instead.")

       # Set
        self.mean_ = means
        self.std_  = std

        # Set columns weights
        self.col_weights_ = col_weights

        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        global_pca = PCA(normalize = False,
                         n_components = self.n_components_,
                         row_labels = base.index.tolist(),
                         col_labels = base.columns.tolist(),
                         row_weights = self.row_weights_.tolist(),
                         col_weights = self.col_weights_.tolist(),
                         parallelize = self.parallelize).fit(base)

        ############################################# Removing duplicate value in cumulative percent #######################"
        cumulative = sorted(list(set(global_pca.eig_[3])))
        
        dim_index = ["Dim."+str(x+1) for x in np.arange(self.n_components_)]
        self.dim_index_ = dim_index

        # Global Principal Components Analysis (PCA)
        self.global_pca_ = global_pca

        ####################################################################################################
        #
        ####################################################################################################
        # Global
        self.global_pca_normalized_data_ = global_pca.normalized_data_

        ####################################################################################################
        #   Eigen values informations
        ##################################################################################################
        # Store all informations
        self.eig_ = global_pca.eig_[:,:len(cumulative)]

        # Eigenvectors
        self.eigen_vectors_ = global_pca.eigen_vectors_[:,:self.n_components_]

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################

        # Row coordinates
        self.row_coord_ = global_pca.row_coord_[:,:self.n_components_]

        # Row contributions
        self.row_contrib_ = global_pca.row_contrib_[:,:self.n_components_]

        # Row - Quality of representation
        self.row_cos2_ = global_pca.row_cos2_[:,:self.n_components_]

        ##########################################################################################################
        #####  Coordonnées des colonnes : Variables continues/Modalités des variables qualitatives
        ##########################################################################################################
        # Continues
        col_coord      = pd.DataFrame().astype("float")
        col_contrib    = pd.DataFrame().astype("float")
        col_cos2       = pd.DataFrame().astype("float")
        summary_quanti = pd.DataFrame().astype("float")
        # Categories
        mod_coord      = pd.DataFrame().astype("float")
        mod_contrib    = pd.DataFrame().astype("float")
        mod_disto      = pd.DataFrame().astype("float")
        mod_cos2       = pd.DataFrame().astype("float")
        mod_vtest      = pd.DataFrame().astype("float")
        quali_eta2     = pd.DataFrame().astype("float")
        summary_quali  = pd.DataFrame().astype("float")

        # If all columns in Data are numerics
        if all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist()):
            # Make a copy
            X_nums = X.copy()
            X_nums.columns = X_nums.columns.droplevel()

            ###################################################################################################
            ################## Compute statistiques
            stats = X_nums.describe().T
            stats = stats.reset_index().rename(columns={"index" : "variable"})
            stats.insert(0,"group",[x[0] for x in X.columns.tolist()])
            stats["count"] = stats["count"].astype("int")
            summary_quanti = pd.concat([summary_quanti,stats],axis=0,ignore_index=True)

            ####################################################################################################
            # Correlation between variables en axis
            coord = np.corrcoef(X_nums.values,self.row_coord_,rowvar=False)[:X_nums.shape[1],X_nums.shape[1]:]
            coord = pd.DataFrame(coord,index=X_nums.columns.tolist(),columns=self.dim_index_)
            col_coord = pd.concat([col_coord,coord],axis=0)

            ####################################################################################################
            # Contribution
            contrib = pd.DataFrame(global_pca.col_contrib_[:,:self.n_components_],index=X_nums.columns.tolist(),columns=self.dim_index_)
            col_contrib = pd.concat([col_contrib,contrib],axis=0)

            ###################################################################################################
            # Cos2
            cos2 = pd.DataFrame(global_pca.col_cos2_[:,:self.n_components_],index=X_nums.columns.tolist(),columns=self.dim_index_)
            col_cos2 = pd.concat([col_cos2,cos2],axis=0)
        # If all columns are categoricals
        elif all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            # Make a copy of original Data
            X_cats = X.copy()
            X_cats.columns = X_cats.columns.droplevel()

            ###################################################################################################
            # Compute statisiques
            stats = pd.DataFrame()
            for col_grp in X.columns.tolist():
                grp, col = col_grp
                eff = X_cats[col].value_counts().to_frame("effectif").reset_index().rename(columns={col : "modalite"})
                eff.insert(0,"variable",col)
                eff.insert(0,"group",grp)
                stats = pd.concat([stats,eff],axis=0,ignore_index=True)
            summary_quali = pd.concat([summary_quali,stats],axis=0,ignore_index=True)

            ######################################################################################################################
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X_cats[col],prefix=col,prefix_sep='_') for col in X_cats.columns.tolist()),axis=1)
            n_k = dummies.sum(axis=0)
            p_k = dummies.mean(axis=0)

            ############################################################################################################################
            # Compute categories coordinates
            coord = pd.concat((pd.concat((pd.DataFrame(self.row_coord_[:,:self.n_components_],index=self.row_labels_,columns=self.dim_index_),dummies[col]),axis=1)
                                    .groupby(col)
                                    .mean().iloc[1,:]
                                    .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
            mod_coord = pd.concat([mod_coord,coord],axis=0)
            print(mod_coord)

            ###############################################################################################################
            # v-test
            vtest = mapply(mapply(coord,lambda x : x/np.sqrt((self.n_rows_- n_k)/((self.n_rows_-1)*n_k)),axis=0,progressbar=False,n_workers=self.n_workers_),
                           lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
            mod_vtest = pd.concat([mod_vtest,vtest],axis=0)

            ##############################################################################################################
            ######### Contribution
            contrib = pd.DataFrame(self.global_pca_.col_contrib_[:,:self.n_components_],index=self.global_pca_.col_labels_,columns=self.dim_index_)[self.dim_index_]
            print(contrib)
            contrib.index = contrib.index.droplevel()
            mod_contrib = pd.concat([mod_contrib,contrib],axis=0)

            #########################################################################################################################
            ##################### Conditionnal mean using Standardize data
            Z = pd.DataFrame(self.global_pca_.normalized_data_,columns=self.global_pca_.col_labels_,index=self.global_pca_.row_labels_)
            Z_coord = pd.concat((pd.concat((Z,dummies[col]),axis=1)
                                .groupby(col)
                                .mean().iloc[1,:]
                                .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
            # Distance au carré
            disto = mapply(Z_coord,lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_)
            mod_disto = pd.concat([mod_disto,disto.to_frame("dist")],axis=0)

            ################## Cos2
            cos2 = mapply(coord,lambda x : x**2/disto.values,axis=0,progressbar=False,n_workers=self.n_workers_)
            mod_cos2 = pd.concat([mod_cos2,cos2],axis=0)

            ############################################################################################################
            ############## Correlation ratio
            var_eta2 = pd.concat(((mapply(coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_)
                                     .mul(p_k,axis="index")
                                     .loc[filter(lambda x: x.startswith(col),coord.index.tolist()),:]
                                     .sum(axis=0).to_frame(name=col).T.div(self.eig_[0][:self.n_components_])) for col in X_cats.columns.tolist()),axis=0)
            quali_eta2 = pd.concat([quali_eta2,var_eta2],axis=0)
        else:
            for grp, cols in self.group_.items():
                Xg = X[grp]
                if all(pd.api.types.is_numeric_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    ###################################################################################################
                    ################## Compute statistiques
                    stats = Xg.describe().T
                    stats = stats.reset_index().rename(columns={"index" : "variable"})
                    stats.insert(0,"group",[grp]*len(cols))
                    stats["count"] = stats["count"].astype("int")
                    summary_quanti = pd.concat([summary_quanti,stats],axis=0,ignore_index=True)

                    ####################################################################################################
                    ########## Correlation between variables en axis
                    ###################################################################################################
                    coord = np.corrcoef(Xg.values,self.row_coord_,rowvar=False)[:Xg.shape[1],Xg.shape[1]:]
                    coord = pd.DataFrame(coord,index=Xg.columns.tolist(),columns=self.dim_index_)
                    col_coord = pd.concat([col_coord,coord],axis=0)

                    ###################################################################################################
                    #   Contributions
                    ###################################################################################################
                    # Extract contributions from global PCA
                    contrib = pd.DataFrame(self.global_pca_.col_contrib_[:,:self.n_components_],columns=self.dim_index_,
                                           index=self.global_pca_.col_labels_)
                    contrib.index = contrib.index.droplevel()
                    col_contrib = pd.concat([col_contrib,contrib.loc[Xg.columns.tolist(),:]],axis=0)

                    ###################################################################################################
                    #   Cos2
                    ###################################################################################################
                     # Extract cos2 from global PCA
                    cos2 = pd.DataFrame(self.global_pca_.col_cos2_[:,:self.n_components_],columns=self.dim_index_,
                                        index=self.global_pca_.col_labels_)
                    cos2.index = cos2.index.droplevel()
                    col_cos2 = pd.concat([col_cos2,cos2.loc[Xg.columns.tolist(),:]],axis=0)
                elif all(pd.api.types.is_string_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    ###################################################################################################
                    # Compute statisiques
                    stats = pd.DataFrame()
                    for col in Xg.columns.tolist():
                        eff = Xg[col].value_counts().to_frame("effectif").reset_index().rename(columns={col : "modalite"})
                        eff.insert(0,"variable",col)
                        eff.insert(0,"group",grp)
                        stats = pd.concat([stats,eff],axis=0,ignore_index=True)
                    summary_quali = pd.concat([summary_quali,stats],axis=0,ignore_index=True)

                    ######################################################################################################################
                    # Compute Dummies table : 0/1
                    dummies = pd.concat((pd.get_dummies(Xg[col],prefix=col,prefix_sep='_') for col in Xg.columns.tolist()),axis=1)
                    n_k = dummies.sum(axis=0)
                    p_k = dummies.mean(axis=0)

                    ############################################################################################################################
                    # Compute categories coordinates
                    coord = pd.concat((pd.concat((pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_),dummies[col]),axis=1)
                                            .groupby(col)
                                            .mean().iloc[1,:]
                                            .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
                    mod_coord = pd.concat([mod_coord,coord],axis=0)

                    ###############################################################################################################
                    # v-test
                    vtest = mapply(mapply(coord,lambda x : x/np.sqrt((self.n_rows_- n_k)/((self.n_rows_-1)*n_k)),axis=0,progressbar=False,n_workers=self.n_workers_),
                                lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
                    mod_vtest = pd.concat([mod_vtest,vtest],axis=0)

                    ###################################################################################################
                    #   Contributions
                    ###################################################################################################
                    # Extract contributions from global PCA
                    contrib = pd.DataFrame(self.global_pca_.col_contrib_[:,:self.n_components_],columns=self.dim_index_,
                                           index=self.global_pca_.col_labels_)
                    contrib.index = contrib.index.droplevel()
                    mod_contrib = pd.concat([mod_contrib,contrib.loc[dummies.columns.tolist(),:]],axis=0)

                    #########################################################################################################################
                    ##################### Conditionnal mean using Standardize data
                    Z = pd.DataFrame(self.global_pca_.normalized_data_,columns=self.global_pca_.col_labels_,index=self.global_pca_.row_labels_)
                    Z_coord = pd.concat((pd.concat((Z,dummies[col]),axis=1)
                                        .groupby(col)
                                        .mean().iloc[1,:]
                                        .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
                    # Distance au carré
                    disto = mapply(Z_coord,lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_)
                    mod_disto = pd.concat([mod_disto,disto.to_frame("dist")],axis=0)

                    ################## Cos2
                    cos2 = mapply(coord,lambda x : x**2/disto.values,axis=0,progressbar=False,n_workers=self.n_workers_)
                    mod_cos2 = pd.concat([mod_cos2,cos2],axis=0)

                    ############################################################################################################
                    ############## Correlation ratio
                    var_eta2 = pd.concat(((mapply(coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_)
                                            .mul(p_k,axis="index")
                                            .loc[filter(lambda x: x.startswith(col),coord.index.tolist()),:]
                                            .sum(axis=0).to_frame(name=col).T.div(self.eig_[0][:self.n_components_])) for col in Xg.columns.tolist()),axis=0)
                    quali_eta2 = pd.concat([quali_eta2,var_eta2],axis=0)

        # Set
        self.col_coord_   = col_coord.iloc[:,:].values
        self.col_contrib_ = col_contrib.iloc[:,:].values
        self.col_cos2_    = col_cos2.iloc[:,:].values
        self.col_cor_     = col_coord.iloc[:,:].values
        # categories
        self.mod_coord_   = mod_coord
        self.mod_contrib_ = mod_contrib
        self.mod_disto_   = mod_disto
        self.mod_cos2_    = mod_cos2
        self.mod_vtest_   = mod_vtest
        self.quali_eta2_  = quali_eta2

        self.summary_quanti_ = summary_quanti
        self.summary_quali_  = summary_quali

        ####################################################################################################
        #   Partiel Row Coordinates
        ####################################################################################################
        # Partiel row coordinates
        self.row_coord_partiel_ = self._row_coord_partiel(X=X)

        #################################################################################################
        ##### Categories partiel coordinates
        mod_coord_partiel = pd.DataFrame().astype("float")
        if all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            for grp, cols in self.group_.items():
                # Make a copy of original Data
                X_cats = X.copy()
                X_cats.columns = X_cats.columns.droplevel()
                ######################################################################################################################
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X_cats[col],prefix=col,prefix_sep='_') for col in X_cats.columns.tolist()),axis=1)
                ############################################################################################################################
                # Compute categories coordinates
                coord_partiel = pd.concat((pd.concat((self.row_coord_partiel_[grp],dummies[col]),axis=1)
                                        .groupby(col)
                                        .mean().iloc[1,:]
                                        .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
                coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partiel.columns.tolist()])
                mod_coord_partiel = pd.concat([mod_coord_partiel,coord_partiel],axis=1)
        else:
            for grp, cols in self.group_.items():
                if self.all_cats_[grp]:
                    # Make a copy of original Data
                    X_cats = X.loc[:,cols][grp]
                    #######################################################################################################################
                    # Compute Dummies table : 0/1
                    dummies = pd.concat((pd.get_dummies(X_cats[col],prefix=col,prefix_sep='_') for col in X_cats.columns.tolist()),axis=1)
                    ############################################################################################################################
                    # Compute categories coordinates
                    for grp2, cols2 in self.group_.items():
                        coord_partiel = pd.concat((pd.concat((self.row_coord_partiel_[grp2],dummies[col]),axis=1)
                                            .groupby(col)
                                            .mean().iloc[1,:]
                                            .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
                        coord_partiel.columns = pd.MultiIndex.from_tuples([(grp2,col) for col in coord_partiel.columns.tolist()])
                        mod_coord_partiel = pd.concat([mod_coord_partiel,coord_partiel],axis=1)

        self.mod_coord_partiel_ = mod_coord_partiel

        ##################################################################################################
        #   Partial axes informations
        #################################################################################################

        ########################################### Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in self.group_.items():
            data = self.separate_analyses_[grp].row_coord_
            correl = np.corrcoef(self.row_coord_,data,rowvar=False)[:self.n_components_,self.n_components_:]
            coord = pd.DataFrame(correl,index=self.dim_index_,columns=self.separate_analyses_[grp].dim_index_)
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        ############################################## Partial axes cos2
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=self.n_workers_)

        #########" Partial correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in self.group_.items():
            data = pd.DataFrame(self.separate_analyses_[grp].row_coord_,index=self.separate_analyses_[grp].row_labels_,
                                columns=self.separate_analyses_[grp].dim_index_)
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        #################################### Partial axes contrib ################################################"
        axes_contrib = pd.DataFrame().astype("float")
        for grp, cols in self.group_.items():
            nbcol = min(self.n_components_,self.separate_analyses_[grp].row_coord_.shape[1])
            eig = self.separate_analyses_[grp].eig_[0][:nbcol]/self.separate_analyses_[grp].eig_[0][0]
            contrib = mapply(partial_axes_coord[grp].iloc[:,:nbcol],lambda x : (x**2)*eig,axis=1,progressbar=False,n_workers=self.n_workers_)
            contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
            axes_contrib  = pd.concat([axes_contrib,contrib],axis=1)
        
        partial_axes_contrib = mapply(axes_contrib,lambda x : 100*x/np.sum(x),axis=1,progressbar=False,n_workers=self.n_workers_)

        ###############
        self.partial_axes_coord_       = partial_axes_coord
        self.partial_axes_cor_         = partial_axes_coord
        self.partial_axes_cos2_        = partial_axes_cos2
        self.partial_axes_contrib_     = partial_axes_contrib
        self.partial_axes_cor_between_ = all_coord.corr()

        ################################################################################################"
        #    Inertia Ratios
        ################################################################################################

        #### "Between" inertia on axis s
        between_inertia = len(self.group)*np.apply_along_axis(func1d=lambda x : np.sum(x**2),axis=0,arr = self.row_coord_)

        ### Total inertial on axis s
        total_inertia = [np.sum((self.row_coord_partiel_.loc[:, (slice(None),dim)]**2).sum()) for dim in self.dim_index_]

        ### Inertia ratio
        inertia_ratio = pd.Series([between_inertia[x]/total_inertia[x] for x in range(len(self.dim_index_))],
                                  index=self.dim_index_,name = "inertia ratio")
        self.inertia_ratio_ = inertia_ratio

        ############################### Within inertia ################################################################
        row_within_inertia = pd.DataFrame(index=self.row_labels_,columns=self.dim_index_).astype("float")
        for i, dim in enumerate(self.dim_index_):
            data = mapply(self.row_coord_partiel_.loc[:, (slice(None),dim)],lambda x : (x - self.row_coord_[:,i])**2,axis=0,
                          progressbar=False,n_workers=self.n_workers_).sum(axis=1)
            row_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=self.n_workers_)

        self.row_within_inertia_ = row_within_inertia

        ######################################## Within partial inertia ################################################
        data = pd.DataFrame().astype("float")
        for i,dim in enumerate(self.dim_index_):
            data1 = mapply(self.row_coord_partiel_.loc[:, (slice(None),dim)],lambda x : (x - self.row_coord_[:,i])**2,axis=0,
                           progressbar=False,n_workers=self.n_workers_)
            data1 = 100*data1/data1.sum().sum()
            data = pd.concat([data,data1],axis=1)

        ######## Rorder inertia by group
        row_within_partial_inertia = pd.DataFrame().astype("float")
        for grp, cols in self.group_.items():
            partial_inertia = data[grp]
            partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
            row_within_partial_inertia = pd.concat([row_within_partial_inertia,partial_inertia],axis=1)

        self.row_within_partial_inertia_ = row_within_partial_inertia

        ################################################################################################################
        ############################### Modalities Within inertia ################################################################
        if all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            mod_within_inertia = pd.DataFrame(index=self.mod_labels_,columns=self.dim_index_).astype("float")
            for dim in self.dim_index_:
                data = mapply(self.mod_coord_partiel_.loc[:, (slice(None),dim)],lambda x : (x - self.mod_coord_[dim].values)**2,axis=0,
                              progressbar=False,n_workers=self.n_workers_).sum(axis=1)
                mod_within_inertia.loc[:,dim] = len(self.group_)*mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,
                                                       progressbar=False,n_workers=self.n_workers_)

            self.mod_within_inertia_ = mod_within_inertia

            ######################################## Within partial inertia ################################################
            data = pd.DataFrame().astype("float")
            for dim in self.dim_index_:
                data1 = mapply(self.mod_coord_partiel_.loc[:, (slice(None),dim)],lambda x : (x - self.mod_coord_[dim].values)**2,axis=0,
                               progressbar=False,n_workers=self.n_workers_)
                data1 = 100*data1/data1.sum().sum()
                data = pd.concat([data,data1],axis=1)

            ######## Rorder inertia by group
            mod_within_partial_inertia = pd.DataFrame().astype("float")
            for grp, cols in self.group_.items():
                partial_inertia = data[grp]
                partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
                mod_within_partial_inertia = len(self.group_)*pd.concat([mod_within_partial_inertia,partial_inertia],axis=1)
            self.mod_within_partial_inertia_ = mod_within_partial_inertia
        else:
            for grp, cols in self.group_.items():
                if self.all_cats_[grp]:
                    mod_within_inertia = pd.DataFrame(index=self.mod_labels_,columns=self.dim_index_).astype("float")
                    for dim in self.dim_index_:
                        data = mapply(self.mod_coord_partiel_.loc[:, (slice(None),dim)],lambda x : (x - self.mod_coord_[dim].values)**2,axis=0,
                                      progressbar=False,n_workers=self.n_workers_).sum(axis=1)
                        mod_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,
                                                               progressbar=False,n_workers=self.n_workers_)
                    self.mod_within_inertia_ = mod_within_inertia

                    ######################################## Within partial inertia ################################################
                    data = pd.DataFrame().astype("float")
                    for dim in self.dim_index_:
                        data1 = mapply(self.mod_coord_partiel_.loc[:, (slice(None),dim)],lambda x : (x - self.mod_coord_[dim].values)**2,axis=0,
                                       progressbar=False,n_workers=self.n_workers_)
                        data1 = 100*data1/data1.sum().sum()
                        data = pd.concat([data,data1],axis=1)
                    ######## Rorder inertia by group
                    mod_within_partial_inertia = pd.DataFrame().astype("float")
                    for grp, cols in self.group_.items():
                        partial_inertia = data[grp]
                        partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
                        mod_within_partial_inertia = pd.concat([mod_within_partial_inertia,partial_inertia],axis=1)
                    self.mod_within_partial_inertia_ = mod_within_partial_inertia

        #################################################################################################################
        # Measuring how similar groups
        #################################################################################################################
        Lg = pd.DataFrame().astype("float")
        for grp1,cols1 in self.group_.items():
            for grp2,cols2 in self.group_.items():
                X1, X2 = X.loc[:,cols1][grp1], X.loc[:,cols2][grp2]
                if (self.all_nums_[grp1] and self.all_nums_[grp2]):
                    # Sum of square coefficient of correlation
                    sum_corr2 = np.array([(np.corrcoef(X1[col1],X2[col2],rowvar=False)[0,1])**2 for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using the eigenvalues of each group
                    weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_corr2
                    Lg.loc[grp1,grp2] = weighted_corr2
                elif (self.all_cats_[grp1] and self.all_cats_[grp2]):
                    # Sum of chi-squared
                    sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                    weighted_chi2 = (1/(self.n_rows_*X1.shape[1]*X2.shape[1]*self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_chi2
                    Lg.loc[grp1,grp2] = weighted_chi2
                elif (self.all_nums_[grp1] and self.all_cats_[grp2]):
                    # Sum of square correlation ratio
                    sum_eta2 = np.array([eta2(X2[col1],X1[col2],digits=10)["correlation ratio"] for col1 in X2.columns.tolist() for col2 in X1.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]*X2.shape[1]))*sum_eta2
                    Lg.loc[grp1,grp2] = weighted_eta2
                    Lg.loc[grp2,grp1] = Lg.loc[grp1,grp2]
                elif (self.all_cats_[grp1] and self.all_nums_[grp2]):
                    # Sum of square correlation ratio
                    sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["correlation ratio"] for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*X1.shape[0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_eta2
                    Lg.loc[grp1,grp2] = weighted_eta2
                    Lg.loc[grp2,grp1] = Lg.loc[grp1,grp2]

        ## RV Coefficient
        RV = pd.DataFrame().astype("float")
        for grp1 in Lg.index:
            for grp2 in Lg.columns:
                RV.loc[grp1,grp2] = Lg.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1])*np.sqrt(Lg.loc[grp2,grp2]))

        # group coordinates
        group_coord = pd.DataFrame(columns = self.dim_index_,index=Lg.index).astype("float")
        for grp, cols in self.group_.items():
            Xg = X[grp]
            if all(pd.api.types.is_numeric_dtype(Xg[c]) for c in Xg.columns.tolist()):
                data = self.separate_analyses_[grp].normalized_data_
                coord =  (np.corrcoef(data,self.row_coord_,
                                      rowvar=False)[:data.shape[1],data.shape[1]:]**2).sum(axis=0)/self.separate_analyses_[grp].eig_[0][0]
            elif all(pd.api.types.is_string_dtype(Xg[c]) for c in Xg.columns.tolist()):
                data = quali_eta2.loc[self.separate_analyses_[grp].var_labels_,:]
                coord = (data.sum(axis=0)/(len(cols)*self.separate_analyses_[grp].eig_[0][0])).values
            group_coord.loc[grp,self.dim_index_] = coord

        ######################################## group cos2 ################################################################
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_disto.loc[grp]).to_frame(grp).T for grp in group_coord.index.tolist()),axis=0)

        ########################################### Group contributions ############################################
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=self.n_workers_)

        ########################################### Group correlations ###############################################
        group_corr = pd.DataFrame(columns = self.dim_index_,index=Lg.index).astype("float")
        for grp in Lg.index:
            group_corr.loc[grp,:] = np.diag(np.corrcoef(self.row_coord_partiel_[grp].values,
                                                         self.row_coord_,rowvar=False)[:self.n_components_,self.n_components_:])
        # Groups
        self.group_coord_        = group_coord
        self.group_contrib_      = group_contrib
        self.group_correlation_  = group_corr
        self.group_disto_        = group_disto
        self.group_cos2_         = group_cos2
        self.group_lg_           = Lg
        self.group_rv_           = RV

        # Model Name
        self.model_ = "mfa"

    def _row_coord_partiel(self,X):
        """


        """
        if not isinstance(X,pd.DataFrame):
            raise ValueError("X must be a dataframe.")

        row_coord_partiel = pd.DataFrame()
        # If all variables in Data are numerics
        if all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            for grp, cols in self.group_.items():
                # Extract categorical variables
                X_cats = X[grp]
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X_cats[col],prefix=col,prefix_sep='_') for col in X_cats.columns.tolist()),axis=1)
                #
                coord_partial = mapply(dummies.dot(self.mod_coord_.loc[dummies.columns.tolist(),:]),
                                       lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_[0][0]),axis=0,
                                       progressbar=False,n_workers=self.n_workers_)
                coord_partial = len(self.group_)*mapply(coord_partial,lambda x : x/self.eig_[0][:self.n_components_],axis=1,progressbar=False,n_workers=self.n_workers_)
                coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                row_coord_partiel = pd.concat([row_coord_partiel,coord_partial],axis=1)
        # If all variables in Data are categoricals
        elif all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist()):
            for grp, cols in self.group_.items():
                # Standardisze data
                Z = (X[grp] - self.mean_[grp])/self.std_[grp]
                # Set columns coordinates
                col_coord = pd.DataFrame(self.col_coord_,index=self.col_labels_,columns=self.dim_index_)
                # Partial coordinates
                coord_partial = mapply(Z.dot(col_coord.loc[Z.columns.tolist(),:]),
                                       lambda x : x/self.separate_analyses_[grp].eig_[0][0],axis=0,
                                       progressbar=False,n_workers=self.n_workers_)
                coord_partial = len(self.group_)*mapply(coord_partial,lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
                coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                row_coord_partiel = pd.concat([row_coord_partiel,coord_partial],axis=1)
        # Otherwises
        else:
            # For each group in data
            for grp, cols in self.group_.items():
                Xg = X[grp]
                # if all variables in group are numerics
                if all(pd.api.types.is_numeric_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    # Standardize the Data
                    Z = (Xg - self.mean_[grp])/self.std_[grp]
                    # Set columns coordinates
                    col_coord = pd.DataFrame(self.col_coord_,columns=self.dim_index_,index=self.col_labels_)
                    # Partiel coordinates
                    coord_partial = mapply(Z.dot(col_coord.loc[Z.columns.tolist(),:]),
                                           lambda x : x/self.separate_analyses_[grp].eig_[0][0],axis=0,
                                        progressbar=False,n_workers=self.n_workers_)
                    coord_partial = len(self.group_)*mapply(coord_partial,lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
                    coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                    row_coord_partiel = pd.concat([row_coord_partiel,coord_partial],axis=1)
                # If all variables in group are categoricals
                elif all(pd.api.types.is_string_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    # Compute Dummies table : 0/1
                    dummies = pd.concat((pd.get_dummies(Xg[col],prefix=col,prefix_sep='_') for col in Xg.columns.tolist()),axis=1)
                    # Partiel coordinates
                    coord_partial = mapply(dummies.dot(self.mod_coord_.loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_[0][0]),axis=0,
                                        progressbar=False,n_workers=self.n_workers_)
                    coord_partial = len(self.group_)*mapply(coord_partial,lambda x : x/self.eig_[0][:self.n_components_],axis=1,progressbar=False,n_workers=self.n_workers_)
                    coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
                    row_coord_partiel = pd.concat([row_coord_partiel,coord_partial],axis=1)

        return row_coord_partiel

    def _determine_groups(self,X,groups):
        """


        """

        if isinstance(groups,list):
            if not isinstance(X.columns,pd.MultiIndex):
                raise ValueError("Error : Groups have to be provided as a dict when X is not a MultiIndex")
            groups = { g: [(g, c) for c in X.columns.get_level_values(1)[X.columns.get_level_values(0) == g]] for g in groups}
        else:
            groups = groups

        return groups

    def _compute_groups_sup_coord(self,X):
        """
        Fit supplementary group
        -----------------------

        Parameters
        ----------
        X : pandas DataFrame, shape (n_rows, n_cols_sup)


        """

        if not isinstance(X,pd.DataFrame):
            raise ValueError("X must be a dataframe.")
        
        if X.columns.nlevels != 2:
            raise ValueError("Error : X must be")

        # Put supplementary as dict
        self.group_sup_ = self._determine_groups(X=X,groups=self.group_sup)

        # Chack group types are consistent
        for grp, cols in self.group_sup_.items():
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError(f'Not all columns in "{grp}" group are of the same type. Used HMFA instead.')
            self.all_nums_[grp] = all_num
            self.all_cats_[grp] = all_cat

        ####################################################################################
        col_sup_coord        = pd.DataFrame().astype("flot")
        col_sup_cos2         = pd.DataFrame().astype("float")
        col_sup_labels       = []
        col_sup_group_labels = []
        #########
        mod_sup_coord        = pd.DataFrame().astype("float")
        mod_sup_disto        = pd.Series(name="dist2").astype("float")
        mod_sup_cos2         = pd.DataFrame().astype("float")
        mod_sup_vtest        = pd.DataFrame().astype("float")
        quali_sup_eta2       = pd.DataFrame().astype("float")
        mod_sup_labels       = []
        var_sup_labels       = []
        var_sup_group_labels = []

        if all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist()):
            # Make a copy of original Data
            X_nums = X.copy()
            X_nums.columns = X_nums.columns.droplevel()
            ####################################################################################################
            # Correlation between variables and axis
            coord = np.corrcoef(X_nums.values,self.row_coord_,rowvar=False)[:X_nums.shape[1],X_nums.shape[1]:]
            coord = pd.DataFrame(coord,index=X_nums.columns.tolist(),columns=self.dim_index_)
            col_sup_coord = pd.concat([col_sup_coord,coord],axis=0)

            #####################################################################################################
            # Cos 2 between variables and axis
            cos2 = mapply(coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_)
            col_sup_cos2 = pd.concat([col_sup_cos2,cos2],axis=0)

            #########################################
            # Set columns labels
            col_sup_labels = col_sup_labels + X_nums.columns.tolist()
            col_sup_group_labels = col_sup_group_labels + [x[0] for x in X.columns.tolist()]

            ###################################################################################################
            ######################################### Add statistics to summary quanti
            stats = X_nums.describe().T
            stats = stats.reset_index().rename(columns={"index" : "variable"})
            stats.insert(0,"group",[x[0] for x in X.columns.tolist()])
            stats["count"] = stats["count"].astype("int")
            self.summary_quanti_ = pd.concat([self.summary_quanti_,stats],axis=0,ignore_index=True)

        elif all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            # Make a copy of original Data
            X_cats = X.copy()
            X_cats.columns = X_cats.columns.droplevel()
            #######################################################################################################################
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X_cats[col],prefix=col,prefix_sep='_') for col in X_cats.columns.tolist()),axis=1)

            ############################################################################################################################
            # Compute categories coordinates
            vsqual = self.global_pca_._compute_quali_sup_stats(X_cats)
            coord = vsqual["coord"]
            mod_sup_coord = pd.concat([mod_sup_coord,coord],axis=0)

            ###############################################################################################################
            # v-test
            vtest = vsqual["vtest"]
            mod_sup_vtest = pd.concat([mod_sup_vtest,vtest],axis=0)

            ################################################################################################################
            ##### Correlation ratio
            sup_eta2 = vsqual["eta2"]
            quali_sup_eta2 = pd.concat([quali_sup_eta2,sup_eta2],axis=0)

            #########################################################################################################################
            ##################### Distance to G
            mod_sup_disto = pd.concat([mod_sup_disto,vsqual["dist"]],axis=0)

            ################## Cos2
            sup_cos2 = vsqual["cos2"]
            mod_sup_cos2 = pd.concat([mod_sup_cos2,sup_cos2],axis=0)

            ###################################################################################################
            # Compute statisiques
            stats = pd.DataFrame()
            for col_grp in X.columns.tolist():
                grp, col = col_grp
                eff = X_cats[col].value_counts().to_frame("effectif").reset_index().rename(columns={"index" : "modalite"})
                eff.insert(0,"variable",col)
                eff.insert(0,"group",grp)
                stats = pd.concat([stats,eff],axis=0,ignore_index=True)
            self.summary_quali_ = pd.concat([self.summary_quali_,stats],axis=0,ignore_index=True)

            #########################################
            # Set columns labels
            mod_sup_labels = mod_sup_labels + dummies.columns.tolist()
            var_sup_labels = var_sup_labels + X_cats.columns.tolist()
            var_sup_group_labels = var_sup_group_labels + [x[0] for x in X.columns.tolist()]

        else:
            for grp, cols in self.group_sup_.items():
                Xg = X[grp]
                # If all variables in groups are numrerics
                if all(pd.api.types.is_numeric_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    ####################################################################################################
                    # Correlation between variables and axis
                    coord = np.corrcoef(Xg.values,self.row_coord_,rowvar=False)[:Xg.shape[1],Xg.shape[1]:]
                    coord = pd.DataFrame(coord,index=Xg.columns.tolist(),columns=self.dim_index_)
                    col_sup_coord = pd.concat([col_sup_coord,coord],axis=0)

                    #####################################################################################################
                    # Cos 2 between variables and axis
                    cos2 = mapply(coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_)
                    col_sup_cos2 = pd.concat([col_sup_cos2,cos2],axis=0)

                    #########################################
                    # Set columns labels
                    col_sup_labels = col_sup_labels + Xg.columns.tolist()
                    col_sup_group_labels = col_sup_group_labels + [grp]*len(Xg.columns.tolist())

                    ######################################### Add statistics to summary quant
                    ###################################################################################################
                    ################## Compute statistiques
                    stats = Xg.describe().T
                    stats = stats.reset_index().rename(columns={"index" : "variable"})
                    stats.insert(0,"group",grp)
                    stats["count"] = stats["count"].astype("int")
                    self.summary_quanti_ = pd.concat([self.summary_quanti_,stats],axis=0,ignore_index=True)

                # If all variables in group are categoricals
                elif all(pd.api.types.is_string_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    # Compute the dummies
                    dummies = pd.get_dummies(Xg)
                    # Categories statistiques
                    mod_sup_stats = dummies.agg(func=[np.sum,np.mean]).T

                    #########################################################################################################################
                    # Compute supplementary using PCA compute quali sup stats function
                    vsqual = self.global_pca_._compute_quali_sup_stats(Xg)

                    ############################################################################################################################
                    # Compute categories coordinates
                    coord = vsqual["coord"]
                    mod_sup_coord = pd.concat([mod_sup_coord,coord],axis=0)

                    ###############################################################################################################
                    # v-test
                    vtest = vsqual["vtest"]
                    mod_sup_vtest = pd.concat([mod_sup_vtest,vtest],axis=0)

                    ###################################################################################################################
                    ################## Correlation ratio
                    sup_eta2 = vsqual["eta2"]
                    quali_sup_eta2 = pd.concat([quali_sup_eta2,sup_eta2],axis=0)

                    #########################################################################################################################
                    # Distance au carré
                    sup_disto = vsqual["dist"]
                    mod_sup_disto = pd.concat([mod_sup_disto,sup_disto],axis=0)

                    ################## Cos2
                    sup_cos2 = vsqual["cos2"]
                    mod_sup_cos2 = pd.concat([mod_sup_cos2,sup_cos2],axis=0)

                    ###################################################################################################
                    # Compute statisiques
                    stats = pd.DataFrame()
                    for col in Xg.columns.tolist():
                        eff = Xg[col].value_counts().to_frame("effectif").reset_index().rename(columns={col : "modalite"})
                        eff.insert(0,"variable",col)
                        eff.insert(0,"group",grp)
                        stats = pd.concat([stats,eff],axis=0,ignore_index=True)
                    self.summary_quali_ = pd.concat([self.summary_quali_,stats],axis=0,ignore_index=True)

                    #########################################
                    # Set columns labels
                    mod_sup_labels = mod_sup_labels + dummies.columns.tolist()
                    var_sup_labels = var_sup_labels + Xg.columns.tolist()
                    var_sup_group_labels = var_sup_group_labels + [grp]*len(Xg.columns.tolist())

        ##########################################################################################################
        #   Partial Categories coordinates
        ##########################################################################################################
        mod_sup_coord_partiel = pd.DataFrame().astype("float")
        if all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            for grp, cols in self.group_.items():
                for grp_sup, cols_sup in self.group_sup_.items():
                    # Make a copy of original Data
                    X_cats = X[grp_sup]
                    #######################################################################################################################
                    # Compute Dummies table : 0/1
                    dummies = pd.get_dummies(Xg)
                    ############################################################################################################################
                    # Compute categories coordinates
                    coord_partiel = pd.concat((pd.concat((self.row_coord_partiel_[grp],dummies[col]),axis=1)
                                        .groupby(col)
                                        .mean().iloc[1,:]
                                        .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
                    coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partiel.columns.tolist()])
                    mod_sup_coord_partiel = pd.concat([mod_sup_coord_partiel,coord_partiel],axis=1)
        else:
            for grp_sup, col_sup in self.group_sup_.items():
                if self.all_cats_[grp_sup]:
                    # Make a copy of original Data
                    Xg = X[grp_sup]
                    #######################################################################################################################
                    # Compute Dummies table : 0/1
                    dummies = pd.concat((pd.get_dummies(Xg[col]) for col in Xg.columns.tolist()),axis=1)
                    for grp, cols in self.group_.items():
                        ############################################################################################################################
                        # Compute categories coordinates
                        coord_partiel = pd.concat((pd.concat((self.row_coord_partiel_[grp],dummies[col]),axis=1)
                                            .groupby(col)
                                            .mean().iloc[1,:]
                                            .to_frame(name=col).T for col in dummies.columns.tolist()),axis=0)
                        coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partiel.columns.tolist()])
                        mod_sup_coord_partiel = pd.concat([mod_sup_coord_partiel,coord_partiel],axis=1)

        ###########################################################################################################
        #   Supplementary Group Coordinates
        ###########################################################################################################
        group_sup_disto = pd.Series(name="dist",index=[grp for grp, cols in self.group_sup_.items()]).astype("float")
        group_sup_coord = pd.DataFrame(columns = self.dim_index_,index=[grp for grp, cols in self.group_sup_.items()]).astype("float")
        for grp, cols in self.group_sup_.items():
            Xg = X[grp]
            if all(pd.api.types.is_numeric_dtype(Xg[c]) for c in Xg.columns.tolist()):
                ################# Principal Components Analysis (PCA) #######################################"
                fa = PCA(normalize=True,
                         n_components=None,
                         row_labels=Xg.index.tolist(),
                         col_labels=Xg.columns.tolist(),
                         parallelize=self.parallelize)
                fa.fit(Xg)
                self.separate_analyses_[grp] = fa

                # Calculate group sup coordinates
                coord = np.sum((np.corrcoef(fa.normalized_data_,self.row_coord_,rowvar=False)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/fa.eig_[0][0]
                group_sup_coord.loc[grp,self.dim_index_] = coord

            elif all(pd.api.types.is_string_dtype(Xg[c]) for c in Xg.columns.tolist()):
                #################### Multiple Correspondence Analysis (MCA) ######################################
                fa = MCA(n_components=None,
                         row_labels=Xg.index.tolist(),
                         var_labels=Xg.columns.tolist(),
                         parallelize=self.parallelize)
                fa.fit(Xg)
                self.separate_analyses_[grp] = fa

                # Calculate group sup coordinates
                data = quali_sup_eta2.loc[fa.var_labels_,:]
                coord = (data.sum(axis=0)/(Xg.shape[1]*fa.eig_[0][0])).values
                group_sup_coord.loc[grp,self.dim_index_] = coord
            else:
                pass
            # Calculate group sup disto
            group_sup_disto.loc[grp] = np.sum(fa.eig_[0]**2)/fa.eig_[0][0]**2
        
        #################################### group sup cos2 ###########################################################
        group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_disto.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index.tolist()),axis=0)

        #################################################################################################################
        # Measuring how similar groups
        #################################################################################################################
        Lg = pd.DataFrame().astype("float")
        for grp1,cols1 in self.group_sup_.items():
            for grp2,cols2 in self.group_sup_.items():
                ##### Extract Data
                X1, X2 = X.loc[:,cols1][grp1], X.loc[:,cols2][grp2]
                # Check if 
                if (self.all_nums_[grp1] and self.all_nums_[grp2]):
                    # Sum of square coefficient of correlation
                    sum_corr2 = np.array([(np.corrcoef(X1[col1],X2[col2],rowvar=False)[0,1])**2 for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using the eigenvalues of each group
                    weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_corr2
                    Lg.loc[grp1,grp2] = weighted_corr2
                elif (self.all_cats_[grp1] and self.all_cats_[grp2]):
                    # Sum of chi-squared
                    sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                    weighted_chi2 = (1/(self.n_rows_*X1.shape[1]*X2.shape[1]*self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_chi2
                    Lg.loc[grp1,grp2] = weighted_chi2
                elif (self.all_nums_[grp1] and self.all_cats_[grp2]):
                    # Sum of correlatio ratio
                    sum_eta2 = np.array([eta2(X2[col1],X1[col2],digits=10)["correlation ratio"] for col1 in X2.columns.tolist() for col2 in X1.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]*X2.shape[1]))*sum_eta2
                    Lg.loc[grp1,grp2] = weighted_eta2
                    Lg.loc[grp2,grp1] = Lg.loc[grp1,grp2]
                elif (self.all_cats_[grp1] and self.all_nums_[grp2]):
                    # Sum of correlatio ratio
                    sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["correlation ratio"] for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*X1.shape[0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_eta2
                    Lg.loc[grp1,grp2] = weighted_eta2
                    Lg.loc[grp2,grp1] = Lg.loc[grp1,grp2]

        ################################################### Lg between active group and supplementary group ###################################################
        Lg2 = pd.DataFrame().astype("float")
        for grp1, cols1 in self.group_sup_.items():
            for grp2, cols2 in self.group_.items():
                X1, X2 = X.loc[:,cols1][grp1], self.active_data_.loc[:,cols2][grp2]
                if (self.all_nums_[grp1] and self.all_nums_[grp2]):
                    # Sum of square coefficient of correlation
                    sum_corr2 = np.array([(np.corrcoef(X1[col1],X2[col2],rowvar=False)[0,1])**2 for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using the eigenvalues of each group
                    weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_corr2
                    Lg2.loc[grp1,grp2] = weighted_corr2
                elif (self.all_cats_[grp1] and self.all_cats_[grp2]):
                    # Sum of chi-squared
                    sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                    weighted_chi2 = (1/(self.n_rows_*X1.shape[1]*X2.shape[1]*self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]))*sum_chi2
                    Lg2.loc[grp1,grp2] = weighted_chi2
                elif (self.all_nums_[grp1] and self.all_cats_[grp2]):
                    # Sum of correlatio ratio
                    sum_eta2 = np.array([eta2(X2[col1],X1[col2],digits=10)["correlation ratio"] for col1 in X2.columns.tolist() for col2 in X1.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*self.separate_analyses_[grp2].eig_[0][0]*X2.shape[1]))*sum_eta2
                    Lg2.loc[grp1,grp2] = weighted_eta2
                elif (self.all_cats_[grp1] and self.all_nums_[grp2]):
                    # Sum of correlatio ratio
                    sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["correlation ratio"] for col1 in X1.columns.tolist() for col2 in X2.columns.tolist()]).sum()
                    # Weighted the sum using eigenvalues and number of categoricals variables
                    weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_[0][0]*X1.shape[1]*self.separate_analyses_[grp2].eig_[0][0]))*sum_eta2
                    Lg2.loc[grp1,grp2] = weighted_eta2
                    
        ############################# Coefficient RV ##########################################################
        rv = pd.DataFrame(index=Lg.index,columns=Lg.columns).astype("float")
        for grp1 in Lg.index.tolist():
            for grp2 in Lg.columns.tolist():
                rv.loc[grp1,grp2] = Lg.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1]*Lg.loc[grp2,grp2]))
        
        rv2 = pd.DataFrame(index=Lg2.index,columns=Lg2.columns).astype("float")
        for grp1 in rv2.index.tolist():
            for grp2 in rv2.columns.tolist():
                rv2.loc[grp1,grp2] = Lg2.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1]*self.group_lg_.loc[grp2,grp2]))

        #################################################################################################################################
        ####################################################### Partial axes coord
        partial_axes_sup_coord = pd.DataFrame().astype("float")
        for grp, cols in self.group_sup_.items():
            data = self.separate_analyses_[grp].row_coord_
            correl = np.corrcoef(self.row_coord_,data,rowvar=False)[:self.n_components_,self.n_components_:]
            coord = pd.DataFrame(correl,index=self.dim_index_,columns=self.separate_analyses_[grp].dim_index_)
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_sup_coord = pd.concat([partial_axes_sup_coord,coord],axis=1)
        
        ############################################################# Partial axes cos2
        partial_axes_sup_cos2 = mapply(partial_axes_sup_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_)

        ########################################################################################################################################

        # Numeric columns
        self.col_sup_labels_       = col_sup_labels
        self.col_sup_group_labels_ = col_sup_group_labels
        self.col_sup_coord_        = col_sup_coord.iloc[:,:].values
        self.col_sup_cor_          = col_sup_coord.iloc[:,:].values
        self.col_sup_cos2_         = col_sup_cos2.iloc[:,:].values

        # Categorical
        self.mod_sup_stats_         = mod_sup_stats
        self.mod_sup_coord_         = mod_sup_coord
        self.mod_sup_coord_partiel_ = mod_sup_coord_partiel
        self.mod_sup_disto_         = mod_sup_disto
        self.mod_sup_cos2_          = mod_sup_cos2
        self.mod_sup_vtest_         = mod_sup_vtest
        self.mod_sup_labels_        = mod_sup_labels

        ####
        self.quali_sup_eta2_   = quali_sup_eta2

        ################# Supplementary group informations
        self.group_sup_coord_ = group_sup_coord
        self.group_sup_disto_ = group_sup_disto
        self.group_sup_cos2_  = group_sup_cos2
        self.group_sup_lg_    = pd.concat([Lg,Lg2],axis=1)
        self.group_sup_rv_    = pd.concat([rv,rv2],axis=1)

        #### Supplementary partial axis
        self.partial_axes_sup_coord_  = partial_axes_sup_coord
        self.partial_axes_sup_cor_    = partial_axes_sup_coord
        self.partial_axes_sup_cos2_   = partial_axes_sup_cos2

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
        return self.row_coord_

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
        
        # Check if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise ValueError("X must be a dataframe.")

        ######## Check if columns is level 2
        if X.columns.nlevels != 2:
            raise ValueError("Error : X must have a MultiIndex columns with 2 levels.")

        # Check New Data has same group
        nrows = X.shape[0]
        row_coord = pd.DataFrame(np.zeros(shape=(nrows,self.n_components_)),index=X.index.tolist(),columns=self.dim_index_)
        #################################################################################
        if all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist()):
            # Make a copy on original Data
            Z = X.copy()
            for grp, cols in self.group_.items():
                # Standardize the data using 
                Z = (X[grp] - self.mean_[grp])/self.std_[grp]
                ####### Apply transition relation 
                # Set columns coordinates
                col_coord = pd.DataFrame(self.col_coord_,index=self.col_labels_,columns=self.dim_index_)
                # Partial coordinates
                coord = mapply(Z.dot(col_coord.loc[Z.columns.tolist(),:]),lambda x : x/self.separate_analyses_[grp].eig_[0][0],axis=0,
                               progressbar=False,n_workers=self.n_workers_)
                #coord = mapply(coord, lambda x : x/np.sqrt(self.eig_[0]),axis=1,progressbar=False,n_workers=self.n_workers_)
                # Add 
                row_coord = row_coord + coord
            
            ################################# Divide by eigenvalues ###########################
            row_coord = mapply(row_coord, lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
        elif all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist()):
            for grp, cols in self.group_.items():
                # Extract categorical variables
                X_cats = X[grp]
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X_cats[col],prefix=col,prefix_sep='_') for col in X_cats.columns.tolist()),axis=1)
                # Apply
                coord = mapply(dummies.dot(self.mod_coord_.loc[dummies.columns.tolist(),:]),
                               lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_[0][0]),axis=0,
                               progressbar=False,n_workers=self.n_workers_)
                row_coord = row_coord + coord
            # Weighted by the eigenvalue
            row_coord = mapply(row_coord ,lambda x : x/self.eig_[0][:self.n_components_],axis=1,progressbar=False,n_workers=self.n_workers_)
        else:
            # For each group in data
            for grp, cols in self.group_.items():
                Xg = X[grp]
                # if all variables in group are numerics
                if all(pd.api.types.is_numeric_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    # Standardize the Data
                    Z = (Xg - self.mean_[grp])/self.std_[grp]
                    ##################### Columns coordinates ##################################
                    col_coord = pd.DataFrame(self.col_coord_,index=self.col_labels_,columns=self.dim_index_)
                    # Partiel coordinates
                    coord_partial = mapply(Z.dot(col_coord.loc[Z.columns.tolist(),:]),lambda x : x/self.separate_analyses_[grp].eig_[0][0],
                                                            axis=0,progressbar=False,n_workers=self.n_workers_)
                    num_coord_partial = len(self.group_)*mapply(coord_partial,lambda x : x/np.sqrt(self.eig_[0][:self.n_components_]),axis=1,progressbar=False,n_workers=self.n_workers_)
                # If all variables in group are categoricals
                elif all(pd.api.types.is_string_dtype(Xg[c]) for c in Xg.columns.tolist()):
                    # Compute Dummies table : 0/1
                    dummies = pd.concat((pd.get_dummies(Xg[col],prefix=col,prefix_sep='_') for col in Xg.columns.tolist()),axis=1)
                    # Partiel coordinates
                    coord_partial = mapply(dummies.dot(self.mod_coord_.loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_[0][0]),
                                           axis=0,progressbar=False,n_workers=self.n_workers_)
                    cat_coord_partial = len(self.group_)*mapply(coord_partial,lambda x : x/self.eig_[0][:self.n_components_],axis=1,progressbar=False,n_workers=self.n_workers_)
            
            row_coord = row_coord + (1/len(self.group_))*(num_coord_partial + cat_coord_partial)

        return row_coord.iloc[:,:].values

######################################################################################################################
#   MULTIPLE FACTOR ANALYSIS FOR CONTINGENCY TABLES (MFACT)
######################################################################################################################

class MFACT(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis For Contingency Tables (MFACT)
    -------------------------------------------------------

    Description
    -----------


    Parameters:
    -----------


    
    
    
    """

    def __init__(self,
                 n_components = None,
                 row_labels = None,
                 row_weight = None,
                 col_weight = None,
                 row_sup_labels = None,
                 table_type = "number",
                 group = None,
                 group_sup = None,
                 parallelize= False):
        
        self.n_components = n_components
        self.row_labels = row_labels
        self.row_weight = row_weight
        self.col_weight = col_weight
        self.row_sup_labels = row_sup_labels
        self.table_type = table_type
        self.group = group
        self.group_sup = group_sup
        self.parallelize = parallelize
    
    def fit(self, X, y=None):
        """
        
        
        """

        # Check if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        ######## Check if columns is level 2
        if X.columns.nlevels != 2:
            raise ValueError("Error : X must have a MultiIndex columns with 2 levels.")

         # set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Check if groups is None
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")
        
        self._compute_stats(X)

        return self
    
    def _compute_stats(self,X):
        """
        
        
        """

        # Set number of rows/columns
        self.n_rows_,self.n_cols_ = X.shape

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]

        # Checks groups are provided
        self.group_ = self._determine_groups(X=X,groups=self.group)

        ######################################## Compute Frequency in all table #######################################
        F = X/X.sum().sum()

        #### Set row margin and columns margin
        row_margin = F.sum(axis=1)
        col_margin = F.sum(axis=0)
        col_margin.index = col_margin.index.droplevel()

        ####################################### Sum of frequency by group #############################################
        sum_term_grp = pd.Series().astype("float")
        for grp, cols in self.group_.items():
            sum_term_grp.loc[grp] = F[grp].sum().sum()
        
        ########################################### Construction of table Z #############################################"
        X1 = mapply(F,lambda x : x/col_margin.values,axis=1,progressbar=False,n_workers=self.n_workers_)
        X2 = pd.DataFrame(columns=self.group,index=self.row_labels_).astype("float")
        for grp,cols in self.group_.items():
            X2[grp] = F[grp].sum(axis=1)/sum_term_grp[grp]
        
        Z = pd.DataFrame().astype("float")
        for grp, cols in self.group_.items():
            Zb = mapply(X1[grp],lambda x : x - X2[grp].values,axis=0,progressbar=False,n_workers=self.n_workers_)
            Zb = mapply(Zb,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=self.n_workers_)
            Zb.columns = pd.MultiIndex.from_tuples([(grp,col) for col in Zb.columns.tolist()])
            Z = pd.concat((Z,Zb),axis=1)
        
        # Set row weight
        ########### Set row weight and columns weight
        # Set row weight
        if self.row_weight is None:
            self.row_weight_ = np.ones(self.n_rows_)/self.n_rows_
        elif not isinstance(self.row_weight,list):
            raise ValueError("Error : 'row_weight' must be a list of row weight.")
        elif len(self.row_weight) != self.n_rows_:
            raise ValueError(f"Error : 'row_weight' must be a list with length {self.n_rows_}.")
        else:
            self.row_weight_ = np.array([x/np.sum(self.row_weight) for x in self.row_weight])

        # Set columns weight
        if self.col_weight is None:
            self.col_weight_ = np.ones(self.n_cols_).reshape(1,-1)
        elif not isinstance(self.col_weight,list):
            raise ValueError("Error : 'col_weight' must be a list of columns weight.")
        elif len(self.col_weight) != self.n_cols_:
            raise ValueError(f"Error : 'col_weight' must be a list with length {self.n_cols_}.")
        else:
            self.col_weight_ = np.array(self.col_weight).reshape(1,-1)

        ########## Compute SVD
        U,lamb, V = np.linalg.svd(Z)
        

            
        
        

        ######################



    
    def _determine_groups(self,X,groups):
        """


        """
        if isinstance(groups,list):
            if not isinstance(X.columns,pd.MultiIndex):
                raise ValueError("Error : Groups have to be provided as a dict when X is not a MultiIndex")
            groups = { g: [(g, c) for c in X.columns.get_level_values(1)[X.columns.get_level_values(0) == g]] for g in groups}
        else:
            groups = groups

        return groups



########################################################################################################
#       Hierarchical Multiple Factor Analysis (HMFA)
#######################################################################################################

class HMFA(BaseEstimator,TransformerMixin):
    """
    Hierarchical Multiple Factor Analysis (HMFA)
    --------------------------------------------

    Description
    -----------


    Parameters
    ----------



    """


    def __init__(self,
                 n_components=None,
                 row_labels=None,
                 row_sup_labels = None,
                 group = None,
                 group_sup = None,
                 parallelize=False):
        self.n_components = n_components
        self.row_labels = row_labels
        self.row_sup_labels = None,
        self.group = group
        self.group_sup = group_sup
        self.parallelize = parallelize


    def fit(self,X,y=None):
        """
        Fit the Hierarchical Multiple Factor Analysis (HMFA)
        ----------------------------------------------------
        
        
        
        """

        if not isinstance(X,pd.DataFrame):
            raise ValueError("Erro : X must be a dataframe.")

        ######## Check if columns is level 2
        if X.columns.nlevels > 2:
            raise ValueError("Error : X must have a MultiIndex columns greater than 2 levels.")
        

        self.active_data_ = X
    
        # Compute statistics for active data
        self._compute_stats(X=X)

        

    def _compute_stats(self,X):
        """
        
        
        """

        # Set number of rows : self.n_rows_
        self.n_rows_ = X.shape[0]

        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        # Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist())

        # Shape of X
        self.n_rows_ = X.shape[0]

        # # Set self.row_labels_
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]



#################################################################################
#   
##############################################################################################     

class PMFA(BaseEstimator,TransformerMixin):
    """
    Procrustean Multiple Factor Analysis (PMFA)
    -------------------------------------------
    
    
    
    """


    def __init__(self):
        pass

















