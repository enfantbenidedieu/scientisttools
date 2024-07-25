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
from .revaluate_cat_variable import revaluate_cat_variable
from .svd_triplet import svd_triplet
from .recodecont import recodecont
from .conditional_average import conditional_average

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)
    ----------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables. 
    
    Missing values are replaced by the column mean.

    Usage
    -----
    ```python
    >>> PCA(standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, quanti_sup = None, quali_sup = None, parallelize=False)
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

    `quanti_sup` : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    `quali_sup` : an integer or a list/tuple indicating the indexes of the categorical supplementary variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `svd_` : singular value decomposition

    `var_` : dictionary of pandas dataframes containing all the results for the active variables (coordinates, correlation between variables and axes, square cosinus, contributions)

    `ind_` : dictionary of pandas dataframes containing all the results for the active individuals (coordinates, square cosinus, contributions)

    `ind_sup_` : dictionary of pandas dataframes containing all the results for the supplementary individuals (coordinates, square cosinus)

    `quanti_sup_` : dictionary of pandas dataframes containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes, square cosinus)

    `quali_sup_` : dictionary of pandas dataframes containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, vtest which is a criterion with a Normal distribution, and eta2 which is the square correlation coefficient between a qualitative variable and a dimension)
    
    `summary_quanti_` : summary statistics for quantitative variables (actives and supplementary)

    `summary_quali_` : summary statistics for supplementary qualitative variables if quali_sup is not None

    `chi2_test_` : chi-squared test. If supplementary qualitative are greater than 2. 
    
    `call_` : dictionary with some statistics

    `others_` : others statistics (Bartlett's test of Spericity, Kaiser threshold, ...)

    `model_` : string specifying the model fitted = 'pca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Bry X. (1996), Analyses factorielles multiple, Economica

    Bry X. (1999), Analyses factorielles simples, Economica

    Escofier B., Pagès J. (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Saporta G. (2006). Probabilites, Analyse des données et Statistiques. Technip

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    Rakotomalala, R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    Tenenhaus, M. (2006). Statistique : Méthodes pour décrire, expliquer et prévoir. Dunod.
    
    See Also
    --------
    get_pca_ind, get_pca_var, get_pca, summaryPCA, dimdesc, reconstruct, predictPCA, supvarPCA, fviz_pca_ind, fviz_pca_var, fviz_pca_biplot, fviz_pca3d_ind

    Examples
    --------
    ```python
    >>> # Load decathlon2 dataset
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

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas/polars DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y` : None
            y is ignored

        Returns
        -------
        `self` : object
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
        
        # Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns:
                X[col] = X[col].astype("object")
        
        # Check if supplementary qualitatives variables
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup]
        else:
            quali_sup_label = None

        #  Check if supplementary quantitatives variables
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup]
        else:
            quanti_sup_label = None
        
        # Check if supplementary individuals
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
            if self.quali_sup is None:
                X = recodecont(X=X)["Xcod"]
            else:
                col_list = [x for x in list(range(X.shape[1])) if x not in quali_sup]
                for i in col_list:
                    if X.iloc[:,i].isnull().any():
                        X.iloc[:,i] = X.iloc[:,i].fillna(X.iloc[:,i].mean())

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
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        # Number of rows/columns
        n_rows, n_cols = X.shape

        # Statistics of quantitatives variables 
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti

        # Set individuals weights
        if self.ind_weights is None:
            ind_weights = np.ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list with length {n_rows}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        # Set variables weights
        if self.var_weights is None:
            var_weights = np.ones(n_cols)
        elif not isinstance(self.var_weights,list):
            raise ValueError("'var_weights' must be a list of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list with length {n_cols}.")
        else:
            var_weights = np.array(self.var_weights)

        # Compute weighted average and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)

        # Initializations - scale data
        means = d1.mean
        if self.standardize:
            std = d1.std
        else:
            std = np.ones(X.shape[1])
        
        # Standardization : Z = (X - mu)/sigma
        Z = (X - means.reshape(1,-1))/std.reshape(1,-1)

        # QR decomposition (to set maximum number of components)
        Q, R = np.linalg.qr(Z)
        max_components = min(np.linalg.matrix_rank(Q),np.linalg.matrix_rank(R))

        # Set number of components
        if self.n_components is None:
            n_components = int(max_components)
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
        
        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot":Xtot,
                      "X" : X,
                      "Z" : Z,
                      "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                      "var_weights" : pd.Series(var_weights,index=X.columns,name="weight"),
                      "means" : pd.Series(means,index=X.columns,name="average"),
                      "std" : pd.Series(std,index=X.columns,name="scale"),
                      "n_components" : n_components,
                      "ind_sup" : ind_sup_label,
                      "quanti_sup" : quanti_sup_label,
                      "quali_sup" : quali_sup_label}

        ## Individuals informations : Squared distance to origin, weights and Inertia
        # Individuals square distance to origin
        ind_dist2 = mapply(Z,lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        # Individuals inertia
        ind_inertia = ind_dist2*ind_weights
        # Store all informations
        ind_infos = pd.DataFrame(np.c_[ind_weights,ind_dist2,ind_inertia],columns=["Weight","Sq. Dist.","Inertia"],index=X.index)

        ## Variables informations : Squared distance to origin, weights and Inertia
        # Variables square distance to origin
        var_dist2 = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        # Variables inertia
        var_inertia = var_dist2*var_weights
        # Store all informations
        var_infos = pd.DataFrame(np.c_[var_weights,var_dist2,var_inertia],columns=["Weight","Sq. Dist.","Inertia"],index=X.columns)
        
        # Generalized Singular Value Decomposition (GSVD)
        svd = svd_triplet(X=Z,row_weights=ind_weights,col_weights=var_weights,n_components=n_components)
        self.svd_ = svd

        # Eigen - values
        eigen_values = svd["vs"][:max_components]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
        # store all informations
        self.eig_ = pd.DataFrame(np.c_[eigen_values,difference,proportion,cumulative],columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(len(eigen_values))])

        ## Individuals informations : coordinates, contributions and squared Cosinus
        # Individuals coordinates
        ind_coord = pd.DataFrame(svd["U"].dot(np.diag(svd["vs"][:n_components])),index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])

        # Individuals contributions
        ind_contrib = mapply(ind_coord,lambda x : 100*(x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Individuals square cosine
        ind_cos2 = mapply(ind_coord,lambda x : (x**2)/ind_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Store all informations
        self.ind_ = {"coord":ind_coord, "cos2":ind_cos2, "contrib":ind_contrib, "infos" : ind_infos}

        ## Variables informations : coordinates, contributions and squared cosinus
        # Variables coordinates
        var_coord = pd.DataFrame(svd["V"].dot(np.diag(svd["vs"][:n_components])),index=X.columns,columns=["Dim."+str(x+1) for x in range(n_components)])

        # Variables contributions
        var_contrib = mapply(var_coord,lambda x : 100*(x**2)*var_weights,axis=0,progressbar=False,n_workers=n_workers)
        var_contrib = mapply(var_contrib, lambda x : x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
    
        # Variables square cosine
        var_cos2 = mapply(var_coord,  lambda x : (x**2)/var_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Store all informations
        self.var_ = {"coord": var_coord, "cor":var_coord, "cos2":var_cos2, "contrib":var_contrib, "infos" : var_infos}

        # Bartlett - statistics
        bartlett_stats = -(n_rows-1-(2*n_cols+5)/6)*np.sum(np.log(eigen_values))
        bs_dof = n_cols*(n_cols-1)/2
        bs_pvalue = 1 - sp.stats.chi2.cdf(bartlett_stats,df=bs_dof)
        bartlett_sphericity_test = pd.DataFrame([np.sum(np.log(eigen_values)),bartlett_stats,bs_dof,bs_pvalue],index=["|CORR.MATRIX|","statistic","dof","p-value"],columns=["value"])
        # Kaiser threshold
        kaiser_threshold = np.mean(eigen_values)
        kaiser_proportion_threshold = 100/np.sum(var_inertia)
        # Karlis - Saporta - Spinaki threshold
        kss_threshold =  1 + 2*np.sqrt((n_cols-1)/(n_rows-1))
        # Broken stick threshold
        broken_stick_threshold = np.flip(np.cumsum(1/np.arange(n_cols,0,-1)))[:n_components]

        # Store all informations
        self.others_ = {"bartlett" : bartlett_sphericity_test,"kaiser" : kaiser_threshold, "kaiser_proportion" : kaiser_proportion_threshold,"kss" : kss_threshold,"bst" : broken_stick_threshold}
        
        # Statistics for supplementary individuals
        if self.ind_sup is not None:
            # Transform to float
            X_ind_sup = X_ind_sup.astype("float")

            # Standardization
            Z_ind_sup = (X_ind_sup - means.reshape(1,-1))/std.reshape(1,-1)

            # Supplementary individuals coordinates
            ind_sup_coord = mapply(Z_ind_sup,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(svd["V"][:,:n_components])
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Supplementary individuals squared distance to origin
            ind_sup_dist2 = mapply(Z_ind_sup,lambda  x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_sup_dist2.name = "Sq. Dist."

            # Supplementary individuals square cosine
            ind_sup_cos2 = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Store all informations
            self.ind_sup_ = {"coord" : ind_sup_coord,"cos2" : ind_sup_cos2,"dist" : ind_sup_dist2}

        # Statistics for supplementary quantitatives variables
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            # Transform to float
            X_quanti_sup = X_quanti_sup.astype("float")

            # Fill missing with mean
            X_quanti_sup = recodecont(X=X_quanti_sup)["Xcod"]

            # Summary statistics
            self.summary_quanti_.insert(0,"group","active")
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            summary_quanti_sup.insert(0,"group","sup")

            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

            # Compute weighted average and standard deviation
            d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)

            # Initializations - scale data
            means_sup = d2.mean
            if self.standardize:
                std_sup = d2.std
            else:
                std_sup = np.ones(X_quanti_sup.shape[1])
            
            # Standardization
            Z_quanti_sup = (X_quanti_sup - means_sup.reshape(1,-1))/std_sup.reshape(1,-1)

            # Supplementary quantitatives variables coordinates
            var_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(svd["U"][:,:n_components])
            var_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Supplementary quantitatives variables squared distance to origin
            var_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            var_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),var_sup_cor)

            # Supplementary quantitatives variables square cosine
            var_sup_cos2 = mapply(var_sup_coord,lambda x : (x**2)/var_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)    

            # Store supplementary quantitatives informations
            self.quanti_sup_ =  {"coord":var_sup_coord,"cor" : var_sup_coord,"cos2" : var_sup_cos2}

        # Statistics for supplementary qualitatives variables
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            
            # Transform to object
            X_quali_sup = X_quali_sup.astype("object")

            # Check if two columns have the same categories
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            # Square correlation ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

            # Conditional mean - Barycenter of original data
            barycentre = conditional_average(X=X,Y=X_quali_sup,weights=ind_weights)
            n_k = pd.concat((X_quali_sup[col].value_counts() for col in X_quali_sup.columns),axis=0)

            # Standardize the barycenter
            bary = (barycentre - means)/std

            # Square distance to origin
            quali_sup_dist2  = mapply(bary, lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            quali_sup_dist2.name = "Sq. Dist."
           
            # Supplementary categories coordinates
            quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(svd["V"][:,:n_components])
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Supplementary categories square cosine
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
            
            # Supplementary categories v-test
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/svd["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)
            quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((n_rows-n_k[k])/((n_rows-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns:
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")

            # Supplementary categories informations
            self.quali_sup_ = {"barycentre" : barycentre,"coord" : quali_sup_coord, "cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"dist" : quali_sup_dist2,"eta2" : quali_sup_eta2}
            self.summary_quali_ = summary_quali_sup
            
        self.model_ = "pca"

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
        X : pandas/polars dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
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
        
        # Set index name as None
        X.index.name = None
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Transform to float
        X = X.astype("float")

        # check if X.shape[1] = ncols
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("'columns' aren't aligned")

        # Extract elements
        means = self.call_["means"].values
        std = self.call_["std"].values
        var_weigths = self.call_["var_weights"].values
        n_components = self.call_["n_components"]

        # Standardize
        Z = (X - means.reshape(1,-1))/std.reshape(1,-1)

        # Multiply by columns weight & Apply transition relation
        coord = mapply(Z,lambda x : x*var_weigths,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        return coord
    
def predictPCA(self,X=None):
    """
    Predict projection for new individuals with Principal Component Analysis (PCA)
    ------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and square distance to origin of new individuals with Principal Component Analysis (PCA)

    Usage
    -----
    ```python
    >>> predictPCA(self,X=None)
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    `X` : pandas/polars dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    dictionary of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : square cosinus of the new individuals

    `dist` : square distance to origin for new individuals
    
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
    >>> ind_sup = load_cars2006(which="indsup")
    >>> from scientisttools import predictPCA
    >>> predict = predictPCA(res_pca,X=ind_sup)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    # Check if columns are aligned
    if X.shape[1] != self.call_["X"].shape[1]:
        raise ValueError("'columns' aren't aligned")

    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()
    
    # Check if X is an instance of pd.DataFrame class
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

    # Extract elements
    means = self.call_["means"].values # Average
    std = self.call_["std"].values # Standard deviation
    var_weights = self.call_["var_weights"].values  # Variables weights
    n_components = self.call_["n_components"] # number of components
    
    # Convert to float
    X = X.astype("float")

    # Standardize data
    Z = (X - means.reshape(1,-1))/std.reshape(1,-1)

    # New data coordinates
    coord = mapply(Z,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
    coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
    
    #  New data square distance to origin
    dist2 = mapply(Z,lambda  x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    dist2.name = "Sq. Dist."

    # New data square cosinus
    cos2 = mapply(coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)

    # Store all informations
    res = {"coord" : coord, "cos2" : cos2,"dist" : dist2}
    return res

def supvarPCA(self,X_quanti_sup=None, X_quali_sup=None):
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
    `self` : an object of class PCA

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives variables (default = None)

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives variables (default = None)

    Returns
    -------
    dictionary of dictionary containing the results for supplementary variables including : 

    `quanti` : dictionary containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : dictionary containing the results of the supplementary qualitatives/categories variables including :
        * coord : factor coordinates of the supplementary categories
        * cos2 : square cosinus of the supplementary categories
        * vtest : value-test of the supplementary categories
        * dist : square distance to origin of the supplementary categories
        * eta2 : square correlation ratio of the supplementary qualitatives variables

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
    >>> X_quanti_sup = load_cars2006(which="varquantsup")
    >>> # Supplementary qualitatives variables
    >>> X_quali_sup = load_cars2006(which="varqualsup")
    >>> from scientisttools import supvarPCA
    >>> sup_var_predict = supvarPCA(res_pca,X_quanti_sup=X_quanti_sup,X_quali_sup=X_quali_sup)
    ```
    """
    # Check if self is and object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    ind_weights = self.call_["ind_weights"].values
    n_components = self.call_["n_components"]

    # Supplementary quantitatives variables statistics
    if X_quanti_sup is not None:
        # check if X_quanti_sup is an instance of polars dataframe
        if isinstance(X_quanti_sup,pl.DataFrame):
            X_quanti_sup = X_quanti_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,pd.Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X_quanti_sup is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to float
        X_quanti_sup = X_quanti_sup.astype("float")

        # Set index name as None
        X_quanti_sup.index.name = None

        # Recode variables
        X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]

        # Compute weighted average and standard deviation
        d1 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)

        # Average
        means_sup = d1.mean.reshape(1,-1)
        if self.standardize:
            std_sup = d1.std.reshape(1,-1)
        else:
            std_sup = np.ones(X_quanti_sup.shape[1]).reshape(1,-1)
        
        # Standardization
        Z_quanti_sup = (X_quanti_sup - means_sup)/std_sup

        # Supplementary quantitatives variables coordinates
        quanti_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(self.svd_["U"][:,:n_components])
        quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        #Supplementary quantitatives variables square distance to origin
        quanti_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        quanti_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_sup_cor)

        # Supplementary quantitatives variables square cosinus
        quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2)/np.sqrt(quanti_sup_dist2),axis=0,progressbar=False,n_workers=n_workers)       

        # Store supplementary quantitatives informations
        quanti_sup =  {"coord":quanti_sup_coord, "cor" : quanti_sup_coord, "cos2" : quanti_sup_cos2}
    else:
        quanti_sup = None
    
    # Supplementary qualitative statistics
    if X_quali_sup is not None:
        # check if X_quali_sup is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X_quali_sup.index.name = None
        
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Variables weights
        var_weights = self.call_["var_weights"].values
        means = self.call_["means"].values
        std = self.call_["std"].values

        # Square correlation ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

        # Barycenter
        barycentre = pd.DataFrame().astype("float")
        n_k = pd.Series().astype("float")
        for col in X_quali_sup.columns:
            vsQual = X_quali_sup[col]
            modalite, counts = np.unique(vsQual, return_counts=True)
            n_k = pd.concat([n_k,pd.Series(counts,index=modalite)],axis=0)
            bary = pd.DataFrame(index=modalite,columns=self.call_["X"].columns)
            for mod in modalite:
                idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
                bary.loc[mod,:] = np.average(self.call_["X"].iloc[idx,:],axis=0,weights=ind_weights[idx])
            barycentre = pd.concat((barycentre,bary),axis=0)
        
        # Supplementary qualitatives squared distance
        bary = (barycentre - means.reshape(1,-1))/std.reshape(1,-1)
        quali_sup_dist2  = mapply(bary, lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        quali_sup_dist2.name = "Sq. Dist."

        # Supplementary qualitatives coordinates
        quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Supplementary qualiatives square cosine
        quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # Supplementary qualitatives value-test
        quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/self.svd_["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)
        quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((n_rows-n_k[k])/((n_rows-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

        # Supplementary categories informations
        quali_sup = {"barycentre":barycentre,"coord" : quali_sup_coord,"cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"dist" : quali_sup_dist2,"eta2" : quali_sup_eta2}
    else:
        quali_sup = None
    
    # Store all informations
    res = {"quanti" : quanti_sup, "quali" : quali_sup}
    return res