# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import warnings

from scipy.spatial.distance import pdist,squareform
from sklearn.utils import check_symmetric
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from scipy.spatial.distance import euclidean

from .sim_dist import sim_dist

class CMDSCALE(BaseEstimator,TransformerMixin):
    """
    Classic Muldimensional Scaling (CMDSCALE)
    -----------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    This is a classical multidimensional scaling also 
    known as Principal Coordinates Analysis (PCoA).

    Performs Classical Multidimensional Scaling (MDS) with supplementary 
    rows points.

    Parameters
    ----------
    n_components : int, default=None
        Number of dimensions in which to immerse the dissimilarities.
    
    sup_labels : list of string, default = None
        Labels of supplementary rows.
    
    proximity :  {'euclidean','precomputed','similarity'}, default = 'euclidean'
        Dissmilarity measure to use :
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset
        
        - 'precomputed':
            Pre-computed dissimilarities are passed disrectly to ``fit`` and ``fit_transform``.
        
        - `similarity`:
            Similarity matrix is transform to dissimilarity matrix before passed to ``fit`` and ``fit_transform``.

    normalized_stress : bool, default = True
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default.
    
    graph : bool, default = True
        if True a graph is displayed

    figsize : tuple of int, default = None
        Width, height in inches.

    Returns
    -------
    n_components_ : int
        The estimated number of components.
    
    labels_ : array of strings
        Labels for the rows.
    
    nobs_ : int
        number of rows
    
    dist_ : ndarray of shape -n_rows, nr_ows)
        Eulidean distances matrix.
        
    eig_ : array of float
        A 4 x n_components_ matrix containing all the eigenvalues
        (1st row), difference (2nd row) the percentage of variance (3rd row) and the
        cumulative percentage of variance (4th row).
    
    eigen_vector_ : array of float:
        A matrix containing eigenvectors
    
    coord_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row coordinates.
    
    res_dist_ : ndarray of shape (n_rows,n_rows_)
        A n_rows x n_rows_ matrix containing the distances based on coordinates.
    
    stress_ : float

    inertia_ : 

    dim_index_ : 
    
    centered_matrix_ : ndarray of shape
    
    model_ : string
        The model fitted = 'cmds'
    
    """
    def __init__(self,
                 n_components=None,
                 ind_sup = None,
                 proximity="euclidean",
                 normalized_stress=True,
                 parallelize=False):
        self.n_components = n_components
        self.ind_sup = ind_sup
        self.proximity = proximity
        self.normalized_stress = normalized_stress
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

        # If proximinity == "euclidean"
        def is_euclidean(centering_matrix,X):
            """
            Compute eigenvalue and eigenvalue for euclidean matrix
            -------------------------------------------------------
            
            """
            B = np.dot(np.dot(centering_matrix,X),np.dot(centering_matrix,X).T)
            value, vector = np.linalg.eig(B)
            return np.real(value), np.real(vector)
        
        # If proximity == "precomputed" or "similarity"
        def is_others(centering_matrix,X,choice = "precomputed"):
            """
            Compute eigenvalue and eigenvector for precomputed matrix
            ---------------------------------------------------------
            
            """
            if choice == "precomputed":
                dist = check_symmetric(X.values, raise_exception=True)
            elif choice == "similarity":
                D = sim_dist(X)
                dist = check_symmetric(D, raise_exception=True)
            
            A = -0.5*np.multiply(dist,dist)
            B = np.dot(centering_matrix,np.dot(A,centering_matrix))
            value, vector = np.linalg.eig(B)
            return np.real(value), np.real(vector)

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.proximity not in ["euclidean","precomputed","similarity"]:
            raise ValueError("'proximity' should be one of 'euclidean', 'precomputed', 'similarity'")
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ############################
        # Check is supplementary rows
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple))  and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        ####################################### Save the base in a new variables
        # Store data
        Xtot = X

        ####################################### Drop supplementary individuals ########################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ##############################################################################################
        # check matrix
        if X.shape[0] == X.shape[1] and self.proximity != "precomputed":
            raise warnings.warn(
                "The ClassicMDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``proximity='precomputed'``."
            )
        
        ################################################### Compute distance matrix
        if self.proximity == "euclidean":
            dist = squareform(pdist(X,metric="euclidean"))
        elif self.proximity == "precomputed":
            dist = check_symmetric(X.values, raise_exception=True)
        elif self.proximity == "similarity":
            dist = sim_dist(X)
        
        # Effectifs
        n_obs = dist.shape[0]
        # 
        centering_matrix = np.identity(n_obs) - (1/n_obs)*np.ones(shape=(n_obs,n_obs))

        # Compute euclidean
        if self.proximity == "euclidean":
            eigen_value, eigen_vector = is_euclidean(centering_matrix=centering_matrix,X=X)
        else:
            eigen_value, eigen_vector = is_others(centering_matrix=centering_matrix,X=X,choice=self.proximity)
        
        # 
        proportion = 100*eigen_value/np.sum(eigen_value)
        difference = np.insert(-np.diff(eigen_value),len(eigen_value)-1,np.nan)
        cumulative = np.cumsum(proportion)
        
        # Set n_components
        if self.n_components is None:
            n_components = (eigen_value > 1e-16).sum()
        else:
            n_components = min(self.n_components, n_obs)
        
        self.call_ = {"n_components" : n_components,
                      "proximity" : self.proximity,
                      "normalized_stress" : self.normalized_stress,
                      "X" : X,
                      "Xtot" : Xtot,
                      "n_obs" : n_obs}
        
        # Store eigenvalue
        eig = np.c_[eigen_value[:n_components],
                    difference[:n_components],
                    proportion[:n_components],
                    cumulative[:n_components]]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(eig.shape[0])])

        # Store Spectral Decomposition of Matrix
        self.svd_ = {"eigenvalues" : eigen_value[:n_components],
                     "eigenvectors" : eigen_vector[:,:n_components]}

        ############################################### Coordinates ##########################################""""
        coord = eigen_vector[:,:n_components]*np.sqrt(eigen_value[:n_components])
        coord = pd.DataFrame(coord,index=X.index.tolist(),columns=["Dim."+str(x+1) for x in range(n_components)])

        # Distance restituées
        res_dist = squareform(pdist(coord,metric="euclidean"))

        #calcul du stress 
        if self.normalized_stress:
            stress = np.sqrt(np.sum((res_dist-dist)**2)/np.sum(dist**2))
        else:
            stress = np.sum((res_dist-dist)**2)
        
        # Customize to pandas
        dist = pd.DataFrame(dist,index=X.index,columns=X.index)
        res_dist = pd.DataFrame(res_dist,index=X.index,columns=X.index)

        # Inertie 
        inertia = np.sum(dist**2)/(2*(n_obs**2))

        # Store informations
        self.result_ = {"coord" : coord, "dist" : dist, "res_dist" : res_dist,"stress" : stress,"inertia" : inertia}

        ##################### Supplementary individuals
        if self.ind_sup is not None:
            self.sup_coord_ = self.transform(X_ind_sup)
        
        self.model_ = "cmdscale"
        
        return self
    
    def transform(self,X,y=None):
        """
        Apply the Multidimensional Scaling reduction on X
        --------------------------------------------------

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

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        d2 = np.sum(self.result_["dist"].values**2,axis=1)/self.call_["n_obs"]
        d3 = np.sum(self.result_["dist"].values**2)/(self.call_["n_obs"]**2)

        if self.proximity == "precomputed":
            sup_dist = X
        elif self.proximity == "euclidean":
            n_supp_obs = X.shape[0]
            sup_dist = pd.DataFrame(np.zeros((n_supp_obs,self.call_["n_obs"])),index=X.index,columns=self.call_["X"].index)
            for i in np.arange(0,n_supp_obs):
                for j in np.arange(0,self.call_["n_obs"]):
                    sup_dist.iloc[i,j] = euclidean(X.iloc[i,:],self.call_["X"].iloc[j,:])
        elif self.proximity == "similarity":
            raise NotImplementedError("This method is not implemented yet.")

        sup_coord = mapply(sup_dist,lambda x : -(1/2)*(x**2 - np.sum(x**2)-d2+d3),axis=1,
                           progressbar=False,n_workers=n_workers).dot(self.result_["coord"])/self.eig_.iloc[:,0]
        sup_coord.columns = ["Dim."+str(x+1) for x in range(sup_coord.shape[1])]
        
        return sup_coord
        
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
        return self.result_["coord"]