# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
import warnings
from sklearn.utils import check_symmetric
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from scipy.spatial.distance import euclidean


class CMDSCALE(BaseEstimator,TransformerMixin):
    """
    Classic Muldimensional Scaling (CMDSCALE)
    -----------------------------------------

    This is a classical multidimensional scaling also 
    known as Principal Coordinates Analysis (PCoA).

    Performs Classical Multidimensional Scaling (MDS) with supplementary 
    rows points.

    Parameters
    ----------
    n_components : int, default=None
        Number of dimensions in which to immerse the dissimilarities.
    
    labels : list of string,   default : None
        Labels for the rows.
    
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
                labels = None,
                sup_labels = None,
                proximity="euclidean",
                normalized_stress=True,
                parallelize=False):
        self.n_components = n_components
        self.labels = labels
        self.sup_labels = sup_labels
        self.proximity = proximity
        self.normalized_stress = normalized_stress
        self.parallelize =parallelize
    
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
        
          # Set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1
        
        # Extract supplementary data
        self.sup_labels_ = self.sup_labels
        if self.sup_labels_ is not None:
            _X = X.drop(index = self.sup_labels_)
            row_sup = X.loc[self.sup_labels_,:]
        else:
            _X = X

        self.data_ = X
        self.active_data_ = _X
        
        # Initialize
        self.sup_coord_ = None
        
        self.nobs_ = _X.shape[0]
        self.centering_matrix_ = np.identity(self.nobs_) - (1/self.nobs_)*np.ones(shape=(self.nobs_,self.nobs_))
        
        self._compute_stats(_X)
        
        if self.sup_labels_ is not None:
            self.sup_coord_ = self.transform(row_sup)
        
        return self
    
    def _is_euclidean(self,X):
        """Compute eigenvalue and eigenvalue for euclidean matrix
        
        """
        self.dist_ = squareform(pdist(X,metric="euclidean"))
        B = np.dot(np.dot(self.centering_matrix_,X),np.dot(self.centering_matrix_,X).T)
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _is_precomputed(self,X):
        """Return eigenvalue and eigenvector for precomputed matrix
        
        """
        self.dist_ = check_symmetric(X.values, raise_exception=True)
        A = -0.5*np.multiply(self.dist_,self.dist_)
        B = np.dot(self.centering_matrix_,np.dot(A,self.centering_matrix_))
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _is_similarity(self,X):
        """Return eigenvalue
        
        """
        D = sim_dist(X)
        self.dist_ = check_symmetric(D, raise_exception=True)
        A = -0.5*np.multiply(self.dist_,self.dist_)
        B = np.dot(self.centering_matrix_,np.dot(A,self.centering_matrix_))
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _compute_stats(self,X):
        """Compute statistic
        
        """
        if X.shape[0] == X.shape[1] and self.proximity != "precomputed":
            raise warnings.warn(
                "The ClassicMDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``proximity='precomputed'``."
            )

        # Compute euclidean
        if self.proximity == "euclidean":
            eigen_value, eigen_vector = self._is_euclidean(X)
        elif self.proximity == "precomputed":
            eigen_value, eigen_vector = self._is_precomputed(X)
        elif self.proximity == "similarity" :
            eigen_value, eigen_vector = self._is_similarity(X)
        else:
            raise ValueError("Error : You must pass a valid 'proximity'.")
        
        proportion = 100*eigen_value/np.sum(eigen_value)
        difference = np.insert(-np.diff(eigen_value),len(eigen_value)-1,np.nan)
        cumulative = np.cumsum(proportion)
        
        # Set n_components
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = (eigen_value > 1e-16).sum()
        elif not self.n_components_:
            self.n_components_ = self.n_components_
        elif self.n_components_ > self.nobs_:
            raise ValueError("Error : You must pass a valid 'n_components'.")
        
        self.eig_ = np.array([eigen_value[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])
        
        self.eigen_vector_ = eigen_vector[:,:self.n_components_]

        self.coord_ = self.eigen_vector_*np.sqrt(eigen_value[:self.n_components_])

        self.res_dist_ = squareform(pdist(self.coord_,metric="euclidean"))

        #calcul du stress 
        if self.normalized_stress:
            self.stress_ = np.sqrt(np.sum((self.res_dist_-self.dist_)**2)/np.sum(self.dist_**2))
        else:
            self.stress_ = np.sum((self.res_dist_-self.dist_)**2)

        # Inertie 
        inertia = np.sum(self.dist_**2)/(2*self.nobs_**2)

        self.inertia_ = inertia
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        # Set labels
        self.labels_ = self.labels
        if self.labels_ is None:
            self.labels_ = [f"label_" + str(i+1) for i in np.arange(0,self.nobs_)]
        
        self.model_ = "cmdscale"

    def transform(self,X,y=None):
        """Apply the Multidimensional Scaling reduction on X

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
        
        d2 = np.sum(self.dist_**2,axis=1)/self.nobs_
        d3 = np.sum(self.dist_**2)/(self.nobs_**2)
        
        if self.proximity == "precomputed":
            sup_coord = mapply(X,lambda x : -(1/2)*(x**2 - np.sum(x**2)-d2+d3),axis=1,progressbar=False,n_workers=self.n_workers_).dot(self.coord_)/self.eig_[0]
        elif self.proximity == "euclidean":
            n_supp_obs = X.shape[0]
            sup_dist = np.zeros((n_supp_obs,self.nobs_))
            for i in np.arange(0,n_supp_obs):
                for j in np.arange(0,self.nobs_):
                    sup_dist[i,j] = euclidean(X.iloc[i,:],self.active_data_.iloc[j,:]) 
            sup_coord = np.apply_along_axis(arr=sup_dist,axis=1,func1d=lambda x : -(1/2)*(x**2 - np.sum(x**2)-d2+d3)).dot(self.coord_)/self.eig_[0]
        elif self.proximity == "similarity":
            raise NotImplementedError("Error : This method is not implemented yet.")
        
        return np.array(sup_coord)
        
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
        return self.coord_

