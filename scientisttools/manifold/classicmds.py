# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from scientisttools.utils import sim_dist
import warnings
from sklearn.utils import check_symmetric
from sklearn.base import BaseEstimator, TransformerMixin
from scientisttools.graphics import plotCMDS

class CMDSCALE(BaseEstimator,TransformerMixin):
    """Classic Muldimensional Scaling (CMDSCALE)
    
    """
    def __init__(self,
                n_components=None,
                labels = None,
                proximity="euclidean",
                normalized_stress=True,
                graph=True,
                figsize=None):
        self.n_components = n_components
        self.labels = labels
        self.proximity = proximity
        self.normalized_stress = normalized_stress
        self.graph = graph
        self.figsize = figsize
    
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
        
        self.nobs_ = X.shape[0]
        self.centering_matrix_ = np.identity(self.nobs_) - (1/self.nobs_)*np.ones(shape=(self.nobs_,self.nobs_))
        
        self._compute_stats(X)

        if self.graph:
            fig, axe = plt.subplots(figsize=self.figsize)
            plotCMDS(self,repel=True,ax=axe)
        
        return self
    
    def _is_euclidean(self,X):
        """
        
        """
        self.dist_ = squareform(pdist(X,metric="euclidean"))
        B = np.dot(np.dot(self.centering_matrix_,X),np.dot(self.centering_matrix_,X).T)
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _is_precomputed(self,X):
        """
        
        """
        self.dist_ = check_symmetric(X.values, raise_exception=True)
        A = -0.5*np.multiply(self.dist_,self.dist_)
        B = np.dot(self.centering_matrix_,np.dot(A,self.centering_matrix_))
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _is_similarity(self,X):
        """
        
        """
        D = sim_dist(X)
        self.dist_ = check_symmetric(D, raise_exception=True)
        A = -0.5*np.multiply(self.dist_,self.dist_)
        B = np.dot(self.centering_matrix_,np.dot(A,self.centering_matrix_))
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _compute_stats(self,X):
        """
        
        """
        if X.shape[0] == X.shape[1] and self.proximity != "precomputed":
            raise warnings.warn(
                "The ClassicMDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``dissimilary='precomputed'``."
            )

        # Compute euclidean
        if self.proximity == "euclidean":
            eigen_value, eigen_vector = self._is_euclidean(X)
        elif self.proximity == "precomputed":
            eigen_value, eigen_vector = self._is_precomputed(X)
        elif self.proximity == "similarity" :
            eigen_value, eigen_vector = self._is_similarity(X)
        else:
            raise ValueError("You must pass a valid 'proximity'.")
        
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
        
        self.model_ = "cmds"

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
        
        raise NotImplementedError("Error : This method is not implemented yet.")


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

