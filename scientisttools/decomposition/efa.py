# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EFA(BaseEstimator,TransformerMixin):
    """Exploratory Factor Analysis
    
    """
    def __init__(self,
                normalize=True,
                n_components = None,
                row_labels = None,
                col_labels = None,
                method = "principal",
                row_sup_labels = None,
                quanti_sup_labels = None,
                quali_sup_labels = None,
                graph =None,
                figsize=None):
        self.normalize = normalize
        self.n_components =n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.method = method
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.graph = graph
        self.figsize= figsize

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
        
        self.data_ = X

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
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

        