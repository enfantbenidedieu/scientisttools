# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from mapply.mapply import mapply
from scientisttools.graphics import plotCA
from sklearn.base import BaseEstimator, TransformerMixin

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)

class CA(BaseEstimator,TransformerMixin):
    """ Correspondence Analysis (CA)
    
    This class inherits from the Base class.
    
    CA performs a Correspondence Analysis, given a contingency table
    containing absolute frequencies ; shape= n_rows x n_columns.
    This implementation only works for dense arrays.
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
        
    row_labels : array of strings or None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
    
    col_labels : array of strings or None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.

    row_sup_labels : array of strings or None
        - If row_sup_labels is an array of strings : this array provides the
          supplementary row labels.
    
    col_sup_labels : array of strings or None
        - If col_sup_labels is an array of strings : this array provides the
          supplementary columns labels.

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.
    
    row_labels_ : array of strings
        Labels for the rows.
    
    col_labels_ : array of strings
        Labels for the columns.
    
    eig_ : array of float
        A 4 x n_components_ matrix containing all the eigenvalues
        (1st row), difference (2nd row), the percentage of variance (3th row) and the
        cumulative percentage of variance (4th row).
    
    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.
    
    col_coord_ : array of float
        A n_columns x n_components_ matrix containing the column
        coordinates.
        
    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    col_contrib_ : array of float
        A n_columns x n_components_ matrix containing the column
        contributions.
    
    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_cos2_ : array of float
        A n_columns x n_components_ matrix containing the column
        cosines.
    total_ : float
        The sum of the absolute frequencies in the X array.
    
    model_ : string
        The model fitted = 'ca'
    """
    
    def __init__(self,
                 n_components=None,
                 row_labels=None,
                 col_labels=None,
                 row_sup_labels=None,
                 col_sup_labels=None,
                 graph=True,
                 figsize=None):
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.row_sup_labels = row_sup_labels
        self.col_sup_labels = col_sup_labels
        self.graph = graph
        self.figsize = figsize
    
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
        
        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary columns
        self.col_sup_labels_ = self.col_sup_labels 
        if self.col_sup_labels is not None:
            X_= _X.drop(columns = self.col_sup_labels_)
            col_sup = _X[self.col_sup_labels_]
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.col_sup_labels_)
        else:
            X_ = _X
        
        self.data_ = X
        
        # Supplementary initialization
        self.row_sup_coord_ = None
        self.row_sup_cos2_ = None

        self.col_sup_coord_ = None
        self.col_sup_cos2_ = None

        self.n_rows_, self.n_cols_ = X_.shape
        self.total_ = X_.sum().sum()

        # Computes Singular Values Decomposition
        self._compute_svd(X=X_)
        
        # Computes Dependance indicators
        self._compute_indicators(X_)

        if self.row_sup_labels is not None:
            self._compute_sup(X=row_sup,row=True)
        
        if self.col_sup_labels is not None:
            self._compute_sup(X=col_sup,row=False)
        
        if self.graph:
            fig, (axe1,axe2) = plt.subplots(1,2,figsize=self.figsize)
            plotCA(self,choice = "row",ax=axe1,repel=True)
            plotCA(self,choice = "col",ax=axe2,repel=True)

        return self
    
    def _compute_stats(self,rowprob,colprob,rowdisto,coldisto):

        row_contrib = np.apply_along_axis(func1d=lambda x : x/self.eig_[0], axis=1,
                        arr=np.apply_along_axis(func1d=lambda x: 100*x**2*rowprob,axis=0,arr=self.row_coord_))
        col_contrib = np.apply_along_axis(func1d=lambda x : x/self.eig_[0], axis=1,
                        arr=np.apply_along_axis(func1d=lambda x: 100*x**2*colprob,axis=0,arr=self.col_coord_))

        # 
        row_cos2 = np.apply_along_axis(func1d=lambda x: x**2/rowdisto, axis = 0, arr=self.row_coord_)
        col_cos2 = np.apply_along_axis(func1d=lambda x: x**2/coldisto, axis = 0, arr=self.col_coord_)
    
        self.row_contrib_ = row_contrib[:,:self.n_components_]
        self.col_contrib_ = col_contrib[:,:self.n_components_]
        self.row_cos2_ = row_cos2[:,:self.n_components_]
        self.col_cos2_ = col_cos2[:,:self.n_components_]

    def _compute_indicators(self,X):
        """
        """
        # 
        prob_conj = mapply(X,lambda x : x/self.total_,axis=0,progressbar=False)

        # probabilité marginale de V1 - marge colonne
        row_prob = prob_conj.sum(axis = 1)

        # Marge ligne (probabilité marginale)
        col_prob = prob_conj.sum(axis = 0)

        # Totaux lignes
        row_sum = X.sum(axis=1)

        # Totaux colonnes
        col_sum = X.sum(axis=0)

        # Compute chi - squared test
        statistic,pvalue,dof, _ = st.chi2_contingency(X, lambda_=None)

        # log - likelihood - tes (G - test)
        g_test_res = st.chi2_contingency(X, lambda_="log-likelihood")

        # Residuaal
        resid = X - self.expected_freq_

        standardized_resid = pd.DataFrame(self.standardized_resid_,index=self.row_labels_,columns=self.col_labels_)

        adjusted_resid = mapply(mapply(standardized_resid,lambda x : x/np.sqrt(1 - col_prob),axis=1,progressbar=False),
                                lambda x : x/np.sqrt(1-row_prob),axis=0,progressbar=False)
        
        chi2_contribution = mapply(standardized_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False)
        # 
        attraction_repulsion_index = X/self.expected_freq_

        # Profils lignes
        row_prof = mapply(prob_conj,lambda x : x/np.sum(x), axis=1,progressbar=False)
        
        ## Profils colonnes
        col_prof = mapply(prob_conj,lambda x : x/np.sum(x), axis=0,progressbar=False)

        # Row distance
        row_dist = squareform(pdist(row_prof,metric= "seuclidean",V=col_prob)**2)
        
        # Distance entre individus et l'origine
        row_disto = mapply(row_prof,lambda x :np.sum((x-col_prob)**2/col_prob),axis = 1,progressbar=False)

        # Poids des observations
        row_weight = row_sum/np.sum(row_sum)
        # Inertie des lignes
        row_inertie = row_disto*row_weight
        # Affichage
        row_infos = np.c_[row_disto, row_weight, row_inertie]
        
        ###################################################################################
        #               Informations sur les profils colonnes
        ###################################################################################

        col_dist = squareform(pdist(col_prof.T,metric= "seuclidean",V=row_prob)**2)

        # Distance à l'origine
        col_disto = mapply(col_prof.T,lambda x : np.sum((x-row_prob)**2/row_prob),axis = 1,progressbar=False)

        # Poids des colonnes
        col_weight = col_sum/np.sum(col_sum)

        # Inertie des colonnes
        col_inertie = col_disto*col_weight
        # Affichage
        col_infos = np.c_[col_disto, col_weight, col_inertie]

        inertia = np.sum(row_inertie)

        # 
        self._compute_stats(row_prob,col_prob,row_disto,col_disto)

        # Return indicators
        self.chi2_test_ = dict({"statistic" : statistic,"pvalue":pvalue,"dof":dof})
        self.log_likelihood_test_ = dict({"statistic" : g_test_res[0],"pvalue":g_test_res[1]})
        self.contingency_association_ = dict({"cramer" : st.contingency.association(X, method="cramer"),
                                              "tschuprow" : st.contingency.association(X, method="tschuprow"),
                                              "pearson" : st.contingency.association(X, method="pearson")})
        self.resid_ = resid
        self.row_infos_ = row_infos
        self.col_infos_ = col_infos
        self.adjusted_resid_ = adjusted_resid
        self.chi2_contribution_ = chi2_contribution
        self.attraction_repulsion_index_ = attraction_repulsion_index
        self.inertia_ = inertia
        self.row_dist_ = row_dist
        self.col_dist_ = col_dist
    
    def _compute_svd(self,X):
        """"Compute a Singular Value Decomposition

        Then, this function computes :
            n_components_ : 
        """
        # Set row labels
        self.row_labels_ = self.row_labels
        if (self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]
        
        # Set col labels
        self.col_labels_ = self.col_labels
        if (self.col_labels_ is None) or (len(self.col_labels_) !=self.n_cols_):
            self.col_labels_ = ["col_" + str(k+1) for k in np.arange(0,self.n_cols_)]
        
        # Expected frequency
        self.expected_freq_ = st.contingency.expected_freq(X)
        
        # Standardized resid
        self.standardized_resid_ = (X - self.expected_freq_)/np.sqrt(self.expected_freq_)

        # Singular Values Decomposition
        U, delta, V_T = np.linalg.svd(self.standardized_resid_/np.sqrt(self.total_),full_matrices=False)

        # Eigenvalues
        lamb = delta**2

        f_max = min(self.n_rows_ -1,self.n_cols_ - 1)
        eigen_values = lamb[:f_max]
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # 
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = (delta > 1e-16).sum()
        
        self.eig_ = np.array([eigen_values[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])
        row_weight = X.sum(axis=1)/self.total_
        col_weight = X.sum(axis=0)/self.total_

        row_coord = np.apply_along_axis(func1d=lambda x : x/np.sqrt(row_weight),axis=0,arr=U[:,:f_max]*delta[:f_max])
        
        col_coord = np.apply_along_axis(func1d=lambda x : x/np.sqrt(col_weight),axis=0,arr=V_T[:f_max,:].T*delta[:f_max])
        #self.data_ = np.array(X)
        self.row_coord_ = row_coord[:,:self.n_components_]
        self.col_coord_ = col_coord[:,:self.n_components_]
        self.dim_index_ = ["Dim."+str(i+1) for i in np.arange(0,self.n_components_)]
        self.kaiser_threshold_ = np.mean(eigen_values)
        self.kaiser_proportion_threshold_ = 100/f_max
        self.res_row_dist_ = squareform(pdist(self.row_coord_,metric="sqeuclidean"))
        self.res_col_dist_ = squareform(pdist(self.col_coord_,metric="sqeuclidean"))

        self.model_ = "ca"
    
    def _compute_sup(self,X,row=True):
        """Compute row/columns supplementary coordinates
        
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if row:
            row_sup_prof = np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=1,arr=X).dot(self.col_coord_)/np.sqrt(self.eig_[0])
            self.row_sup_coord_ = row_sup_prof[:,:self.n_components_]
        else:
            col_sup_prof = np.transpose(np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=0,arr=X)).dot(self.row_coord_)/np.sqrt(self.eig_[0])
            self.col_sup_coord_ = col_sup_prof[:,:self.n_components_]

    
    def transform(self,X,y=None,row=True):
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

        if row: 
            row_sup_prof = np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=1,arr=X)
            return row_sup_prof.dot(self.col_coord_) / np.sqrt(self.eig_[0])
        else:
            col_sup_prof = np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=0,arr=X)
            return col_sup_prof.T.dot(self.row_coord_)/np.sqrt(self.eig_[0])
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
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

        return self.row_coord_
    