# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from mapply.mapply import mapply
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.base import BaseEstimator, TransformerMixin
from scientisttools.graphics import plotPPCA
from adjustText import adjust_text
import pingouin as pg
from scientisttools.utils import global_kmo_index,per_item_kmo_index
from scientisttools.decomposition import PCA
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

class PartialPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Components Analysis
    """
    def __init__(self,
                n_components=None,
                normalize=True,
                row_labels=None,
                col_labels=None,
                partial_labels=None,
                graph = False,
                figsize=None):
        self.n_components = n_components
        self.normalize = normalize
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.partial_labels = partial_labels
        self.graph = graph
        self.figsize = figsize
    
    def fit(self,X,y=None):
        """
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")


        self.n_rows_ = X.shape[0]
        self.n_cols_ = X.shape[1]
        self.data_ = X

        self._compute_stats(X)
        self._compute_svds(X)

        if self.graph:
            fig,(axe1,axe2) = plt.subplots(1,2,figsize=self.figsize)
            plotPPCA(self,choice="ind",ax=axe1)
            plotPPCA(self,choice="var",ax=axe2)

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
        normalized_data = mapply(self.data_,lambda x : (x - x.mean())/x.std(),axis=0,progressbar=False)
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
        self.eigen_vectors_ = my_pca.eigen_vectors_
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