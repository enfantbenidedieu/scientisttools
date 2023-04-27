# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scientisttools.manifold import SMACOF
from scipy.spatial.distance import pdist,squareform
from sklearn.utils import check_symmetric
from scientisttools.utils import sim_dist
from scientisttools.graphics import plotMDS
from sklearn.base import BaseEstimator, TransformerMixin
from adjustText import adjust_text

class MDS(BaseEstimator,TransformerMixin):
    """Metric and Non - Metric Multidimensional Scaling (MDS)
    
    """
    def __init__(self,
                n_components=2,
                proximity ='euclidean',
                metric=True,
                n_init=4,
                max_iter=300,
                verbose=0,
                eps=1e-3,
                n_jobs=None,
                random_state=None,
                labels = None,
                normalized_stress=True,
                graph =True,
                figsize=(10,10)):
        self.n_components = n_components
        self.proximity = proximity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.labels = labels
        self.normalized_stress =normalized_stress
        self.graph = graph
        self.figsize = figsize
    
    def fit(self,X,y=None, init=None):
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
        

        if self.proximity == "euclidean":
            self.dist_ = squareform(pdist(X,metric="euclidean"))
        elif self.proximity == "precomputed":
            self.dist_ = check_symmetric(X.values, raise_exception=True)
        elif self.proximity == "similarity":
            self.dist_ = sim_dist(X)

        #Set Labels
        self.labels_ = self.labels
        if self.labels_ is None:
            self.labels_ = ["label_"+str(i+1) for i in range(0,X.shape[0])]
        
        if self.metric:
            self.title_ = "Metric multidimensional scaling (mMDS)"
        else:
            self.title_ = "Non-metric multidimensional scaling (NMDS)"

        self.fit_transform(X,init=init)

        self.res_dist_ = squareform(pdist(self.coord_,metric="euclidean"))

        #calcul du stress 
        if self.normalized_stress:
            self.stress_ = np.sqrt(np.sum((self.res_dist_-self.dist_)**2)/np.sum(self.dist_**2))
        else:
            self.stress_ = np.sum((self.res_dist_-self.dist_)**2)

        self.model_ = "mds"

        if self.graph:
            fig, axe = plt.subplots(figsize=self.figsize)
            plotMDS(self,repel=True,ax=axe)

        return self
    
    def fit_transform(self,X, y=None, init=None):
        """
        
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        

        self.coord_, self.stress_, self.n_iter_ = SMACOF(
            X=X,
            metric=self.metric,
            n_components=self.n_components,
            proximity = self.proximity, 
            init=init,
            n_init=self.n_init, 
            n_jobs=self.n_jobs, 
            max_iter=self.max_iter, 
            verbose=self.verbose,
            eps=self.eps, 
            random_state=self.random_state, 
            return_n_iter=True,
            )
        #set n_compoents
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = self.coord_.shape[1]
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        return self.coord_
    
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