# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform 
from sklearn.base import BaseEstimator, TransformerMixin

from .covariance_to_correlation import covariance_to_correlation

class VARHCA(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis of Continuous Variables (VARHCA)
    -----------------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs an agglomerative hierarchical clustering on continuous variables.

    Parameters
    ----------

    n_clusters : an integer.  If a (positive) integer, the tree is cut with nb.cluters clusters.
                if None, n_clusters is set to 3
    
    var_sup : an integer or a list/tuple indicating the indexes of the supplementary individuals
    
    matrix_type : Three choices  
                    - "completed" for original data
                    - "correlation" for pearson correlation matrix
                    - "covariance" for covariance matrix

    metric : The metric used to built the tree, default = "euclidean"

    method : The method used to built the tree, default = "ward"

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply
    
    Return
    ------

    call_ : A list or parameters and internal objects.

    cluster_ : a dictionary with clusters informations 

    corr_sup_ : conditional mean

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    R. Rakotomalala, « Classification de variables », Tutoriels Tanagra pour le Data Mining
    """
    def __init__(self,
                 n_clusters=3,
                 var_sup = None,
                 matrix_type = "completed",
                 metric = "euclidean",
                 method = "ward",
                 parallelize=False):
        self.n_clusters = n_clusters
        self.var_sup = var_sup
        self.matrix_type = matrix_type
        self.metric = metric
        self.method = method
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : pandas/polars DataFrame of float, shape (n_rows, n_columns) or (n_columns, n_columns)

        y : None
            y is ignored

        Returns:
        --------
        self : object
                Returns the instance itself
        """

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check of matrix type is one of 'completed' or 'correlation'
        if self.matrix_type not in ["completed","correlation","covariance"]:
            raise ValueError("'matrix_type' should be one of 'completed', 'correlation', 'covariance'")

        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        if not all_num:
            raise TypeError("All columns must be numeric")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        #  Check if supplementary variables
        if self.var_sup is not None:
            if (isinstance(self.var_sup,int) or isinstance(self.var_sup,float)):
                var_sup = [int(self.var_sup)]
            elif ((isinstance(self.var_sup,list) or isinstance(self.var_sup,tuple))  and len(self.var_sup)>=1):
                var_sup = [int(x) for x in self.var_sup]
            
        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

        ####################################### Drop supplementary variables columns ########################################
        if self.var_sup is not None:
            if self.matrix_type == "completed":
                X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in var_sup])
            elif self.matrix_type == "correlation":
                X = (X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in var_sup])
                      .drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in var_sup]))
            elif self.matrix_type == "covariance":
                X = covariance_to_correlation(X)
                X = (X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in var_sup])
                      .drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in var_sup]))

        ##################################### Compute Pearson correlation matrix ##############################################
        if self.matrix_type == "completed":
            corr_matrix = X.corr(method="pearson")
        elif self.matrix_type in ["correlation","covariance"]:
            corr_matrix = X

        # Linkage matrix
        if self.method is None:
            method = "ward"
        else:
            method = self.method
        
        ########################## metrics
        if self.metric is None:
            metric = "euclidean"
        else:
            metric = self.metric
        
        ################## Check numbers of clusters
        if self.n_clusters is None:
            n_clusters = 3
        elif not isinstance(self.n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")
        else:
            n_clusters = self.n_clusters
        
        # Compute dissimilary matrix : sqrt(1 - x**2)
        D = mapply(corr_matrix,lambda x : np.sqrt(1 - x**2),axis=0,progressbar=False,n_workers=n_workers)

        # Linkage Matrix with vectorize dissimilarity matrix
        link_matrix = hierarchy.linkage(squareform(D),method=method,metric = metric)

         # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, )
        cluster = pd.Series([str(x) for x in cutree], index = corr_matrix.index.tolist(),name = "clust")

        # Tree elements
        tree = {"linkage" : link_matrix,
                "height": link_matrix[:,2],
                "method": method,
                "metric" : metric,
                "merge":link_matrix[:,:2],
                "n_obs":link_matrix[:,3],
                "data": corr_matrix,
                "n_clusters" : n_clusters}
        
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "tree" : tree}

        ################################### Informations abouts clusters
        data_clust = pd.concat((corr_matrix,cluster),axis=1)
        # Count by cluster
        cluster_count = data_clust.groupby("clust").size()
        cluster_count.name = "effectif"

        # Store cluster informations
        self.cluster_ = {"cluster" : cluster,"data_clust" : data_clust ,"effectif" : cluster_count}

        ################## 
        if self.var_sup is not None:
            if self.matrix_type == "completed":
                X_sup = Xtot.iloc[:,var_sup].astype("float")
                # Compute correlation between 
                corr_sup = np.corrcoef(X,X_sup,rowvar=False)[:X.shape[1],X.shape[1]:]
                corr_sup = pd.DataFrame(corr_sup,index=X.columns.tolist(),columns=X_sup.columns.tolist())
            elif self.matrix_type == "correlation":
                corr_sup = Xtot.iloc[:X.shape[1],var_sup]
            elif self.matrix_type == "covariance":
                corr_sup = covariance_to_correlation(Xtot).iloc[:X.shape[1],var_sup]
             
            #moyenne des carrés des corrélations avec les groupes
            corr_mean_sup = mapply(pd.concat([corr_sup,self.cluster_["cluster"]],axis=1).groupby("clust"),
                                      lambda x : np.mean(x**2,axis=0),progressbar=False,n_workers=n_workers)
            corr_mean_sup.index.name = None
            self.corr_sup_ = corr_mean_sup.T
        # Model name
        self.model_ = "varhca"

        return self
        
    def transform(self,X,y=None):
        """
        
        
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        if not all_num:
            raise TypeError("All columns must be numeric")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        if self.matrix_type == "completed":
            corr_with = pd.DataFrame(np.corrcoef(self.call_["X"],X,rowvar=False)[:self.call_["X"].shape[1],self.call_["X"].shape[1]:],
                                     index = self.call_["X"].columns.tolist(),columns=X.columns.tolist())
        elif self.matrix_type == "correlation":
            corr_with = X
        else:
            raise ValueError("Not implemented")
        
        #moyenne des carrés des corrélations avec les groupes
        corr_mean_square = mapply(pd.concat([corr_with,self.cluster_["cluster"]],axis=1).groupby("clust"),
                                  lambda x : np.mean(x**2,axis=0),progressbar=False,n_workers=n_workers)
        corr_mean_square.index.name = None
        return corr_mean_square.T
    
    def fit_transform(self,X,y=None):
        """
        
        """
        self.fit(X)
        return self.cluster_["cluster"]