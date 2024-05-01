# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from mapply.mapply import mapply
from scientistmetrics import association
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform 
from sklearn.base import BaseEstimator, TransformerMixin

from .revaluate_cat_variable import revaluate_cat_variable


def funSqDice(col1,col2):
    return 0.5*np.sum((col1-col2)**2)

def funbothpos(col1,col2):
    return 1 - (1/len(col1))*np.sum(col1*col2)

class CATVARHCA(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis of Categorical Variables (VATVARHCA)
    ---------------------------------------------------------------------

    Parameters
    ----------

    n_clusters : an integer.  If a (positive) integer, the tree is cut with nb.cluters clusters.
                if None, n_clusters is set to 3
    
    var_sup : an integer or a list/tuple indicating the indexes of the supplementary individuals
    
    min_cluster : an integer. The least possible number of clusters suggested.

    max_cluster : an integer. The higher possible number of clusters suggested; by default the minimum between 10 and the number of individuals divided by 2.
 
    
    diss_metric : {"cramer","dice","bothpos"}
    """
    def __init__(self,
                 n_clusters=None,
                 var_sup = None,
                 diss_metric = "cramer",
                 metric="euclidean",
                 method="ward",
                 parallelize=False):
        self.n_clusters = n_clusters
        self.var_sup = var_sup
        self.diss_metric = diss_metric
        self.metric = metric
        self.method = method
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """
        
        """
        
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist())
        if not all_cat:
            raise TypeError("All actives columns must be categoricals")

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

        if self.var_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in var_sup])
        
        ########### Compute dissimilarity matrix
        
        # Compute Dissimilarity Matrix
        if self.diss_metric == "cramer":
            D = mapply(association(X=X,method="cramer"),lambda x : 1 - x, axis=0,progressbar=False,n_workers=n_workers)
        elif self.diss_metric in ["dice","bothpos"]:
            D = self._diss_modality(X)
        
         # Linkage method
        if self.method is None:
            method = "ward"
        else:
            method = self.method
        
        # Linkage metric
        if self.metric is None:
            metric = "euclidean"
        else:
            metric = self.metric

        # Set number of cluster
        if self.n_clusters is None:
            n_clusters = 3
        elif not isinstance(self.n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")
        else:
            n_clusters = self.n_clusters

        # Linkage matrix
        link_matrix = hierarchy.linkage(squareform(D),method=method,metric = metric)

        # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, )
        cluster = pd.Series([str(x) for x in cutree], index = D.index.tolist(),name = "clust")

        # Tree elements
        tree = {"linkage" : link_matrix,
                "height": link_matrix[:,2],
                "method": method,
                "metric" : metric,
                "merge":link_matrix[:,:2],
                "n_obs":link_matrix[:,3],
                "data": squareform(D),
                "n_clusters" : n_clusters}
        
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "tree" : tree}

        ################################### Informations abouts clusters
        data_clust = pd.concat((D,cluster),axis=1)
        # Count by cluster
        cluster_count = data_clust.groupby("clust").size()
        cluster_count.name = "effectif"

        # Store cluster informations
        self.cluster_ = {"cluster" : cluster,"data_clust" : data_clust ,"effectif" : cluster_count}

        # Model name
        self.model_ = "catvarhca"

        return self
    
    def _diss_modality(self,X):
        """Compute Distance matrix using Dice index ot
        
        """
        X = revaluate_cat_variable(X)
        # Disjonctif matrix
        M =  pd.concat((pd.get_dummies(X[cols],dtype=int) for cols in X.columns.tolist()),axis=1)

        # Compute Dissimilarity Matrix
        D = pd.DataFrame(index=M.columns,columns=M.columns).astype("float")
        for row in M.columns:
            for col in M.columns:
                if self.diss_metric == "dice":
                    D.loc[row,col] = np.sqrt(funSqDice(M[row].values,M[col].values))
                elif self.diss_metric == "bothpos":
                    D.loc[row,col] = funbothpos(M[row].values,M[col].values)
        # Specific to bothpos
        if self.diss_metric == "bothpos":
            np.fill_diagonal(D.values,0)
        return D

    def transform(self,X):
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


        if self.diss_metric == "cramer":
            # V de cramer 
            D = pd.DataFrame(index=self.cluster_["data_clust"].index,columns=X.columns).astype("float")
            for row in self.call_["X"].columns:
                for col in X.columns:
                    tab = pd.crosstab(self.call_["X"][row],X[col])
                    D.loc[row,col] = sp.stats.contingency.association(tab,method="cramer") 
        elif self.diss_metric in ["dice","bothpos"]:
            # Active data
            active_data  = revaluate_cat_variable(self.call_["X"])
            dummies = pd.concat((pd.get_dummies(active_data[cols],dtype=int) for cols in active_data.columns),axis=1)

            # Projected 
            if X.shape[1] == 1:
                dummies2 = pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep=" = ",dtype=int) for cols in X.columns),axis=1)
            else:
                X = revaluate_cat_variable(X)
                dummies2 = pd.concat((pd.get_dummies(X[cols],dtype=int) for cols in X.columns),axis=1)

            # Compute Dissimilarity Matrix
            D = pd.DataFrame(index=dummies.columns,columns=dummies2.columns).astype("float")
            for row in dummies.columns:
                for col in dummies2.columns:
                    if self.diss_metric == "dice":
                        D.loc[row,col] = funSqDice(dummies[row].values,dummies2[col].values)
                    elif self.diss_metric == "bothpos":
                        D.loc[row,col] = funbothpos(dummies[row].values,dummies2[col].values)
        # 
        if self.method in ["ward","average"]:
            corr_sup = pd.concat([D,self.cluster_["cluster"]],axis=1).groupby("clust").mean()
        elif self.method == "single":
            corr_sup = pd.concat([D,self.cluster_],axis=1).groupby("clust").min()
        elif self.method == "complete":
            corr_sup = pd.concat([D,self.cluster_],axis=1).groupby("clust").max()
        # Drop index name
        corr_sup.index.name = None
        return corr_sup.T