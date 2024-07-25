# -*- coding: utf-8 -*-
import pandas as pd
from mapply.mapply import mapply
from scipy.cluster import hierarchy
from sklearn.base import BaseEstimator, TransformerMixin

class VARHCPC(BaseEstimator,TransformerMixin):
    """
    Variables Hierachical Clustering on Principal Components (VARHCPC)
    ------------------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Hierarchical Clustering on variables using principal components.

    Usage
    -----
    ```
    >>> VARHCPC(model,n_clusters=None,metric="euclidean",method="ward",parallelize=False)
    ```

    Parameters
    -----------
    `model` : an object of class PCA, MCA

    `n_clusters` : an integer.  If a (positive) integer, the tree is cut with nb.cluters clusters.  If None, n_clusters is set to 3

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Return
    ------
    `call_` : dictionary with some statistics

    `cluster_` : dictionary with cluster statistics

    `model_` : string specifying the model fitted = 'varhcpc'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    def __init__(self,
                 model,
                 n_clusters=None,
                 metric = "euclidean",
                 method = "ward",
                 parallelize = False):
        
        # Weighted average with grouping
        def weighted_average(val_col_name, wt_col_name):
            def inner(group):
                return (group[val_col_name] * group[wt_col_name]).sum() / group[wt_col_name].sum()
            inner.__name__ = 'weighted_averages'
            return inner
        
        if model.model_ not in ["pca","mca"]:
            raise TypeError("'model' must be an objet of class 'PCA','MCA'")
        
        # Set parallelize
        if parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Extract principal components
        X = model.var_["coord"]

        # Set method
        if method is None:
            method = "ward"
        
        # Set metric
        if metric is None:
            metric = "euclidean"

        # Set n_clusters
        if n_clusters is None:
            n_clusters = 3
        elif not isinstance(n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")
        
        # Linkage matrix
        link_matrix = hierarchy.linkage(X,method=method,metric = metric)

        # cut the hierarchical tree
        cutree = (hierarchy.cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, )
        cluster = pd.Series([str(x) for x in cutree], index =  X.index,name = "clust")
        
        ## Concatenate
        data_clust = pd.concat((X,cluster),axis=1)
        self.data_clust_ = data_clust

        # Tree elements
        tree = {"linkage" : link_matrix,
                "height":link_matrix[:,2],
                "method":method,
                "metric" : metric,
                "merge":link_matrix[:,:2],
                "n_obs":link_matrix[:,3],
                "data": X,
                "n_clusters" : n_clusters}

        self.call_ = {"model" : model,"X" : data_clust,"tree" : tree}

        # Cluster example
        cluster_count = data_clust.groupby("clust").size()
        cluster_count.name = "effectif"
        
        if model.model_ == "pca":
            cluster_coord = data_clust.groupby("clust").mean()
        elif model.model_ == "mca":
            # Weight of categories multiply by number of variables
            weight = model.var_["infos"].loc[:,"Weight"]*model.var_["eta2"].shape[0]
            coord_classe = pd.concat([weight,data_clust], axis=1)
            cluster_coord = pd.concat((mapply(coord_classe.groupby("clust"),
                                                weighted_average(col,"Weight"),axis=0,n_workers=n_workers,progressbar=False).to_frame(col) for col in X.columns),axis=1)
        
        # Store cluster informations
        self.cluster_ = {"cluster" : cluster,"coord" : cluster_coord ,"effectif" : cluster_count}
        
        # Modèle
        self.model_ = "varhcpc"