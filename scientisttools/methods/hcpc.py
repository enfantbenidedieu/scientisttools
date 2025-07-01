# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from mapply.mapply import mapply
from scipy.cluster import hierarchy

from sklearn.base import BaseEstimator, TransformerMixin
from .eta2 import eta2
from .splitmix import splitmix
from .revaluate_cat_variable import revaluate_cat_variable
from .auto_cut_tree import auto_cut_tree
from .quanti_var_desc import quanti_var_desc
from .quali_var_desc import quali_var_desc

class HCPC(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering on Principal Components (HCPC)
    ------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs an agglomerative hierarchical clustering on results from a factor analysis. Results include paragons, description of the clusters.

    Usage
    -----
    ```
    >>> HCPC(model, n_clusters=3,min_cluster = 3,max_cluster = None,metric="euclidean",method="ward",proba = 0.05,n_paragons = 5,order = True,parallelize = False)
    ```

    Parameters
    ----------
    `model` : an object of class PCA, MCA, FAMD

    `n_clusters` : an integer.  If a (positive) integer, the tree is cut with nb.cluters clusters. if None, the tree is automatically cut
    
    `min_cluster` : an integer. The least possible number of clusters suggested.

    `max_cluster` : an integer. The higher possible number of clusters suggested; by default the minimum between 10 and the number of individuals divided by 2.

    `metric` : the metric used to built the tree, default = "euclidean"

    `method` : the method used to built the tree, default = "ward"

    `proba` : the probability used to select axes and variables, default = 0.05

    `n_paragons` : an integer. The number of edited paragons.

    `order` : A boolean. If True, clusters are ordered following their center coordinate on the first axis.

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `call_` : a dictionary or parameters and internal objects.

    `cluster_` : a dictionary with clusters informations 

    `data_clust_` :  the original data with a supplementary column called clust containing the partition.

    `desc_var_` : the description of the classes by the variables.

    `desc_axes_` : the description of the classes by the factors (axes)

    `desc_ind_` : the paragons (para) and the more typical individuals of each cluster

    `model_` : a string specifying the model fitted = 'hcpc'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2008), Analyses Factorielles Simples et Multiples.4ed, Dunod<
    """
    def __init__(self,
                 model,
                 n_clusters=3,
                 min_cluster = 3,
                 max_cluster = None,
                 metric="euclidean",
                 method="ward",
                 proba = 0.05,
                 n_paragons = 5,
                 order = True,
                 parallelize = False):
        
        # Check if model 
        if model.model_ not in ["pca","mca","famd","pcamix","mpca","mfa","mfaqual","mfamix","mfact"]:
            raise TypeError("'model' must be an objet of class PCA,MCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX, MFACT")
        
        # Set parallelize 
        if parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Set linkage method
        if method is None:
            method = "ward"
        
        # Set linkage metric
        if metric is None:
            metric = "euclidean"
        
        # Set max cluster
        if max_cluster is None:
            max_cluster = min(10,round(model.ind_["coord"].shape[0]/2))
        else:
            max_cluster = min(max_cluster,model.ind_["coord"].shape[0]-1)
        
        # Set number of clusters
        if n_clusters is None:
            n_clusters = auto_cut_tree(model=model,min_clust=min_cluster,max_clust=max_cluster,method=method,metric=metric,order=order,weights=np.ones(model.ind_["coord"].shape[0]))
        elif not isinstance(n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")

        # Agglomerative clustering
        link_matrix = hierarchy.linkage(model.ind_["coord"],method=method,metric=metric)
        # cut the hierarchical tree
        cutree = (hierarchy.cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, )
        cluster = pd.Series([str(x) for x in cutree],index =  model.ind_["coord"].index.tolist(),name = "clust")

        # # Overall data
        data = model.call_["Xtot"]

        # Drop the supplementary individuals
        if hasattr(model,"ind_sup_"):
            data = data.drop(index=model.call_["ind_sup"])
        
        # Concatenate with cluster
        data_clust = pd.concat((data,cluster),axis=1)
        self.data_clust_ = data_clust

        # Tree elements
        tree = {"order":order,
                "linkage" : link_matrix,
                "height":link_matrix[:,2],
                "method":method,
                "metric" : metric,
                "merge":link_matrix[:,:2],
                "n_obs":link_matrix[:,3],
                "data": model.ind_["coord"],
                "n_clusters" : n_clusters}

        self.call_ = {"model" : model,"X" : data_clust,"tree" : tree}

        # Concatenate individuals coordinates with classe
        coord_classe = pd.concat([model.ind_["coord"], cluster], axis=1)
        # Count by cluster
        cluster_count = coord_classe.groupby("clust").size()
        cluster_count.name = "effectif"

        # Coordinates by cluster
        cluster_coord = coord_classe.groupby("clust").mean()

        # Value - test by cluster
        axes_mean =  model.ind_["coord"].mean(axis=0)
        axes_std = model.ind_["coord"].std(axis=0,ddof=0)
        cluster_vtest = mapply(cluster_coord,lambda x :np.sqrt(cluster.shape[0]-1)*(x-axes_mean.values)/axes_std.values,axis=1,progressbar=False,n_workers=n_workers)
        cluster_vtest = pd.concat(((cluster_vtest.loc[i,:]/np.sqrt((cluster.shape[0]-cluster_count.loc[i])/cluster_count.loc[i])).to_frame(i).T for i in cluster_count.index),axis=0)
        
        # Store cluster informations
        self.cluster_ = {"cluster" : cluster,"coord" : cluster_coord , "vtest" : cluster_vtest, "effectif" : cluster_count}

        # Distance to origin
        if model.model_ == "pca":
            cluster_var = pd.concat((model.call_["Z"],cluster),axis=1).groupby("clust").mean()
            cluster_dist2 = mapply(cluster_var,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            cluster_dist2.name = "dist"
            self.cluster_["dist"] =  np.sqrt(cluster_dist2)
        
        ## Axis description
        axes_desc = quanti_var_desc(X=model.ind_["coord"],cluster=cluster,weights=model.call_["ind_weights"],proba=proba,n_workers=n_workers)
        dim_clust = pd.concat((model.ind_["coord"],cluster),axis=1)

        axes_call = {"X" : dim_clust,"proba" : proba,"num_var" : dim_clust.shape[1]}
        self.desc_axes_ = {"quanti_var" : axes_desc[0],"quanti" : axes_desc[1], "call" : axes_call}
        
        #   Individuals description
        paragons = {}
        disto_far = {}
        for k in np.unique(cluster):
            group = coord_classe.query("clust == @k").drop(columns=["clust"])
            disto = mapply(group.sub(cluster_coord.loc[k,:],axis="columns"),lambda x : x**2,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            disto.name = "distance"
            paragons[f"Cluster : {k}"] = disto.sort_values(ascending=True).iloc[:n_paragons]
            disto_far[f"Cluster : {k}"] = disto.sort_values(ascending=False).iloc[:n_paragons]
        
        self.desc_ind_ = {"para" : paragons, "dist" : disto_far}

        # General Factor Analysis description
        data_call = {"X" : data_clust, "proba" : proba, "num_var" : data_clust.shape[1]}

        # Descriptive of variable (quantitative and qualitative)
        # Split data into tow
        X_quanti = splitmix(X=data)["quanti"]
        X_quali = splitmix(X=data)["quali"]
        desc_var = {}

        # Description of quantitatives variables
        if X_quanti is not None:
            quanti_var, quanti = quanti_var_desc(X=X_quanti,cluster=cluster,weights=model.call_["ind_weights"],proba=proba,n_workers=n_workers)
            desc_var = {**desc_var,**{"quanti_var" : quanti_var,"quanti" : quanti}}
        
        # Description of qualitatives variables
        if X_quali is not None:
            chi2_test, category = quali_var_desc(X=X_quali,cluster=cluster,proba=proba)
            desc_var = {**desc_var, **{"test_chi2" : chi2_test,"category" : category}}
        
        desc_var["call"] = data_call
        self.desc_var_ = desc_var

        # Modèle
        self.model_ = "hcpc"
