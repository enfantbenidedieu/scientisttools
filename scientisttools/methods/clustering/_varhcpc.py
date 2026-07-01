# -*- coding: utf-8 -*-
from numpy import c_, average, array, ndarray
from pandas import DataFrame, concat, Series
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict, namedtuple
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# interns functions
from ..functions.utils import check_is_dataframe

class VARHCPC(BaseEstimator,TransformerMixin):
    """
    Variables Agglomerative Hierachical Clustering on Principal Components (VARHCPC)
    
    Performs agglomerative hierarchical clustering on continuous variables using principal components.

    Parameters
    -----------
    ncl : int.  default = 3
        If a (positive) integer, the tree is cut with nb_cluters clusters. 
        if None, the tree is automatically cut.

    consol : bool, default = False
        If True, a k-means consolidation is performed after agglomerative hierarchical clustering.

    max_iter : int, default = 300
        The maximum number of iterations for the consolidation.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    metric : str, default = "euclidean"
        The metric used to built the tree. It must be one of the options allowed by `scipy.spatial.distance.pdist` for its metric parameter, or a metric listed in :func:`sklearn.metrics.pairwise.distance_metrics`.

    method : str, default = "ward"
        the method used to built the tree. 

    **kwargs : key words parameters
        Additionals parameters for sklearn.cluster.KMeans.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        X : DataFrame of shape (n_samples, n_components)
            Coordinates of continuous variables, where ``n_columns`` is the number of continuous variables and ``n_components`` is the number of components.

        ncl : int
            The number of clusters.

        tree : tree
            The results for the hierarchical tree.

        km : class, optional
            The results of k-means.

    cluster_ : cluster
        An object containing the results of the clusters, with the following attributes:

        coord : DataFrame of shape (n_clusters, ncp)
            The coordinates of the clusters - cluster centers
    
    quanti_var_ : quanti_var
        An object containing the description of the clusters by the continuous variables, with the following attributes:

        cluster : Series of shape (n_samples,)
            The labels of variables.
        dist : DataFrame of shape (n_samples, n_clusters)
            The distance of variables to the cluster centers.
        member : DataFrame of shape (n_samples, 3)
            Cluster's members of variables (distance to own cluster, distance to next closest, ratio (own/next)).

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing the description of the clusters by the supplementary continuous variables, with the following attributes:

        cluster : Series of shape (n_samples_sup,)
            The labels of supplementary variables.
        dist : DataFrame of shape (n_samples_sup, n_clusters)
            The distance of supplementary variables to the cluster centers.
        member : DataFrame of shape (n_samples_sup, 3)
            Cluster's members of supplementary variables (distance to own cluster, distance to next closest, ratio (own/next)).

    References
    ----------
    [1] R. Rakotomalala, « Classification de variables », Tutoriels Tanagra pour le Data Mining.

    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientistools import PCA, VARHCPC
    >>> # HCPC after PCA
    >>> clf = PCA(ncp=5,ind_sup=range(41,46),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> clf2 = VARHCPC(n_clusters=3)
    >>> clf2.fit(clf)
    """
    def __init__(
            self, ncl=3, consol=True, max_iter=300, random_state=0, metric = "euclidean", method = "ward", **kwargs
    ):
        self.ncl = ncl
        self.consol = consol
        self.max_iter = max_iter
        self.random_state = random_state
        self.metric = metric
        self.method = method
        self.kwargs = kwargs
        
    def fit(self,obj,y=None,sample_weight=None):
        """
        Compute agglomerative hierarchical clustering with ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : 1d array-like of shape (n_samples,), default = None
            An optional sample weights. The weights for each observation.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if linkage method is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.method in ("average","complete","single","ward")):
            raise ValueError("'method' should be one of 'average', 'complete', 'single', 'ward'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if obj is an object of class PCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (obj.__class__.__name__ in ("PCA","FAMD","PCAmix","MPCA","MFA")):
            raise TypeError("'obj' must be an object of class PCA, FAMD, PCAmix, MPCA, MFA")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # agglomerative hierarchical clustering on principal component
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # extract principal components
        X = obj.quanti_var_.coord
        # copy of principal components
        D = X.copy()

        # number of columns
        n_cols = X.shape[0]
        # weighted of continuous variables extract from factor analysis
        if sample_weight is None: 
            w = obj.call_.col_w.loc[X.index]
        elif not isinstance(sample_weight,(list,tuple,ndarray,Series)): 
            raise TypeError("'sample_weight' must be a 1d array-like of sample weights.")
        elif len(sample_weight) != n_cols: 
            raise ValueError(f"'sample_weight' must be a 1d array-like of shape ({n_cols},).")
        else: 
            w = Series(array(sample_weight),index=X.index,name="weight")

        # linkage matrix
        Z = linkage(D,method=self.method,metric=self.metric)
        # height
        height = (DataFrame(c_[list(range(1,Z.shape[0]+1)),Z[:,2][::-1]],columns=["cluster","height"]).
                  assign(
                      diff_1 = lambda x : -1*x["height"].diff(1),
                      diff_2 = lambda x : x["diff_1"].diff(-1)
                  ))
        height["cluster"] = height["cluster"].astype(int)

        #convert to ordered dictionary
        tree_ = OrderedDict(D=D,Z=Z,height=height,merge=Z[:,:2],size=Z[:,3])
        #convert to namedtuple
        tree = namedtuple("tree",tree_.keys())(*tree_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set numbers of clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncl is None:
            ncl = height[height["diff_2"]==height["diff_2"].max()]["cluster"].values[0]
        elif self.ncl < 0:
            raise TypeError("'ncl' should be a positive integer.")
        elif not isinstance(self.ncl,int):
            raise TypeError("'ncl' should be an integer")
        else:
            ncl = self.ncl
        
        # assign cluster
        cluster = Series((cut_tree(Z,n_clusters=ncl)+1).reshape(-1, ), index = D.index,name = "cluster",dtype="category")
        # unique cluster
        uq_cluster = sorted(list(cluster.unique()))
        # coordinates of the clusters - cluster centers
        cluster_coord = DataFrame(index=uq_cluster,columns=X.columns).astype(float)
        for i in uq_cluster:
            ix = list(cluster[cluster==i].index)
            cluster_coord.loc[i,:] = average(a=X.loc[ix,:],axis=0,weights=w.loc[ix])

        # convert to ordered dictionary
        call_ = OrderedDict(obj=obj,X=X,w=w,ncl=ncl,tree=tree)

        # consolidation
        if self.consol:
            # K-means clustering
            km = KMeans(n_clusters=ncl,init=cluster_coord,max_iter=self.max_iter,random_state=self.random_state,**self.kwargs).fit(X=X,sample_weight=w)
            # assign cluster
            cluster = Series(array(km.labels_)+1, index = D.index, name = "cluster", dtype="category")
            # coordinates of the clusters - cluster centers
            cluster_coord = DataFrame(km.cluster_centers_,index=list(range(1,ncl+1)),columns=km.feature_names_in_)
            cluster_coord.index = cluster_coord.index.astype("category")
            # add to dictionary
            call_["km"] = km
        # concatenate principal components with cluster
        call_["data_clust"] = concat((X,cluster),axis=1)
        # convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to ordered dictionary
        cluster_ = OrderedDict(coord=cluster_coord)
        #convert to namedtuple
        self.cluster_ = namedtuple("cluster",cluster_.keys())(*cluster_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # distance of continuous to cluster centers
        dist_cluster_center = DataFrame(squareform(pdist(concat((X,cluster_coord),axis=0),metric=self.metric))[:n_cols,n_cols:],index=D.index,columns=uq_cluster)
        # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
        cluster_member = DataFrame(index=D.index,columns=["Own Cluster","Next Closest"]).astype(float)
        cluster_member["Own Cluster"] = dist_cluster_center.min(axis=1)
        cluster_member["Next Closest"] = dist_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        cluster_member["Ratio (Own/Next)"] = cluster_member["Own Cluster"]/cluster_member["Next Closest"]
        # convert to ordered dictionary
        quanti_var_ = OrderedDict(cluster=cluster,dist=dist_cluster_center,member=cluster_member)
        # convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for supplementary continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"quanti_var_sup_"):
            # coordinates for supplementary continuous
            X_sup = obj.quanti_var_sup_.coord
            # number of supplementary continuous variables
            n_cols_sup = X_sup.shape[0]
            # distance for supplementary continuous variables to cluster centers
            dist_sup_cluster_center = DataFrame(squareform(pdist(concat((X_sup,cluster_coord),axis=0),metric=self.metric))[:n_cols_sup,n_cols_sup:],index=X_sup.index,columns=uq_cluster)
            # assign cluster to supplementary continuous variables
            quanti_var_sup_cluster = dist_sup_cluster_center.idxmin(axis=1).astype("category")
            quanti_var_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = DataFrame(index=X_sup.index,columns=["Own Cluster","Next Closest"]).astype(float)
            cluster_member_sup["Own Cluster"] = dist_sup_cluster_center.min(axis=1)
            cluster_member_sup["Next Closest"] = dist_sup_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            cluster_member_sup["Ratio (Own/Next)"] = cluster_member_sup["Own Cluster"]/cluster_member_sup["Next Closest"]
            #convert to ordered dictionary
            quanti_var_sup_ = OrderedDict(cluster=quanti_var_sup_cluster,dist=dist_sup_cluster_center,member=cluster_member_sup)
            #convert to namedtuple
            self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

        return self
    
    def fit_predict(self,obj,y=None,sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(obj) followed by predict(X).

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : 1d array-like of shape (n_samples,), default = None
            An optional sample weights. The weights for each observation.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Index of the cluster each column belongs to.
        """
        self.fit(obj=obj,sample_weight=sample_weight)
        return self.quanti_var_.cluster
    
    def fit_transform(self,obj,y=None,sample_weight=None):
        """
        Fit the model with ``obj`` and apply the hierarchical clustering on ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.
        
        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : 1d array-like of shape (n_samples,), default = None
            An optional sample weights. The weights for each observation.
        
        Returns
        -------
        X_new : DataFrame of shape (n_columns, n_clusters)
            X transformed in the new space.
        """
        self.fit(obj=obj,sample_weight=sample_weight)
        return self.quanti_var_.dist
    
    def predict(self,X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_columns, n_components)
            New data to predict, where ``n_columns`` is the number of columns and ``n_components`` is the number of components.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Labels of the cluster each column belongs to.
        """
        # distance for new data points to cluster centers
        dist = self.transform(X)
        # assign cluster to new individuals
        cluster = dist.idxmin(axis=1).astype("category")
        cluster.name = "cluster"
        return cluster
    
    def transform(self,X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.
        
        Parameters
        ----------
        X : DataFrame of shape (n_columns, n_components)
            New data to transform, where ``n_columns`` is the number of columns and ``n_components`` is the number of components.

        Returns
        -------
        X_new : DataFrame of shape (n_columns, n_clusters)
            X transformed in the new space.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an object of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set index name as None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if convient column shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("Inconvenient column length")
        
        # number of new columns
        n_rows = X.shape[0]
        # distance for new data points to cluster centers
        dist = DataFrame(squareform(pdist(concat((X,self.cluster_.coord),axis=0),metric=self.metric))[:n_rows,n_rows:],index=X.index,columns=self.cluster_.coord.index)
        return dist