# -*- coding: utf-8 -*-
from numpy import array, ndarray
from pandas import DataFrame, concat, Series
from collections import OrderedDict, namedtuple
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# intern function
from ..functions.utils import check_is_dataframe

class CatVARKMeansPC(BaseEstimator,TransformerMixin):
    """
    Categorical Variables K-Means clustering on Principal Components (CatVARKMeansPC)
    
    Performs k-means clustering on categorical variables using principal components.

    Parameters
    ----------
    ncl : int.  default = 3
        If a (positive) integer, the tree is cut with ncl clusters. 
        if None, ncl is set to 3.

    max_iter : int, default = 300
        The maximum number of iterations for the consolidation.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    **kwargs : key words parameters
        Additionals parameters for sklearn.cluster.KMeans.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        obj : class
            An object of class :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        X : DataFrame of shape (n_samples, n_components)
            Coordinates of levels.

        data_clust : DataFrame of shape (n_samples, n_components +1) 
            Coordinates of levels with cluster column.

        ncl : int
            The number of clusters.

        km : class
            The fitted K-Means object.

    cluster_ : cluster
        An object containing the results of the clusters, with the following attributes:

        coord : DataFrame of shape (n_clusters, n_components)
            The coordinates of the clusters - cluster centers.

        inertia : float
            Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    
    levels_ : levels
        An object containing the description of the clusters by the levels.

        cluster : Series of shape (n_samples,)
            The labels of each data point.
        dist : DataFrame of shape (n_samples, n_clusters)
            The distance of each data points to the cluster centers.
        member : DataFrame of shape (n_samples, 3)
            Cluster's members of each data point (distance to own cluster, distance to next closest, ratio).

    levels_sup_ : levels_sup, optional
        An object containing the description of the clusters by the supplementary levels.

        cluster : Series of shape (n_samples_sup,)
            The labels of each supplementary data point.
        dist : DataFrame of shape (n_samples_sup, n_clusters)
            The distance of each supplementary data points to the cluster centers.
        member : DataFrame of shape (n_samples_sup, 3)
            Cluster's members of each data point (distance to own cluster, distance to next closest, ratio).

    References
    ----------
    [1] R. Rakotomalala, « Classification de variables », Tutoriels Tanagra pour le Data Mining.

    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Examples
    --------
    >>> from scientisttools.datasets import loisirs
    >>> from scientistools import PCA, CatVARKMeansPC
    >>> # KMeans after MCA
    >>> clf = MCA(ncp=2)
    >>> clf.fit(loisirs)
    >>> clf2 = CatVARKMeansPC(ncl=3)
    >>> clf2.fit(clf)
    """
    def __init__(
            self, ncl=3, max_iter=300, random_state=0, **kwargs
    ):
        self.ncl = ncl
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        
    def fit(self,obj,y=None,sample_weight=None):
        """
        Compute k-means clustering with ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

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
        #check if obj is an object of class MCA, FAMD, PCAmix, MPCA, MFA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (obj.__class__.__name__ in ("MCA","FAMD","PCAmix","MPCA","MFA")):
            raise TypeError("'obj' must be an objet of class MCA, FAMD, PCAmix, MPCA, MFA")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # K-means clustering
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # extract principal components
        X = obj.levels_.coord
        # copy of principal components
        D = X.copy()

        # number of levels
        n_levels = X.shape[0]
        # weighted of levels extract from factor analysis
        if sample_weight is None: 
            w = obj.call_.col_w.loc[X.index]
        elif not isinstance(sample_weight,(list,tuple,ndarray,Series)): 
            raise TypeError("'sample_weight' must be a 1d array-like of sample weights.")
        elif len(sample_weight) != n_levels: 
            raise ValueError(f"'sample_weight' must be a 1d array-like of shape ({n_levels},).")
        else: 
            w = Series(array(sample_weight),index=X.index,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set numbers of clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncl is None:
            ncl = 3
        elif self.ncl < 0:
            raise TypeError("'ncl' should be a positive integer.")
        elif not isinstance(self.ncl,int):
            raise TypeError("'ncl' should be an integer")
        else:
            ncl = self.ncl
        
        # K-Means clustering
        km = KMeans(n_clusters=ncl,max_iter=self.max_iter,random_state=self.random_state, **self.kwargs).fit(X=D,sample_weight=w)
        # assign cluster
        cluster = Series(array(km.labels_)+1, index = D.index,name = "cluster",dtype="category")

        #convert to ordered dictionary
        call_ = OrderedDict(obj=obj,X=D,D=D,data_clust=concat((D,cluster),axis=1),w=w,ncl=ncl,km=km)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coordinates of the clusters - cluster centers
        cluster_coord = DataFrame(km.cluster_centers_,index=list(range(1,ncl+1)),columns=km.feature_names_in_)
        cluster_coord.index = cluster_coord.index.astype("category")
        # convert to ordered dictionary
        cluster_ = OrderedDict(coord=cluster_coord,inertia=km.inertia_)
        # convert to namedtuple
        self.cluster_ = namedtuple("cluster",cluster_.keys())(*cluster_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # unique cluster
        uq_cluster = sorted(list(cluster.unique()))
        # distance for levels to the cluster centers
        dist_cluster_center = DataFrame(squareform(pdist(concat((D,cluster_coord),axis=0),metric="euclidean"))[:n_levels,n_levels:],index=D.index,columns=uq_cluster)
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = DataFrame(index=D.index,columns=["Own Cluster","Next Closest"]).astype(float)
        cluster_member["Own Cluster"] = dist_cluster_center.min(axis=1)
        cluster_member["Next Closest"] = dist_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        cluster_member["Ratio (Own/Next)"] = cluster_member["Own Cluster"]/cluster_member["Next Closest"]
        # convert to ordered dictionary
        levels_ = OrderedDict(cluster=cluster,dist=dist_cluster_center,member=cluster_member)
        # convert to namedtuple
        self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for supplementary levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"levels_sup_"):
            # coordinates for supplementary levels
            X_sup = obj.levels_sup_.coord
            # number of supplementary levels
            n_levels_sup = X_sup.shape[0]
            # distance for supplementary levels to the cluster centers
            dist_sup_cluster_center = DataFrame(squareform(pdist(concat((X_sup,cluster_coord),axis=0),metric="euclidean"))[:n_levels_sup,n_levels_sup:],index=X_sup.index,columns=uq_cluster)
            # assign cluster to supplementary levels
            levels_sup_cluster = dist_sup_cluster_center.idxmin(axis=1).astype("category")
            levels_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance next closest, ratio (own/next)
            cluster_member_sup = DataFrame(index=X_sup.index,columns=["Own Cluster","Next Closest"]).astype(float)
            cluster_member_sup["Own Cluster"] = dist_sup_cluster_center.min(axis=1)
            cluster_member_sup["Next Closest"] = dist_sup_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            cluster_member_sup["Ratio (Own/Next)"] = cluster_member_sup["Own Cluster"]/cluster_member_sup["Next Closest"]
            #convert to ordered dictionary
            levels_sup_ = OrderedDict(cluster=levels_sup_cluster,dist=dist_sup_cluster_center,member=cluster_member_sup)
            #convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

        return self
    
    def fit_predict(self,obj,y=None,sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(obj) followed by predict(X).

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : 1d array-like of shape (n_samples,), default = None
            An optional sample weights. The weights for each observation.

        Returns
        -------
        labels : Series of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(obj=obj,sample_weight=sample_weight)
        return self.levels_.cluster
    
    def fit_transform(self,obj,y=None,sample_weight=None):
        """
        Compute k-means clustering with ``obj`` and transform X to cluster-distance space.

        Equivalent to fit(obj).transform(X), but more efficiently implemented.

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.
        
        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : 1d array-like of shape (n_samples,), default = None
            An optional sample weights. The weights for each observation.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        self.fit(obj=obj,sample_weight=sample_weight)
        return self.levels_.dist
    
    def predict(self,X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to predict, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        labels : Series of shape (n_samples,)
            Labels of the cluster each sample belongs to.
        """
        # distance of new data points to the cluster centers
        dist = self.transform(X)
        # assign cluster to new data points
        cluster = dist.idxmin(axis=1).astype("category")
        cluster.name = "cluster"
        return cluster
    
    def transform(self,X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to transform, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_clusters)
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
        
        # number of new data points
        n_rows = X.shape[0]
        # distance of new data points to the cluster centers
        dist = DataFrame(squareform(pdist(concat((X,self.cluster_.coord),axis=0),metric="euclidean"))[:n_rows,n_rows:],index=X.index,columns=self.cluster_.coord.index)
        return dist