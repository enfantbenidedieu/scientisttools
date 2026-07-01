# -*- coding: utf-8 -*-
from numpy import ndarray, array
from pandas import DataFrame, Series, concat
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict, namedtuple
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..others._catdes import catdes
from ..functions.utils import check_is_dataframe

class KMeansPC(BaseEstimator,TransformerMixin):
    """
    K-Means Clustering on Principal Components (HCPC)
    
    Performs an k-means clustering on results from a factor analysis.

    Parameters
    ----------
    ncl : int.  default = 3
        If a (positive) integer, the tree is cut with nb_cluters clusters. 
        if None, nb_clusters is set to 3.

    max_iter : int, default = 300
        The maximum number of iterations for the consolidation.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    proba : float, default = 0.05
        The probability used to select axes and variables.

    **kwargs : key words parameters
        Additionals parameters for sklearn.cluster.KMeans.

    Returns
    -------
    axes_ : desc_axes
        An object containing the description of the clusters by the principal components.
        See catdes.

    call_ : call
        An object containing the summary called parameters with the following attributes:

        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.

        X : DataFrame of shape (n_samples, n_columns)
            Input data without supplementary individuals

        data_clust : DataFrame of shape (n_samples, n_components + 1)
            The original data with a supplementary column called cluster containing the partition.

        ncl : int
            The number of clusters.

        proba : float
            The probability used to select axes and variables.

        km : class
            The results of k-means.

    cluster_ : cluster
        An object containing the results of the clusters, with the following attributes:

        coord : DataFrame of shape (n_clusters, n_components)
            The coordinates of the clusters (cluster centers).

        inertia : float
            Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.

    ind_ : dind
        An object containing the description of the clusters by the individuals, with the following attributes:

        cluster : Series of shape (n_samples,)
            The labels of individuals.
        dist : DataFrame of shape (n_samples, n_clusters)
            The distance of individuals to the cluster centers.
        member : DataFrame of shape (n_samples, 3)
            Cluster's members of individuals (distance to own cluster, distance to next closest, ratio (own/next)).

    ind_sup_ : ind_sup
        An object containing the description of the clusters by the supplementary individuals, with the following attributes:

        cluster : Series of shape (n_samples_sup,)
            The labels of supplementary individuals.
        dist : DataFrame of shape (n_samples_sup, n_clusters)
            The distance of supplementary individuals to the cluster centers.
        member : DataFrame of shape (n_samples_sup, 3)
            Cluster's members of supplementary individuals (distance to own cluster, distance to next closest, ratio).

    var_ : var
        An object containing the description of the clusters by the original data. See :class:`~scientisttools.catdes`
    
    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod.

    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Examples
    --------
    >>> from scientisttools.datasets import usarrests, tea
    >>> from scientistools import PCA, MCA, KMeansPC
    >>> # KMeansPC after PCA
    >>> clf = PCA(ncp=3)
    >>> clf.fit(usarrests)
    >>> clf2 = KMeansPC(n_clusters=4)
    >>> clf2.fit(clf)
    >>> # KMeansPC after MCA
    >>> clf = MCA(ncp=20,sup_var=range(18,36))
    >>> clf.fit(tea)
    >>> clf2 = KMeansPC(n_clusters=3)
    >>> clf2.fit(clf)
    """
    def __init__(
            self, ncl=3, max_iter=300, random_state=0, proba=0.05, **kwargs
    ):
        self.ncl = ncl
        self.max_iter = max_iter
        self.random_state = random_state
        self.proba = proba
        self.kwargs = kwargs
        
    def fit(self,obj,y=None,sample_weight=None):
        """
        Compute k-means clustering with ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

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
        # check if obj is an object of class PCA, MCA, FAMD, PCAmix, MPCA, MFA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (obj.__class__.__name__ in ("PCA","MCA","FAMD","PCAmix","MPCA","MFA")):
            raise TypeError("'obj' must be an objet of class PCA, MCA, FAMD, PCAmix, MPCA, MFA")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set proba
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.proba is None:
            proba = 0.05
        elif not isinstance(self.proba,float):
            raise TypeError(f"{type(self.proba)} is not supported")
        elif self.proba < 0 or self.proba > 1:
            raise ValueError(f"the 'proba' value {self.proba} is not within the required range of 0 and 1.")
        else:
            proba = self.proba
        
        # extract individuals coordinates
        X = obj.ind_.coord
        # copy of coordinates
        D = X.copy()

        # set number of individuals
        n_rows = X.shape[0]
        # set individuals weights
        if sample_weight is None: 
            w = obj.call_.row_w
        elif not isinstance(sample_weight,(list,tuple,ndarray,Series)): 
            raise TypeError("'sample_weight' must be a 1d array-like of sample weights.")
        elif len(sample_weight) != n_rows: 
            raise ValueError(f"'sample_weight' must be 1d array-like of shape ({n_rows},).")
        else: 
            w = Series(array(sample_weight)/sum(sample_weight),index=X.index,name="weight")

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
        cluster = Series(array(km.labels_)+1, index = D.index,name = "cluster", dtype="category")
        # convert to ordered dictionary
        call_ = OrderedDict(obj=obj,X=X,D=D,data_clust=concat((X,cluster),axis=1),ncl=ncl,w=w,proba=proba,km=km)
        # convert to namedtuple
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
        # statistics for continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # unique cluster
        uq_cluster = sorted(list(cluster.unique()))
        # distance of continuous to cluster centers
        dist_cluster_center = DataFrame(squareform(pdist(concat((X,cluster_coord),axis=0),metric="euclidean"))[:n_rows,n_rows:],index=D.index,columns=uq_cluster)
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = DataFrame(index=D.index,columns=["Own Cluster","Next Closest"]).astype(float)
        cluster_member["Own Cluster"] = dist_cluster_center.min(axis=1)
        cluster_member["Next Closest"] = dist_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        cluster_member["Ratio (Own/Next)"] = cluster_member["Own Cluster"]/cluster_member["Next Closest"]
        # convert to ordered dictionary
        ind_ = OrderedDict(cluster=cluster,dist=dist_cluster_center,member=cluster_member)
        # convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for principals components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # axis description
        axes_ = catdes(X=concat((cluster,D),axis=1),num_var="cluster",w=w,proba=proba)._asdict()
        # convert to namedtuple
        self.axes_ = namedtuple("axes",axes_.keys())(*axes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for variables
        var_ = catdes(X=concat((cluster,X),axis=1),num_var="cluster",w=w,proba=proba)._asdict()
        # convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"ind_sup_"):
            #coordinates for the supplementary individuals
            D_sup = obj.ind_sup_.coord
            n_rows_sup = D_sup.shape[0]
            # distance for supplementary individuals to cluster centers
            dist_sup_cluster_center = DataFrame(squareform(pdist(concat((D_sup,cluster_coord),axis=0),metric="euclidean"))[:n_rows_sup,n_rows_sup:],index=D_sup.index,columns=uq_cluster)
            # assign cluster to supplementary individuals
            ind_sup_cluster = dist_sup_cluster_center.idxmin(axis=1).astype("category")
            ind_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = DataFrame(index=D_sup.index,columns=["Own Cluster","Next Closest"]).astype(float)
            cluster_member_sup["Own Cluster"] = dist_sup_cluster_center.min(axis=1)
            cluster_member_sup["Next Closest"] = dist_sup_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            cluster_member_sup["Ratio (Own/Next)"] = cluster_member_sup["Own Cluster"]/cluster_member_sup["Next Closest"]
            #convert to ordered dictionary
            ind_sup_ = OrderedDict(cluster=ind_sup_cluster,dist=dist_sup_cluster_center,member=cluster_member_sup)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self
    
    def fit_predict(self,obj,y=None,sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(obj) followed by predict(X).

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

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
        return self.ind_.cluster
    
    def fit_transform(self,obj,y=None,sample_weight=None):
        """
        Compute k-means clustering with ``obj`` and transform X to cluster-distance space.

        Equivalent to fit(obj).transform(X), but more efficiently implemented.

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.
        
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
        return self.ind_.dist
    
    def predict(self,X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_components)
            New data to predict, where ``n_samples`` is the number of samples and ``n_components`` is the number of components.

        Returns
        -------
        labels : Series of shape (n_samples,)
            Labels of the cluster each sample belongs to.
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
        X : DataFrame of shape (n_samples, n_components)
            New data to transform, where ``n_samples`` is the number of samples and ``n_components`` is the number of components.

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
        # check if X is an object of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set index name as None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if convient column shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("Inconvenient column length")
        
        # number of new individuals
        n_rows = X.shape[0]
        # distance for new data points to cluster centers
        dist = DataFrame(squareform(pdist(concat((X,self.cluster_.coord),axis=0),metric="euclidean"))[:n_rows,n_rows:],index=X.index,columns=self.cluster_.coord.index)
        return dist