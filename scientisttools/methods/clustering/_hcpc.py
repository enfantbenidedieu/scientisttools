# -*- coding: utf-8 -*-
from numpy import ones,average, c_,array
from pandas import DataFrame, Series, concat
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..others._auto_cut_tree import auto_cut_tree
from ..others._catdes import catdes

class HCPC(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering on Principal Components (HCPC)
    
    Performs an agglomerative hierarchical clustering on results from a factor analysis. Results include paragons, description of the clusters.

    Parameters
    ----------
    ncl : int, default =  3
        If a (positive) integer, the tree is cut with ncl clusters. If None, the tree is automatically cut.

    consol : bool, default = False
        If True, a k-means consolidation is performed after agglomerative hierarchical clustering.

    max_iter : int, default = 300
        The maximum number of iterations for the consolidation.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
    
    mincl : int, default = 3
        The least possible number of clusters suggested.

    maxcl : int, default = None
        The higher possible number of clusters suggested; by default the minimum between 10 and the number of individuals divided by 2.
        
    method : {"average","complete","single","ward"}, default = "ward"
        The method used to built the tree. The following are methods for calculating the distance between the
        newly formed cluster :math:`u` and each :math:`v`.

        * method = "single" assigns

        .. math::
            d(u,v) = \\min(dist(u[i],v[j]))

        for all points :math:`i` in cluster :math:`u` and
        :math:`j` in cluster :math:`v`. This is also known as the
        Nearest Point Algorithm.
        
        * method = "complete" assigns

        .. math::
            d(u, v) = \\max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.
        
        * method = "average" assigns

        .. math::
            d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}{(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm.
        
        * method = "ward" uses the Ward variance minimization algorithm.
        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

            d(u,v) = \\sqrt{\\frac{|v|+|s|}{T}d(v,s)^2 + \\frac{|v|+|t|}{T}d(v,t)^2 - \\frac{|v|}{T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.

    metric : str, default = "euclidean"
        The metric used to built the tree. It must be one of the options allowed by :func:`scipy.spatial.distance.pdist` for 
        its metric parameter, or a metric listed in :func:`sklearn.metrics.pairwise.distance_metrics`.

    proba : float, default = 0.05
        The probability used to select axes and variables.

    order : bool, default = True
        If True, clusters are ordered following their center coordinate on the first axis.

    **kwargs : key words parameters
        Additionals parameters for :func:`sklearn.cluster.KMeans`.

    Attributes
    ----------
    axes_ : desc_axes
        An object containing the description of the clusters by the principal components.
        See catdes.

    call_ : call
        An object containing the summary called parameters with the following attributes:

        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, 
            :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.

        X : DataFrame of shape (n_samples, n_columns)
            Input data without supplementary individuals

        data_clust : DataFrame of shape (n_samples, ncp + 1)
            The original data with a supplementary column called cluster containing the partition.

        ncl : int
            The number of clusters.

        proba : float
            The probability used to select axes and variables.

        tree : tree
            The results for the hierarchical tree.

        km : class, optional
            The results of k-means.

    cluster_ : cluster
        An object containing the results of the clusters, with the following attributes:

        coord : DataFrame of shape (ncl, ncp)
            The coordinates of the clusters (cluster centers).

    ind_ : ind
        An object containing the description of the clusters by the individuals, with the following attributes:

        cluster : Series of shape (n_samples,)
            The labels of individuals.
        dist : DataFrame of shape (n_samples, ncl)
            The distance of individuals to the cluster centers.
        member : DataFrame of shape (n_samples, 3)
            Cluster's members of individuals (distance to own cluster, distance to next closest, ratio (own/next)).

    ind_sup_ : ind_sup, optional
        An object containing the description of the clusters by the supplementary individuals, with the following attributes:

        cluster : Series of shape (n_samples_sup,)
            The labels of supplementary individuals.
        dist : DataFrame of shape (n_samples_sup, ncl)
            The distance of supplementary individuals to the cluster centers.
        member : DataFrame of shape (n_samples_sup, 3)
            Cluster's members of supplementary individuals (distance to own cluster, distance to next closest, ratio).

    var_ : var
        An object containing the description of the clusters by the original data. See :class:`~scientisttools.catdes`
    
    References
    ----------
    [1] Escofier B, Pagès J. (2008) `Analyses Factorielles Simples et Multiples <https://cdn-cms.f-static.com/uploads/1460418/normal_5b9ba5dc15394.pdf>`_. Dunod. Paris 4ed.

    [2] Lebart L., Piron M., & Morineau A. (2006). `Statistique exploratoire multidimensionnelle <https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-12/010038111.pdf>`_. Dunod. Paris 4ed.

    See Also
    --------
    KMeansPC : K-Means Clustering on Principal Components
    
    Examples
    --------
    >>> from scientisttools.datasets import usarrests, tea
    >>> from scientistools import PCA, MCA, HCPC
    >>> # HCPC after PCA
    >>> clf = PCA(ncp=3)
    >>> clf.fit(usarrests)
    PCA(ncp=3)
    >>> clf2 = HCPC(ncl=4)
    >>> clf2.fit(clf)
    HCPC(ncl=4)
    >>> # HCPC after MCA
    >>> clf = MCA(ncp=20,sup_var=range(18,36))
    >>> clf.fit(tea)
    MCA(ncp=20,sup_var=range(18,36))
    >>> clf2 = HCPC(ncl=3)
    >>> clf2.fit(clf)
    HCPC(ncl=3)
    """
    def __init__(
            self, 
            ncl = 3, 
            consol = True, 
            max_iter = 300, 
            random_state = 0, 
            mincl = 3, 
            maxcl = None, 
            method = "ward", 
            metric = "euclidean", 
            proba = 0.05, 
            order = True, 
            **kwargs
    ):
        self.ncl = ncl
        self.consol = consol
        self.max_iter = max_iter
        self.random_state = random_state
        self.mincl = mincl
        self.maxcl = maxcl
        self.method = method
        self.metric = metric
        self.proba = proba
        self.order = order
        self.kwargs = kwargs

    def fit(self,obj,y=None):
        """Compute agglomerative clustering with obj

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, 
            :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if obj is an object class PCA, MCA, FAMD, PCAmix, MCA, MFA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (obj.__class__.__name__ in ("PCA","MCA","FAMD","PCAmix","MPCA","MFA")):
            raise TypeError("'obj' must be an objet of class PCA, MCA, FAMD, PCAMIX, MPCA, MFA")
        
        # set number of individuals
        n_rows = obj.ind_.coord.shape[0]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set max cluster
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.maxcl is None:
            maxcl = min(10,round(n_rows/2))
        else:
            maxcl = min(self.maxcl,n_rows-1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set proba
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.proba is None:
            proba = 0.05
        elif not isinstance(self.proba,float):
            raise TypeError(f"{type(self.proba)} is not supported")
        elif self.proba < 0 or self.proba > 1:
            raise ValueError(f"the 'proba' value {self.proba} is not within the required range of 0 and 1.")
        else:
            proba = self.proba

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #agglomerative clustering
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coordinates for individuals
        D = obj.ind_.coord
        # linkage matrix
        Z = linkage(D,method=self.method,metric=self.metric)
        #height
        height = (DataFrame(c_[list(range(1,Z.shape[0]+1)),Z[:,2][::-1]],columns=["cluster","height"]).
                  assign(
                      diff_1 = lambda x : -1*x["height"].diff(1),
                      diff_2 = lambda x : x["diff_1"].diff(-1)
                  ))
        height["cluster"] = height["cluster"].astype(int)

        #convert to dictionary
        tree_ = {"D":D,"Z":Z,"height":height,"merge":Z[:,:2],"size":Z[:,3]}
        #convert to namedtuple
        tree = namedtuple("tree",tree_.keys())(*tree_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncl is None:
            t = auto_cut_tree(
                obj = obj,
                mincl = self.mincl,
                maxcl = maxcl,
                method = self.method,
                metric = self.metric,
                order = self.order,
                w = ones(n_rows)
            )
            ncl = t.ncl
        elif not isinstance(self.ncl,int):
            raise TypeError("ncl must be an integer")
        elif self.ncl > maxcl:
            raise ValueError(f"ncl must be less than or equal to {maxcl}")
        else:
            ncl = self.ncl

        # assign cluster to each individual
        cluster = Series((cut_tree(Z,n_clusters=ncl)+1).reshape(-1,), index = D.index, name = "cluster", dtype="category")
        # unique cluster
        uq_cluster = sorted(cluster.unique())
        # coordinates of cluster centers
        cluster_coord = DataFrame(index=uq_cluster,columns=D.columns).astype("float")
        for i in uq_cluster:
            ix = cluster[cluster==i].index
            cluster_coord.loc[i,:] = average(a=D.loc[ix,:],axis=0,weights=obj.call_.row_w.loc[ix])
        cluster_coord.index = cluster_coord.index.astype("category")
        
        # original data (continuous and/or categorical) without supplementary individuals
        X = obj.call_.Xtot
        #drop the supplementary individuals
        if hasattr(obj,"ind_sup_"):
            X = X.drop(index=obj.call_.ind_sup)

        #call informations
        call_ = {"obj":obj,"X":D,"ncl":ncl,"proba":proba,"tree":tree}

        # consolidation
        if self.consol:
            # K-means clustering
            km = KMeans(n_clusters=ncl,init=cluster_coord,max_iter=self.max_iter,random_state=self.random_state,**self.kwargs).fit(X=D,sample_weight=obj.call_.row_w)
            # assign cluster
            cluster = Series(array(km.labels_)+1, index = D.index, name = "cluster",dtype="category")
            # coordinates of the clusters - cluster centers
            cluster_coord = DataFrame(km.cluster_centers_,index=list(range(1,ncl+1)),columns=km.feature_names_in_)
            cluster_coord.index = cluster_coord.index.astype("category")
            # add to dictionary
            call_["km"] = km
        # add 
        call_["data_clust"] = concat((D,cluster),axis=1)
        
        # convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values()) 
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for clusters : coordinates, square distance to origin and square cosinus
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to ordered dictionary
        cluster_ = {"coord":cluster_coord}
        #convert to namedtuple
        self.cluster_ = namedtuple("cluster",cluster_.keys())(*cluster_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # unique cluster
        uq_cluster = sorted(cluster.unique())
        # distance for individuals to the cluster centers
        dist_cluster_center = DataFrame(squareform(pdist(concat((D,cluster_coord),axis=0),metric=self.metric))[:n_rows,n_rows:],index=D.index,columns=uq_cluster)
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = DataFrame(index=D.index,columns=["Own Cluster","Next Closest"]).astype("float")
        cluster_member["Own Cluster"] = dist_cluster_center.min(axis=1)
        cluster_member["Next Closest"] = dist_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        cluster_member["Ratio (Own/Next)"] = cluster_member["Own Cluster"]/cluster_member["Next Closest"]
        #convert to ordered dictionary
        ind_ = {"cluster":cluster,"dist":dist_cluster_center,"member":cluster_member}
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for principals components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #axis description
        axes_ = catdes(X=concat((cluster,D),axis=1),num_var="cluster",w=obj.call_.row_w,proba=proba)._asdict()
        #convert to namedtuple
        self.axes_ = namedtuple("axes",axes_.keys())(*axes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for variables
        var_ = catdes(X=concat((cluster,X),axis=1),num_var="cluster",w=obj.call_.row_w,proba=proba)._asdict()
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
            dist_sup_cluster_center = DataFrame(squareform(pdist(concat((D_sup,cluster_coord),axis=0),metric=self.metric))[:n_rows_sup,n_rows_sup:],index=D_sup.index,columns=uq_cluster)
            # assign cluster to supplementary individuals
            ind_sup_cluster = dist_sup_cluster_center.idxmin(axis=1).astype("category")
            ind_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance next closest, ratio (own/next)
            cluster_member_sup = DataFrame(index=D_sup.index,columns=["Own Cluster","Next Closest"]).astype("float")
            cluster_member_sup["Own Cluster"] = dist_sup_cluster_center.min(axis=1)
            cluster_member_sup["Next Closest"] = dist_sup_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            cluster_member_sup["Ratio (Own/Next)"] = cluster_member_sup["Own Cluster"]/cluster_member_sup["Next Closest"]
            #convert to ordered dictionary
            ind_sup_ = {"cluster":ind_sup_cluster,"dist":dist_sup_cluster_center,"member":cluster_member_sup}
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self
    
    def fit_predict(self,obj,y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(obj) followed by predict(X).

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, 
            :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Series of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(obj)
        return self.ind_.cluster
    
    def fit_transform(self,obj,y=None):
        """Compute agglomerative clustering with obj and transform X to cluster-distance space.

        Equivalent to fit(obj).transform(X), but more efficiently implemented.

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, 
            :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, ncl)
            X transformed in the new space.
        """
        self.fit(obj)
        return self.ind_.dist
    
    def predict(self,X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, ncp)
            New data to predict, where ``n_samples`` is the number of samples 
            and ``ncl`` is the number of components.

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
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, ncp)
            New data to transform, where ``n_samples`` is the number of samples 
            and ``ncp`` is the number of components.

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
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                            "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

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
        
        # number of new individuals
        n_rows = X.shape[0]
        # distance for new data points to cluster centers
        dist = DataFrame(squareform(pdist(concat((X,self.cluster_.coord),axis=0),metric=self.metric))[:n_rows,n_rows:],index=X.index,columns=self.cluster_.coord.index)
        return dist