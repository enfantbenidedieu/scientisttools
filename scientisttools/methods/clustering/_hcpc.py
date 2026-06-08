# -*- coding: utf-8 -*-
from numpy import ones,sqrt, c_, outer,divide,add,triu_indices,cumsum,argmax
from pandas import DataFrame, Series, concat,get_dummies
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from ..functions.statistics import func_groupby
from ..others._catdes import catdes

class HCPC(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering on Principal Components (HCPC)
    
    Performs an agglomerative hierarchical clustering on results from a factor analysis. Results include paragons, description of the clusters.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

    n_clusters : int.  
        If a (positive) integer, the tree is cut with nb.cluters clusters. if None, the tree is automatically cut
    
    min_cluster : int. 
        The least possible number of clusters suggested.

    max_cluster : int. 
        The higher possible number of clusters suggested; by default the minimum between 10 and the number of individuals divided by 2.

    metric : str, default = "euclidean"
        The metric used to built the tree. It must be one of the options allowed by `scipy.spatial.distance.pdist` for its metric parameter, or a metric listed in :func:`sklearn.metrics.pairwise.distance_metrics`.

    method : str, default = "ward"
        the method used to built the tree. 

    proba : float, default = 0.05
        The probability used to select axes and variables.

    n_paragons : int. 
        The number of edited paragons.

    order : bool, default = True
        If True, clusters are ordered following their center coordinate on the first axis.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.

        X : DataFrame of shape (n_samples, n_columns)
            Input data without supplementary individuals

        data_clust : DataFrame of shape (n_samples, n_columns + 1)
            The original data with a supplementary column called cluster containing the partition.

        n_clusters : int
            The number of clusters.

        proba : float
            The probability used to select axes and variables.

        tree : tree
            The results for the hierarchical tree.

    cluster_ : cluster
        An object containing the results of the clusters, with the following attributes:

        coord : DataFrame of shape (n_clusters, ncp)
            The coordinates of the clusters.
        cos2 : DataFrame of shape (n_clusters, ncp)
            The squared cosinus of the clusters.
        dist2 : Series of shape (n_clusters,)
            The squared distance to origin of the clusters.
        vtest : DataFrame of shape (n_clusters, ncp)
            The value-test (which is a criterion with a Normal distribution) of the clusters.

    desc_axes_ : desc_axes
        An object containing the description of the clusters by the principal components.

    desc_ind_ : desc_ind
        An object containing the description of the clusters by the individuals.

    desc_ind_sup_ : desc_ind_sup
        An object containing the description of the clusters by the supplementary individuals.

    desc_var_ : desc_var
        An object containing the description of the clusters by the original data.
    
    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod.

    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon, tea
    >>> from scientistools import PCA, MCA, HCPC
    >>> # HCPC after PCA
    >>> clf = PCA(ncp=5,ind_sup=range(41,46),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> clf2 = HCPC(n_clusters=3)
    >>> clf2.fit(clf)
    >>> # HCPC after MCA
    >>> clf = MCA(ncp=20,sup_var=range(18,36))
    >>> clf.fit(tea)
    >>> clf2 = HCPC(n_clusters=3)
    >>> clf2.fit(clf)
    """
    def __init__(
            self, n_clusters=3, consol=True, max_iter=300, min_cluster=3, max_cluster=None, metric="euclidean", method="ward", proba=0.05, order=True, ind_sup = False
    ):
        self.n_clusters = n_clusters
        self.consol = consol
        self.max_iter = max_iter
        self.min_cluster = min_cluster
        self.max_cluster = max_cluster
        self.metric = metric
        self.method = method
        self.proba = proba
        self.order = order
        self.ind_sup = ind_sup

    @staticmethod
    def _auto_cut_tree(obj,min_clust,max_clust,metric,method,order,w=None):
        """
        Automatic tree cut
        
        Automatic tree cut to determine optimal number of clusters.

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        min_clust : int
            The least possible number of clusters suggested.

        max_clust : int
            The higher possible number of clusters suggested.

        metric : str
            The metric used to build the tree. For more, see `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`.

        method : str
            The method used to build the tree. For more, see `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`.

        order : bool
            If ``True``, clusters are ordered following their center coordinate on the first axis.

        weights : 1d array-like of shape (n_samples,)
            Weights for each observation, with same length as zero axis of data.

        Returns
        -------
        nb_clust : int
            The number of clusters.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if obj is an object of class PCA, MCA, FAMD, PCAmix, MPCA, MFA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (obj.__class__.__name__ in ("PCA","MCA","FAMD","PCAmix","MPCA","MFA")):
            raise TypeError("'obj' must be an object of class PCA, MCA, FAMD, PCAmix, MPCA, MFA")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #order dataset
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if order:
            data = concat((obj.ind_.coord,obj.call_.X,obj.call_.row_w),axis=1)
            if w is not None:
                w = w[::-1]
            data = data.sort_values(by=data.columns.tolist()[0],ascending=True)
            obj.ind_ = obj.ind_._replace(coord = data.iloc[:,:obj.ind_.coord.shape[1]])
            obj.call_ = obj.call_._replace(X=data.iloc[:,(obj.ind_.coord.shape[1]+1):(data.shape[1]-1)],row_w=data.iloc[:,-1])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #automatic tree cut
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract individuals coordinates
        X = obj.ind_.coord
        #dissimilarity matrix
        do = pdist(X,metric=metric)**2
        #set weights
        if w is None:
            w = ones(X.shape[0])

        #effectifs
        eff = divide((outer(w,w)/sum(w)),add.outer(w,w))
        dissi = do*eff[triu_indices(eff.shape[0], k = 1)]
        #agglometrive clustering
        link_matrix = linkage(dissi,metric=metric,method=method)
        inertia_gain = link_matrix[:,2][::-1]
        intra = cumsum(inertia_gain[::-1])[::-1]
        quot = inertia_gain[(min_clust-1):max_clust]/inertia_gain[(min_clust):(max_clust+1)]
        nb_clust = (argmax(quot)+1) + min_clust - 1

        #convert to ordered dictionary
        res_ = OrderedDict(obj=obj,tree=link_matrix,nb_clust=nb_clust,within=intra,inertia_gain=inertia_gain,quot=quot)
        #convert to namedtuple
        return namedtuple("auto_cut_tree",res_.keys())(*res_.values())

    def fit(self,obj,y=None):
        """
        Fit the model with ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

        y : None
            Y is ignored.
        
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
        if self.max_cluster is None:
            max_cluster = min(10,round(n_rows/2))
        else:
            max_cluster = min(self.max_cluster,n_rows-1)

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
        # Linkage matrix
        link_matrix = linkage(D,method=self.method,metric=self.metric)
        #height
        height = (DataFrame(c_[list(range(1,link_matrix.shape[0]+1)),link_matrix[:,2][::-1]],columns=["cluster","height"]).
                  assign(
                      diff_1 = lambda x : -1*x["height"].diff(1),
                      diff_2 = lambda x : x["diff_1"].diff(-1)
                  ))
        height["cluster"] = height["cluster"].astype(int)

        #convert to ordered dictionary
        tree_ = OrderedDict(D=D,linkage=link_matrix,height=height,merge=link_matrix[:,:2],n_obs=link_matrix[:,3])
        #convert to namedtuple
        tree = namedtuple("tree",tree_.keys())(*tree_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.n_clusters is None:
            t = self._auto_cut_tree(
                obj = obj,
                min_clust = self.min_cluster,
                max_clust = max_cluster,
                method = self.method,
                metric = self.metric,
                order = self.order,
                w = ones(n_rows)
            )
            n_clusters = t.nb_clust
        elif not isinstance(self.n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")
        elif self.n_clusters > max_cluster:
            raise ValueError(f"n_clusters must be less than or equal to {max_cluster}")
        else:
            n_clusters = self.n_clusters

        #assign cluster to each individual
        cluster = Series((cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, ), index = D.index,name = "cluster",dtype="string")

        # original data (continuous and/or categorical) without supplementary individuals
        X = obj.call_.Xtot
        #drop the supplementary individuals
        if hasattr(obj,"ind_sup_"):
            X = X.drop(index=obj.call_.ind_sup)

        #call informations
        call_ = OrderedDict(obj=obj,Xtot=obj.call_.Xtot,X=X,data_clust=concat((X,cluster),axis=1),n_clusters=n_clusters,proba=proba,tree=tree)
        #set as model attributes
        self.call_ = namedtuple("call",call_.keys())(*call_.values()) 
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for clusters : coordinates, square distance to origin and square cosinus
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the clusters - barycenter of individuals
        cluster_coord = func_groupby(X=D,by=cluster,func="mean",w=obj.call_.row_w)
        #center and scale
        center, scale =  D.mean(axis=0), D.std(axis=0,ddof=0)
        #proportion of cluster
        p_k = (get_dummies(cluster,dtype=int).T * obj.call_.row_w).sum(axis=1)
        #value-test for the clusters
        cluster_vtest = (((cluster_coord - center)/scale).T * sqrt((n_rows-1)/((1/p_k) - 1))).T
        #conditional weighted average of standardized data
        Z_cluster = func_groupby(X=obj.call_.Z,by=cluster,func="mean",w=obj.call_.row_w)
        #dist2 of the clusters
        cluster_sqdisto = ((Z_cluster**2)*obj.call_.col_w).sum(axis=1)
        cluster_sqdisto.name = "sq. Dist."
        #cos2 of the clusters
        cluster_sqcos = ((cluster_coord**2).T/cluster_sqdisto).T
        #convert to ordered dictionary
        cluster_ = OrderedDict(coord=cluster_coord,cos2=cluster_sqcos,dist2=cluster_sqdisto,vtest=cluster_vtest)
        #convert to namedtuple
        self.cluster_ = namedtuple("cluster",cluster_.keys())(*cluster_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # distance to cluster
        uq_cluster = sorted(list(cluster.unique()))
        # distance to cluster
        disto_cluster = DataFrame(squareform(pdist(concat((D,cluster_coord),axis=0),metric=self.metric))[:n_rows,n_rows:],index=D.index,columns=uq_cluster)
        # cluster's members
        D_cluster = DataFrame(index=D.index,columns=["own","next"]).astype(float)
        D_cluster["own"] = disto_cluster.min(axis=1)
        D_cluster["next"] = disto_cluster.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        D_cluster["ratio"] = D_cluster["own"]/D_cluster["next"]

        #convert to ordered dictionary
        desc_ind_ = OrderedDict(cluster=cluster,dist=disto_cluster,infos=D_cluster)
        #convert to namedtuple
        self.desc_ind_ = namedtuple("desc_ind",desc_ind_.keys())(*desc_ind_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for principals components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #axis description
        desc_axes_ = catdes(X=concat((cluster,D),axis=1),num_var="cluster",w=obj.call_.row_w,proba=proba)._asdict()
        #convert to namedtuple
        self.desc_axes_ = namedtuple("desc_axes",desc_axes_.keys())(*desc_axes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for variables
        desc_var_ = catdes(X=concat((cluster,X),axis=1),num_var="cluster",w=obj.call_.row_w,proba=proba)._asdict()
        # convert to namedtuple
        self.desc_var_ = namedtuple("desc_var",desc_var_.keys())(*desc_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None and hasattr(obj,"ind_sup_"):
            #coordinates for the supplementary individuals
            D_sup = obj.ind_sup_.coord
            n_rows_sup = D_sup.shape[0]
            # distance for supplementary individuals to cluster
            disto_sup_cluster = DataFrame(squareform(pdist(concat((D_sup,cluster_coord),axis=0),metric=self.metric))[:n_rows_sup,n_rows_sup:],index=D_sup.index,columns=uq_cluster)
            # assign cluster to supplementary individuals
            ind_sup_cluster = disto_sup_cluster.idxmin(axis=1)
            ind_sup_cluster.name = "cluster"
            # cluster's members
            D_sup_cluster = DataFrame(index=D_sup.index,columns=["own","next"]).astype(float)
            D_sup_cluster["own"] = disto_cluster.min(axis=1)
            D_sup_cluster["next"] = disto_cluster.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            D_sup_cluster["ratio"] = D_sup_cluster["own"]/D_sup_cluster["next"]

            #convert to ordered dictionary
            desc_ind_sup_ = OrderedDict(cluster=ind_sup_cluster,dist=disto_sup_cluster,infos=D_sup_cluster)
            #convert to namedtuple
            self.desc_ind_sup_ = namedtuple("desc_ind_sup",desc_ind_sup_.keys())(*desc_ind_sup_.values())

        return self
    
    def fit_transform(self,obj,y=None):
        """
        Fit the model with ``obj`` and apply the hierarchical clustering on ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.
        
        y : None
            y is ignored.
        
        Returns
        -------
        clusters : Series of shape (n_rows,)
            Clusters labels.
        """
        self.fit(obj)
        return self.ind_.cluster
    
    def predict(self,X):
        """
        Predict clusters for samples in X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        X_new : Series of shape (n_samples,)
            predicted clusters.
        """
        # distance to cluster
        dist = self.transform(X)
        # assign cluster to new individuals
        cluster = dist.idxmin(axis=1)
        cluster.name = "cluster"
        return cluster
    
    def transform(self,X):
        """
        Distance to centroids

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_clusters)
            Distance to centroids.
        """
        #number of new individuals
        n_rows = X.shape[0]
        #coordinates for the new individuals
        D = self.call_.obj.transform(X)
        # distance for new individuals to centroids
        dist = DataFrame(squareform(pdist(concat((D,self.cluster_.coord),axis=0),metric=self.metric))[:n_rows,n_rows:],index=D.index,columns=self.cluster_.coord.index)
        return dist