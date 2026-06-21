# -*- coding: utf-8 -*-
from numpy import array, sqrt, c_, dot, sum
from pandas import concat, get_dummies, DataFrame, Series
from collections import OrderedDict, namedtuple
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.utils import convert_series_to_dataframe, check_is_dataframe

class CatVARHCA(BaseEstimator,TransformerMixin):
    """
    Categorical Variables Hierarchical Clustering Analysis (CatVARHCA)
    
    Performs hierarchical agglomerative clustering algorithm for levels of categorical Variables. Supplementary categorical variables may be used.

    Parameters
    ----------
    n_clusters : int, default = 3
        If a (positive) integer, the tree is cut with nb_cluters clusters.
        if None, then the nb_clusters is determine using using optimal point.
    
    metric : {"dice", "bothpos"}, default = "dice"
        The similarity method.

    method : {"average", "complete", "single", "ward"} default = "ward"
        The method used to built the tree.
    
    quali_sup : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary categorical columns/variables.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.

        X : DataFrame of shape (n_samples, n_columns)
            Input data without supplementary categorical variables.

        M : DataFrame of shape (n_samples, n_levels)
            Disjunctive table.

        n_clusters : int
            The number of clusters.

        tree : tree
            The results for the hierarchical agglomerative clustering algorithm.

        quali_sup : None, list
            The names of the supplementary categorical variables.

    levels_ : levels
        An object containing the description of the clusters by the levels:

        cluster : Series of shape (n_levels,)
            The labels of each level.
        dist : DataFrame of shape (n_levels, n_clusters)
            The distance of each level to the cluster centers.
        member : DataFrame of shape (n_levels, 3)
            Cluster's members of each level (distance to own cluster, distance to next closest, ratio (own/next)).

    levels_sup_ : levels_sup, optional
        An object containing the description of the clusters by the supplementary levels:

        cluster : Series of shape (n_levels_sup,)
            The labels of each supplementary levels.
        dist : DataFrame of shape (n_levels_sup, n_clusters)
            The distance of each supplementary level to the cluster centers.
        member : DataFrame of shape (n_levels_sup, 3)
            Cluster's members of each supplementary level (distance to own cluster, distance to next closest, ratio (own/next)).

    References
    ----------
    [1] R. Rakotomalala, « Classification de variables », Tutoriels Tanagra pour le Data Mining.

    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Examples
    --------
    >>> from scientisttools.datasets import congressvotingrecords
    >>> from scientistools import CatVARHCA
    >>> vote = congressvotingrecords
    >>> clf = CatVARHCA(n_clusters=None,method="average",quali_sup=0)
    >>> clf.fit(vote)
    """
    def __init__(
            self, n_clusters = 3, metric = "dice", method = "ward", quali_sup = None
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.quali_sup = quali_sup

    def fit(self,X,y=None):
        """
        Compute hierarchical agglomerative clustering on X

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples and ``n_columns`` is the number of columns.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if metric option is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.metric in ("dice","bothpos")):
            raise ValueError("'metric' should be one of 'dice', 'bothpos'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if linkage method is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.method in ("average","complete","single","ward")):
            raise ValueError("'method' should be one of 'average', 'complete', 'single', 'ward'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary categprocal labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        quali_sup_label = get_sup_label(X=X,indexes=self.quali_sup,axis=1)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary categorical variables
        if self.quali_sup is not None:
            X_quali_sup, X = X.loc[:,quali_sup_label], X.drop(columns=quali_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # categorical variables HCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # disjunctive table
        M =  concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
        # set number of rows and number of categories
        n_rows, n_levels = M.shape
        # Compute metric Matrix
        if self.metric == "dice":
            # similarity matrix
            S = None
            # dissimilarity matrix
            D = DataFrame(sqrt(squareform(pdist(M.T,metric="sqeuclidean"))/2),index=M.columns,columns=M.columns)
        else:
            # similarity matrix
            S = DataFrame(index=M.columns,columns=M.columns).astype(float)
            for i in range(n_levels-1):
                for j in range(i+1,n_levels):
                    S.iloc[i,j] = self._distance(x=M.iloc[:,i].values,y=M.iloc[:,j].values,metric="bothpos")
                    S.iloc[j,i] = S.iloc[i,j]
            for i in range(n_levels):
                S.iloc[i,i] = sum(M.iloc[:,i].values)/n_rows
            # dissimilarity matrix
            D = 1 - S

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # agglomerative hierarchical clustering 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # linkage matrix
        Z = linkage(squareform(D,checks=False), method=self.method, metric="euclidean")
        # height
        height = (DataFrame(c_[list(range(1,Z.shape[0]+1)),Z[:,2][::-1]],columns=["cluster","height"]).
                  assign(
                      diff_1 = lambda x : -1*x["height"].diff(1),
                      diff_2 = lambda x : x["diff_1"].diff(-1)
                  ))
        height["cluster"] = height["cluster"].astype(int)

        #convert to ordered dictionary
        tree_ = OrderedDict(S=S,D=D,Z=Z,height=height,merge=Z[:,:2],N=Z[:,3])
        #convert to namedtuple
        tree = namedtuple("tree",tree_.keys())(*tree_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set numbers of clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.n_clusters is None:
            n_clusters = height[height["diff_2"]==height["diff_2"].max()]["cluster"].values[0]
        elif self.n_clusters < 0:
            raise TypeError("'n_clusters' should be a positive integer.")
        elif not isinstance(self.n_clusters,int):
            raise TypeError("'n_clusters' should be an integer")
        else:
            n_clusters = self.n_clusters

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=Xtot,X=X,M=M,n_clusters=n_clusters,tree=tree,quali_sup=quali_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Informations for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series((cut_tree(Z,n_clusters=n_clusters)+1).reshape(-1, ), index = D.index,name = "cluster",dtype="string")
        # distance for levels to the cluster centers
        uq_cluster = sorted(list(cluster.unique()))
        # distance for levels to the cluster
        dist_cluster = DataFrame(index=D.index,columns=uq_cluster).astype(float)
        for l in D.index:
            for k in uq_cluster:
                index = list(cluster[cluster==k].index)
                dist = D.loc[l,index]
                # remove l distance
                if l in dist.index:
                    dist = dist.drop(index=[l])
                # compromise distance to cluster
                if self.method in ("average","ward"):
                    d = dist.sum()/len(index)
                elif self.method == "single":
                    d = dist.min()
                elif self.method == "complete":
                    d = dist.max()
                dist_cluster.loc[l,k] = d
        # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
        cluster_member = DataFrame(index=D.index,columns=["Distance Own Cluster","Distance Next Closest"]).astype(float)
        cluster_member["Distance Own Cluster"] = dist_cluster.min(axis=1)
        cluster_member["Distance Next Closest"] = dist_cluster.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        cluster_member["Ratio (Own/Next)"] = cluster_member["Distance Own Cluster"]/cluster_member["Distance Next Closest"]
        #convert to ordered dictionary
        levels_ = OrderedDict(cluster=cluster,dist=dist_cluster,member=cluster_member)
        #convert to namedtuple
        self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #clusters for supplementary levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            # disjunctive table for supplementary variables
            M_sup = concat((get_dummies(X_quali_sup[j],dtype=int) for j in X_quali_sup.columns),axis=1)
            # compute similarity/dissimilarity matrix
            D_sup = DataFrame(index=M.columns,columns=M_sup.columns).astype(float)
            for i in M.columns:
                for j in M_sup.columns:
                    D_sup.loc[i,j] = self._distance(x=M[i].values,y=M_sup[j].values,metric=self.metric)
            if self.metric == "bothpos":
                D_sup = 1 - D_sup
            # distance for supplementary levels to the cluster centers
            dist_sup_cluster_center = DataFrame(index=M_sup.columns,columns=uq_cluster).astype(float)
            for l in M_sup.columns:
                for k in uq_cluster:
                    index = list(cluster[cluster==k].index)
                    dist_sup = D_sup.T.loc[l,index]
                    # remove l distance
                    if l in dist_sup.index:
                        dist_sup = dist_sup.drop(index=[l])
                    # find compromise distance
                    if self.method in ("ward","average"):
                        d_sup = dist_sup.sum()/len(index)
                    elif self.method == "single":
                        d_sup = dist_sup.min()
                    elif self.method == "complete":
                        d_sup = dist_sup.max()
                    dist_sup_cluster_center.loc[l,k] = d_sup
            # assign cluster to supplementary levels
            levels_sup_cluster = dist_sup_cluster_center.idxmin(axis=1)
            levels_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = DataFrame(index=D_sup.columns,columns=["Distance Own Cluster","Distance Next Closest"]).astype(float)
            cluster_member_sup["Distance Own Cluster"] = dist_sup_cluster_center.min(axis=1)
            cluster_member_sup["Distance Next Closest"] = dist_sup_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            cluster_member_sup["Ratio (Own/Next)"] = cluster_member_sup["Distance Own Cluster"]/cluster_member_sup["Distance Next Closest"]
            #convert to ordered dictionary
            levels_sup_ = OrderedDict(cluster=levels_sup_cluster,dist=dist_sup_cluster_center,member=cluster_member_sup)
            #convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

        return self
    
    @staticmethod
    def _distance(x,y,metric="dice"):
        """
        Compute distance between two dummies vectors x and y

        Parameters
        ----------
        x : 1D array-like of shape (n_samples,)
            First vector.
        y : 1D array-like of shape (n_samples,)
            Second vector.
        metric : str, default = "dice"
            Metric.
        
        Returns
        -------
        value : float
            Distance between x and y
        """
        x, y = array(x), array(y)
        value = sqrt(sum((x-y)**2)/2) if metric == "dice" else dot(x,y)/len(x)
        return value
    
    def fit_predict(self,X,y=None):
        """
        Compute cluster centers and predict cluster index for each level.

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Series of shape (n_levels,)
            Index of the cluster each level belongs to.
        """
        self.fit(X)
        return self.levels_.cluster

    def fit_transform(self,X,y=None):
        """
        Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples and ``n_columns`` is the number of columns.

        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_new : DataFrame of shape (n_levels, n_clusters)
            X transformed in the new space.
        """
        self.fit(X)
        return self.levels_.dist
    
    def predict(self,X):
        """
        Predict the closest cluster each level in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to predict, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        labels : Series of shape (n_levels,)
            Labels of the cluster each level belongs to.
        """
        # distance for new data points to cluster centers
        dist = self.transform(X)
        # assign cluster to new data points
        cluster = dist.idxmin(axis=1)
        cluster.name = "cluster"
        return cluster

    def transform(self,X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_levels, n_clusters) 
            X transformed in the new space.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # convert to pd.DataFrame if pd.Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = convert_series_to_dataframe(X)

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
        # check if convient row shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[0] != self.call_.X.shape[0]:
            raise ValueError("Inconvenient row length")

        cluster = self.levels_.cluster
        uq_cluster = sorted(list(cluster.unique()))
        # disjunctive table for supplementary variables
        M = concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
        # compute dimilarity matrix
        D = DataFrame(index=self.call_.M.columns,columns=M.columns).astype(float)
        for i in self.call_.M.columns:
            for j in M.columns:
                D.loc[i,j] = self._distance(x=self.call_.M[i].values,y=M[j].values,metric=self.metric)
        if self.metric == "bothpos":
            D = 1 - D
        # distance to cluster centers
        dist_cluster_center = DataFrame(index=M.columns,columns=uq_cluster).astype(float)
        for l in M.columns:
            for k in uq_cluster:
                index = list(cluster[cluster==k].index)
                dist = D.T.loc[l,index]
                # remove l distance
                if l in dist.index:
                    dist = dist.drop(index=[l])
                # find compromise distance
                if self.method in ("ward","average"):
                    d = dist.sum()/len(index)
                elif self.method == "single":
                    d = dist.min()
                elif self.method == "complete":
                    d = dist.max()
                dist_cluster_center.loc[l,k] = d
        return dist_cluster_center