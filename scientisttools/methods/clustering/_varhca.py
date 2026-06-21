# -*- coding: utf-8 -*-
from numpy import sqrt, c_
from pandas import Series, concat, DataFrame
from collections import namedtuple, OrderedDict
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.utils import convert_series_to_dataframe, check_is_dataframe

class VARHCA(BaseEstimator,TransformerMixin):
    """
    Variable Hierarchical Clustering Analysis (VARHCA)
    
    Performs hierarchical agglomerative clustering algorithm on continuous Variables. Supplementary continuous variables may be used.

    Parameters
    ----------
    n_clusters : int, default = 3
        If a (positive) integer, the tree is cut with nb_cluters clusters.

    similarity : {'pearson','kendall','spearman'} or callable, default = 'pearson'
        Similarity correlation method:

        - pearson : standard correlation coefficient
        - kendall : Kendall Tau correlation coefficient
        - spearman : Spearman rank correlation
        - callable : callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr will have 1 along the diagonals and will be symmetric regardless of the callable’s behavior.

    dissimilarity : callable, default = lambda x : sqrt(1 - x**2)
        Callable with input a float.

    metric : str, default = "euclidean"
        The metric used to build the tree. For more, see `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`.

    method : str, default = "ward"
        The method used to build the tree. For more, see `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`.

    quanti_sup : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary continuous variables.
    
    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.

        X : DataFrame of shape (n_samples, n_columns)
            Input data without supplementary continuous variables.

        n_clusters : int
            The number of clusters.

        tree : tree
            The results for the hierarchical agglomerative clustering algorithm.

        quanti_sup : None, list
            The names of the supplementary continuous variables.

    quanti_var_ : quanti_var
        An object containing the description of the clusters by the continuous variables, with following attributes:

        cluster : Series of shape (n_columns,)
            The labels of each continuous variable.
        dist : DataFrame of shape (n_columns, n_clusters)
            The distance of each continuous variable to the cluster centers.
        rsquared: DataFrame of shape (n_columns, n_clusters)
            The level of link between each continuous variable and a group of variables
        member : DataFrame of shape (n_columns, 3)
            Cluster's members of each continuous variable (distance to own cluster, distance to next closest, ratio (own/next)).

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing the description of the clusters by the supplementary continuous variables, with following attributes:

        cluster : Series of shape (n_columns_sup,)
            The labels of each supplementary continuous variable.
        dist : DataFrame of shape (n_columns_sup, n_clusters)
            The distance of each supplementary continuous variable to the cluster centers.
        rsquared: DataFrame of shape (n_columns_sup, n_clusters)
            The level of link between each supplementary continuous variable and a group of variables
        member : DataFrame of shape (n_columns_sup, 3)
            Cluster's members of each supplementary continuous variable (distance to own cluster, distance to next closest, ratio (own/next)).

    References
    ----------
    [1] R. Rakotomalala, « Classification de variables », Tutoriels Tanagra pour le Data Mining.

    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Examples
    --------
    >>> from scientisttools.datasets import jobrate
    >>> from scientistools import VARHCA
    >>> vote = congressvotingrecords
    >>> clf = VARHCA(n_clusters=4,quanti_sup=13)
    >>> clf.fit(jobrate)
    """
    def __init__(
            self, n_clusters=3, similarity="pearson", dissimilarity=lambda x : sqrt(1 - x**2), metric = "euclidean", method = "ward", quanti_sup = None
    ):
        self.n_clusters = n_clusters
        self.similarity = similarity
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.method = method
        self.quanti_sup = quanti_sup

    def fit(self,X,y=None):
        """
        Compute agglomerative hierarchical clustering
        
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
        #check if linkage method is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.method in ("average","complete","single","ward")):
            raise ValueError("'method' should be one of 'average', 'complete', 'single', 'ward'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set sup_var_label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary continuous variables labels
        quanti_sup_label = get_sup_label(X=X, indexes=self.quanti_sup, axis=1)
            
        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary continuous variables columns
        if self.quanti_sup is not None:
            X_sup, X = X.loc[:,quanti_sup_label], X.drop(columns=quanti_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables hierarchical clustering
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # compute the similarity matrix - correlation matrix
        S = X.corr(method=self.similarity)
        # compute dissimilary matrix
        D = S.apply(lambda x : self.dissimilarity(x),axis=0)
        # linkage matrix with vectorize dissimilarity matrix
        Z = linkage(squareform(D,checks=False),method=self.method,metric = self.metric)

        height = (DataFrame(c_[list(range(1,Z.shape[0]+1)),Z[:,2][::-1]],columns=["cluster","height"]).
                  assign(
                      diff_1 = lambda x : -1*x["height"].diff(1),
                      diff_2 = lambda x : x["diff_1"].diff(-1)
                  ))
        height["cluster"] = height["cluster"].astype(int)

        #convert to ordered dictionary
        tree_ = OrderedDict(S=S,D=D,Z=Z,height=height,merge=Z[:,:2],n_obs=Z[:,3])
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
        call_ = OrderedDict(Xtot=Xtot,X=X,n_clusters=n_clusters,tree=tree,quanti_sup=quanti_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Informations for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series((cut_tree(Z,n_clusters=n_clusters)+1).reshape(-1, ), index = D.index, name = "cluster",dtype="category")
        # unique cluster
        uq_cluster = sorted(list(cluster.unique()))
        # distance to cluster centers
        dist_cluster_center = DataFrame(index=D.index,columns=uq_cluster).astype(float)
        # level of link between a variable and a group of variable
        rsquared = DataFrame(index=D.index,columns=uq_cluster).astype(float)
        for i, l in enumerate(D.index):
            for j, k in enumerate(uq_cluster):
                index = list(cluster[cluster==k].index)
                dist, R2 = D.loc[l,index], S.loc[l,index]**2
                # remove l distance
                if l in dist.index:
                    dist, R2 = dist.drop(index=[l]), R2.drop(index=[l])
                # find compromise
                if self.method in ("ward","average"):
                    d, r2 = dist.sum()/len(index), R2.sum()/len(index)
                elif self.method == "single":
                    d, r2 = dist.min(), R2.min()
                elif self.method == "complete":
                    d, r2 = dist.max(), R2.max()
                dist_cluster_center.iloc[i,j], rsquared.iloc[i,j] = d, r2
        # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
        cluster_member = DataFrame(index=D.index,columns=["Distance Own Cluster","Distance Next Closest"]).astype(float)
        cluster_member["Distance Own Cluster"] = dist_cluster_center.min(axis=1)
        cluster_member["Distance Next Closest"] = dist_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        cluster_member["Ratio (Own/Next)"] = cluster_member["Distance Own Cluster"]/cluster_member["Distance Next Closest"]
        #convert to ordered dictionary
        quanti_var_ = OrderedDict(cluster=cluster,dist=dist_cluster_center,rsquared=rsquared,member=cluster_member)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters for supplementary continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            # similarity matrix between active variables and supplementary variables
            S_sup = concat((X.corrwith(other=X_sup[k],method=self.similarity,axis=0).to_frame(k) for k in quanti_sup_label),axis=1)
            # dissimilarity matrix
            D_sup = S_sup.apply(lambda x : self.dissimilarity(x),axis=0) 
            # distance to cluster centers          
            dist_sup_cluster_center = DataFrame(index=quanti_sup_label,columns=uq_cluster).astype(float)
            # level of link 
            rsquared_sup = DataFrame(index=quanti_sup_label,columns=uq_cluster).astype(float)
            for i, l in enumerate(quanti_sup_label):
                for j, k in enumerate(uq_cluster):
                    index = list(cluster[cluster==k].index)
                    dist_sup, R2_sup = D_sup.T.loc[l,index], S_sup.T.loc[l,index]**2
                    # remove l distance
                    if l in dist_sup.index:
                        d_sup, R2_sup = dist_sup.drop(index=[l]), R2_sup.drop(index=[l])
                    # find compromise
                    if self.method in ("ward","average"):
                        d_sup, r2_sup = dist_sup.sum()/len(index), R2_sup.sum()/len(index)
                    elif self.method == "single":
                        d_sup, r2_sup = dist_sup.min(), R2_sup.min()
                    elif self.method == "complete":
                        d_sup, r2_sup = dist_sup.max(), R2_sup.max()
                    dist_sup_cluster_center.iloc[i,j], rsquared_sup.iloc[i,j] = d_sup, r2_sup
            # assign cluster
            quanti_var_sup_cluster = dist_sup_cluster_center.idxmin(axis=1)
            quanti_var_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = DataFrame(index=D_sup.columns,columns=["Distance Own Cluster","Distance Next Closest"]).astype(float)
            cluster_member_sup["Distance Own Cluster"] = dist_sup_cluster_center.min(axis=1)
            cluster_member_sup["Distance Next Closest"] = dist_sup_cluster_center.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            cluster_member_sup["Ratio (Own/Next)"] = cluster_member_sup["Distance Own Cluster"]/cluster_member_sup["Distance Next Closest"]
            #convert to ordered dictionary
            quanti_var_sup_ = OrderedDict(cluster=quanti_var_sup_cluster,dist=dist_sup_cluster_center,rsquared=rsquared_sup,member=cluster_member_sup)
            #convert to namedtuple
            self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

        return self
    
    def fit_predict(self,X,y=None):
        """
        Compute cluster centers and predict cluster index for each column.

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Index of the cluster each column belongs to.
        """
        self.fit(X)
        return self.quanti_var_.cluster

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
        X_new : DataFrame of shape (n_columns, n_clusters)
            X transformed in the new space.
        """
        self.fit(X)
        return self.quanti_var_.dist
    
    def predict(self,X):
        """
        Predict the closest cluster each column in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to predict, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Labels of the cluster each column belongs to.
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
        X_new : DataFrame of shape (n_columns, n_clusters) 
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

        cluster = self.quanti_var_.cluster
        uq_cluster = sorted(list(cluster.unique()))
        # similarity matrix
        S = concat((self.call_.X.corrwith(other=X[k],method=self.similarity,axis=0).to_frame(k) for k in X.columns),axis=1)
        # dissimilarity matrix
        D = S.apply(lambda x : self.dissimilarity(x),axis=0)
        # distance to cluster centers
        dist_cluster_center = DataFrame(index=X.columns,columns=uq_cluster).astype(float)
        for i, l in enumerate(X.columns):
            for j, k in enumerate(uq_cluster):
                index = list(cluster[cluster==k].index)
                value = D.T.loc[l,index]
                # remove l distance
                if l in value.index:
                    value = value.drop(index=[l])
                if self.method in ("ward","average"):
                    dist = value.sum()/len(index)
                elif self.method == "single":
                    dist = value.min()
                elif self.method == "complete":
                    dist = value.max()
                dist_cluster_center.iloc[i,j] = dist
        return dist_cluster_center