# -*- coding: utf-8 -*-
from numpy import array, sqrt, c_, dot, sum, empty
from pandas import concat, get_dummies, DataFrame, crosstab, Series
from collections import OrderedDict, namedtuple
from scipy.stats.contingency import association
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.utils import check_is_dataframe

class CatVARHCA(BaseEstimator,TransformerMixin):
    """
    Categorical Variables Hierarchical Clustering Analysis (CatVARHCA)
    
    Performs hierarchical agglomerative clustering algorithm on categorical Variables. Supplementary categorical variables may be used.

    Parameters
    ----------
    n_clusters : int, default = 3
        If a (positive) integer, the tree is cut with nb_cluters clusters.
        if None, then the nb_clusters using optimal point.
    
    similarity: {"cramer", "dice", "bothpos"} (default = "cramer")
        The similarity method.
        
    metric : str, default = "euclidean"
        The metric used to built the tree. It must be one of the options allowed by `scipy.spatial.distance.pdist` for its metric parameter, or a metric listed in :func:`sklearn.metrics.pairwise.distance_metrics`.

    method : {"average", "complete", "single", "ward"} default = "ward"
        The method used to built the tree.
    
    quali_sup : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary categorical variables.

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
            The results for the hierarchical tree.

        quali_sup : None, list
            The names of the supplementary categorical variables.

    levels_ : levels
        An object containing the description of the clusters by the levels/variables.

    levels_sup_ : levels_sup, optional
        An object containing the description of the clusters by the supplementary levels/variables.

    References
    ----------
    [1] R. Rakotomalala, «Classification de variables », Tutoriels Tanagra pour le Data Mining.

    Examples
    --------
    >>> from scientisttools.datasets import congressvotingrecords
    >>> from scientistools import CatVARHCA
    >>> vote = congressvotingrecords
    >>> # cramer similarity
    >>> clf = CatVARHCA(n_clusters=2,similarity="cramer",method="ward",sup_var=0)
    >>> clf.fit(vote)
    >>> # dice dissimilarity
    >>> clf = CatVARHCA(n_clusters=None,similarity="dice",method="average",quali_sup=0)
    >>> clf.fit(vote)
    >>> # bothpos similarity
    >>> clf = CatVARHCA(n_clusters=3,similarity="bothpos",method="average",quali_sup=0)
    >>> clf.fit(vote)
    """
    def __init__(
            self, n_clusters = 3, similarity = "cramer", metric = "euclidean", method = "ward", quali_sup = None
    ):
        self.n_clusters = n_clusters
        self.similarity = similarity
        self.metric = metric
        self.method = method
        self.quali_sup = quali_sup

    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns:
        --------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if similarity option is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.similarity in ("cramer","dice","bothpos")):
            raise ValueError("'similarity' should be one of 'cramer', 'dice', 'bothpos'")

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
        #disjunctive table
        M =  concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
        #set number of rows and number of categories
        n_rows, n_cols, n_levels = X.shape[0], X.shape[1], M.shape[1]
        # Compute similarity Matrix
        if self.similarity == "cramer":
            # similarity matrix
            S = DataFrame(index=X.columns,columns=X.columns).astype(float)
            # cramer V between x and y
            for i in range(n_cols-1):
                for j in range(i+1,n_cols):
                    S.iloc[i,j] = association(crosstab(X.iloc[:,i],X.iloc[:,j]),method=self.similarity)
                    S.iloc[j,i] = S.iloc[i,j]
            # cramere V between x and x
            for i in range(n_cols):
                S.iloc[i,i] = 1
            # dissimilarity matrix
            D = 1 - S
        elif self.similarity == "dice":
            S = None
            # dissimilarity matrix
            D = DataFrame(sqrt(0.5*squareform(pdist(M.T,metric="sqeuclidean"))),index=M.columns,columns=M.columns)
        else:
            # similarity matrix
            S = DataFrame(empty(shape=(M.shape[1],M.shape[1]),dtype=float),index=M.columns,columns=M.columns)
            for i in range(n_levels-1):
                for j in range(i+1,n_levels):
                    S.iloc[i,j] = self._distance(x=M.iloc[:,i].values,y=M.iloc[:,j].values,option=self.similarity)
                    S.iloc[j,i] = S.iloc[i,j]
            for i in range(n_levels):
                S.iloc[i,i] = sum(M.iloc[:,i].values)/n_rows
            # dissimilarity matrix
            D = 1 - S

        # Linkage matrix
        link_matrix = linkage(squareform(D,checks=False), method=self.method, metric=self.metric)
        # height
        height = (DataFrame(c_[list(range(1,link_matrix.shape[0]+1)),link_matrix[:,2][::-1]],columns=["cluster","height"]).
                  assign(
                      diff_1 = lambda x : -1*x["height"].diff(1),
                      diff_2 = lambda x : x["diff_1"].diff(-1)
                  ))
        height["cluster"] = height["cluster"].astype(int)

        #convert to ordered dictionary
        tree_ = OrderedDict(S=S,D=D,linkage=link_matrix,height=height,merge=link_matrix[:,:2],n_obs=link_matrix[:,3])
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
        #assign cluster
        cluster = Series((cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, ), index = D.index,name = "cluster",dtype="string")
        # distance to cluster
        uq_cluster = sorted(list(cluster.unique()))
        dist = DataFrame(index=D.index,columns=uq_cluster).astype(float)
        for i, l in enumerate(D.index):
            for j, k in enumerate(uq_cluster):
                index = list(cluster[cluster==k].index)
                value = D.loc[l,index]
                if self.method == "average":
                    d = value.mean()
                elif self.method == "single":
                    d = value.min()
                elif self.method == "complete":
                    d = value.max()
                else:
                    d = value.mean()
                dist.iloc[i,j] = d

        # cluster's members
        D_cluster = DataFrame(index=D.index,columns=["own","next"]).astype(float)
        D_cluster["own"] = dist.min(axis=1)
        D_cluster["next"] = dist.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
        D_cluster["ratio"] = D_cluster["own"]/D_cluster["next"]
        #convert to ordered dictionary
        levels_ = OrderedDict(cluster=cluster,dist=dist,infos=D_cluster)
        #convert to namedtuple
        self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #clusters for supplementary levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            if self.similarity == "cramer":
                # V de cramer
                S_sup = DataFrame(index=X.columns,columns=X_quali_sup.columns).astype(float)
                for i in X.columns:
                    for j in X_quali_sup.columns:
                        S_sup.loc[i,j] = association(crosstab(X[i],X_quali_sup[j]),method=self.similarity) 
                D_sup = 1 - S_sup
            else:
                # disjunctive table for supplementary variables
                M_sup = concat((get_dummies(X_quali_sup[j],dtype=int) for j in X_quali_sup.columns),axis=1)
                
                # Compute dimilarity matrix
                D_sup = DataFrame(index=M.columns,columns=M_sup.columns).astype(float)
                for i in M.columns:
                    for j in M_sup.columns:
                        D_sup.loc[i,j] = self._distance(x=M[i].values,y=M_sup[j].values,option=self.similarity)
            # distance to cluster
            dist_sup = concat((D_sup,cluster),axis=1).groupby("cluster",observed=False)
            if self.method == "average":
                dist_sup = dist_sup.mean().T
            elif self.method == "single":
                dist_sup = dist_sup.min().T
            elif self.method == "complete":
                dist_sup = dist_sup.max().T
            else:
                dist_sup = dist_sup.mean().T
            # assign cluster
            levels_sup_cluster = dist_sup.idxmin(axis=1)
            levels_sup_cluster.name = "cluster"

            # cluster's members
            D_sup_cluster = DataFrame(index=D_sup.columns,columns=["own","next"]).astype(float)
            D_sup_cluster["own"] = dist_sup.min(axis=1)
            D_sup_cluster["next"] = dist_sup.apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
            D_sup_cluster["ratio"] = D_sup_cluster["own"]/D_sup_cluster["next"]
            
            #convert to ordered dictionary
            levels_sup_ = OrderedDict(cluster=levels_sup_cluster,dist=dist_sup,infos=D_sup_cluster)
            #convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

        return self
    
    @staticmethod
    def _distance(x,y,option="dice"):
        """
        Compute distance between two dummies vectors x and y

        Parameters
        ----------
        x : 1D array-like of shape (n_samples,)
            First vector.
        y : 1D array-like of shape (n_samples,)
            Second vector.
        option : str, default = "dice"
            Similarity method
        
        Returns
        -------
        value : float
            Distance between x and y
        """
        x, y = array(x), array(y)
        value = sum((x-y)**2)/2 if option == "dice" else dot(x,y)/len(x)
        return value

    def fit_transform(self,X,y=None):
        """
        Fit the model with ``X`` and apply the hierarchical clustering on ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored
        
        Returns
        -------
        clusters : Series of shape (n_columns,) or (n_levels,)
            Clusters labels.
        """
        self.fit(X)
        return self.levels_.cluster
    
    def predict(self,X):
        """
        Predict clusters for columns in X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        X_new : Series of shape (n_levels,) or (n_features,)
            predicted clusters.
        """
        # distance to cluster
        dist = self.transform(X)
        # assign cluster to new categorical variables
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
        X_new : DataFrame of shape (n_features, n_clusters) or (n_levels, n_clusters) 
            Distance to centroids.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
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

        if self.similarity == "cramer":
            # V de cramer between active variables and supplementay variables
            S = DataFrame(index=self.call_.X.columns,columns=X.columns).astype(float)
            for i in self.call_.X.columns:
                for j in X.columns:
                    S.loc[i,j] = association(crosstab(self.call_.X[i],X[j]),method="cramer") 
            D = 1 - S
        else:
            # disjunctive table for supplementary variables
            M = concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
            
            # Compute dimilarity matrix
            D = DataFrame(index=self.call_.M.columns,columns=M.columns).astype(float)
            for i in self.call_.M.columns:
                for j in M.columns:
                    D.loc[i,j] = self._distance(x=self.call_.M[i].values,y=M[j].values,option=self.similarity)

        # distance to cluster
        dist = concat((D,self.levels_.cluster),axis=1).groupby("cluster",observed=False)
        if self.method == "average":
            dist = dist.mean().T
        elif self.method == "single":
            dist = dist.min().T
        elif self.method == "complete":
            dist = dist.max().T
        else:
            dist = dist.mean().T
        return dist