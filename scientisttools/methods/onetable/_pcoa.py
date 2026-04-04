# -*- coding: utf-8 -*-
from numpy import ones, array,ndarray, linalg, real, c_, insert, cumsum,diff,nan,sqrt, trace
from pandas import DataFrame, Series,concat
from collections import OrderedDict, namedtuple
from scipy.spatial.distance import pdist,squareform
from sklearn.utils import check_symmetric
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.get_sup_label import get_sup_label
from ..functions.utils import check_is_dataframe

class PCoA(BaseEstimator,TransformerMixin):
    """
    Principal Coordinates Analysis (PCoA)
    
    Performs Principal Coordinates Analysis(PCoA), also knows as Classical Multidimensional Scaling (CMDSCALE) or Torgerson's scaling, with supplementary individuals.

    Parameters
    ----------
    ncp : int, default = 2
        Number of embedding dimensions.

    metric :  str or callablse, default = 'euclidean'
        Metric to use for dissimilarity computation. Default is "euclidean".

        If metric is a string, it must be one of the options allowed by
        `scipy.spatial.distance.pdist` for its metric parameter, or a metric
        listed in :func:`sklearn.metrics.pairwise.distance_metrics`

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    metric_params : dict, default = None
        Additional keyword arguments for the dissimilarity computation.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional individuals weights. The weights are given only for the active individuals.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters, with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        dist : DataFrame of shape (n_rows, n_rows)
            Pairwise dissimilarities between the points.
        D : DataFrame of shape (n_rows, n_rows)
            The square pairwise dissimilarities between the points.
        S : DataFrame of shape (n_rows, n_rows)
            The cross-product matrix, which is the square distance after double centering operation.
        d1: Series of shape (n_rows,)
            The rows weighted sums of the square pairwise dissimilarities matrix.
        d2 : Series of shape (n_rows,)
            The columns weighted sums of the square pairwise dissimilarities matrix.
        d3 : float
            The overall weighted sums of the square pairwise dissimilarities matrix.
        row_w : Series of shape (n_rows,)
            The weights of the individuals.
        ncp : int
            The number of components kepted.
        ind_sup : None, list
            The names of the supplementary individuals.

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    evd_ : evdResult
        An object containing all the eigen values decomposition, with the following attributes:

        V : 2D array-like of shape (n_samples, maxcp)
            Eigen vectors.

        d : 1D array-like of shape (maxcp,)
            Eigen values.

    ind_ : ind
        An object containing all the results for the individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, ncp)
            The coordinates of the individuals. The position of the dataset in the embedding space.
        contrib : DataFrame of shape (n_samples, ncp)
            The contributions of the individuals.
        cos2 : DataFrame of shape (n_samples, ncp)
            The square cosinus of the individuals.
        dist2 : Series of shape (n_samples,)
            The square euclidean distance of the individuals.

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_samples, ncp)
            The square cosinus of the supplementary individuals.
        dist2 : Series of shape (n_samples,)
            The square euclidean distance of the supplementary individuals.

    References
    ----------
    [1] Borg, I.; Groenen P (1997), Modern Multidimensional Scaling - Theory and Applications, Springer Series in Statistics.

    [2] Rakotomalala, R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.
    
    Examples
    --------
    >>> from scientisttools.datasets import autosmds
    >>> from scientisttools import PCoA
    >>> clf = PCoA(ncp=2,ind_sup=(12,13,14))
    >>> clf.fit(autosmds)
    PCoA(ind_sup=(12,13,14),ncp=2)
    """
    def __init__(
            self, ncp=2, metric="euclidean", metric_params=None, row_w=None, ind_sup = None, tol = 1e-7
    ):
        self.ncp = ncp
        self.metric = metric
        self.metric_params = metric_params
        self.row_w = row_w
        self.ind_sup = ind_sup
        self.tol = tol
    
    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an object of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #set index name to None
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X,indexes=self.ind_sup,axis=0)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal coordinates analysis (PCoA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set nmber of rows
        n_rows = X.shape[0]
        #set individuals weights
        if self.row_w is None:
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)):
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows:
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else:
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")

        #Compute dissimilarity matrix
        if self.metric == "precomputed":
            dist = check_symmetric(X.values, raise_exception=True) 
        else:
            dist = squareform(pdist(X=X,metric=self.metric,**(self.metric_params if self.metric_params is not None else {})))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #centering matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to DataFrame
        dist = DataFrame(dist,index=X.index,columns=X.index)
        #squared distance and centering matrices
        D = DataFrame(dist**2,index=X.index,columns=X.index)
        #double centering operation
        d1, d2, d3 = (D.T * row_w).T.sum(axis=1), (D.T * row_w).T.sum(axis=0), (D.T * (row_w**2)).T.sum(axis=0).sum()
        #cross-product matrix
        S = -0.5*(((D - d2).T - d1).T + d3)
        #trace of S
        Itot = trace(S)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #importance of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen value decomposition
        value, vector = linalg.eigh(S)
        #convert to real values
        value, vector = real(value[::-1]), real(vector[:,::-1])
        #set maximum number of components
        rank = sum((value/value[0]) > self.tol)
        #update with rank
        eigvals, eigvects = value[:rank], vector[:,:rank]

        #set number of components
        if self.ncp is None:
            ncp = rank
        elif self.ncp < 1:
            raise TypeError("ncp must be positive")
        else:
            ncp = min(self.ncp,rank)

        #convert to Ordered dictionary
        evd_ = OrderedDict(V=eigvects,d=eigvals,rank=rank,ncp=ncp)
        #convert to namedtuple
        self.evd_ = namedtuple("evdResult",evd_.keys())(*evd_.values())

        #proportion and difference
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/Itot
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = ["Dim"+str(x+1) for x in range(len(eigvals))])

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=Xtot,X=X,dist=dist,D=D,S=S,d1=d1,d2=d2,d3=d3,row_w=row_w,ncp=ncp,ind_sup=ind_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals: coordinates, contributions,  square euclidean distance and square cosinus
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the individuals
        coord = DataFrame(eigvects[:,:ncp]*sqrt(eigvals[:ncp]),index=X.index,columns=["Dim"+str(x+1) for x in range(ncp)])
        #contributions for the individuals
        ctr = (coord**2)/eigvals[:ncp]
        #square euclidean distance
        sqdist = Series(((eigvects*sqrt(eigvals))**2).sum(axis=1),index=X.index,name="Sq. Dist.")
        #cos2 for the individuals
        sqcos = ((coord**2).T/sqdist).T
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=coord,contrib=ctr,cos2=sqcos,dist2=sqdist)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            dist_sup = X_ind_sup
            if self.metric != "precomputed":
                n_rows_sup = len(ind_sup_label)
                dist_sup = DataFrame(squareform(pdist(concat((X_ind_sup,self.call_.X),axis=0),metric=self.metric,**(self.metric_params if self.metric_params is not None else {})))[:n_rows_sup,n_rows_sup:],
                                    index=ind_sup_label,columns=self.call_.X.index)
            #double centering
            D_sup = dist_sup**2
            d1_sup = D_sup.sum(axis=1)
            S_sup = (-0.5*(((D_sup.T - d1_sup).T - d2) + d3))
            #coordinates for the supplementary individuals
            ind_sup_coord = DataFrame(S_sup.values.dot(eigvects[:,:ncp]/sqrt(eigvals[:ncp])),index=ind_sup_label,columns=self.eig_.index[:ncp])
            #squared euclidean distance of the supplementary individuals
            ind_sup_sqdist = Series(((S_sup.values.dot(eigvects/sqrt(eigvals)))**2).sum(axis=1),index=ind_sup_label,name="Sq. Dist.")
            #cos2 of the supplementary individuals
            ind_sup_cos2 = ((ind_sup_coord**2).T/ind_sup_sqdist).T
            #convert to ordered dictionary
            ind_sup_ = OrderedDict(coord=ind_sup_coord,cos2=ind_sup_cos2,dist2=ind_sup_sqdist)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_columns)
            Training data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : Dataframe of shape (n_samples, ncp)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def transform(self,X):
        """
        Apply dimensionality reduction to ``X``.

        ``X`` is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            New data, where ``n_rows`` is the number of rows and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Projection of ``X`` in the first principal components, where ``n_rows`` is the number of rows and ``ncp`` is the number of the components.
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

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X contains original columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not set(self.call_.X.columns).issubset(X.columns): 
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the {} result".format(self.__class__.__name__))
        X = X[self.call_.X.columns]

        dist = X
        if self.metric != "precomputed":
            n_rows = X.shape[0]
            dist = DataFrame(squareform(pdist(concat((X,self.call_.X),axis=0),metric=self.metric,**(self.metric_params if self.metric_params is not None else {})))[:n_rows,n_rows:],
                             index=X.index,columns=self.call_.X.index)
        #double centering
        D = dist**2
        d1 = D.sum(axis=1)
        S = (-0.5*(((D.T - d1).T - self.call_.d2) + self.call_.d3))
        #coordinates for the new rows
        return DataFrame(S.values.dot(self.evd_.V[:,:self.evd_.ncp]/sqrt(self.evd_.d[:self.evd_.ncp])),index=X.index,columns=self.eig_.index[:self.evd_.ncp])