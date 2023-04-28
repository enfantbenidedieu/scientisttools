# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
import warnings
from sklearn.utils import check_symmetric
from scientisttools.utils import sim_dist
from sklearn.base import BaseEstimator, TransformerMixin
from scientisttools.pyplot import plotMDS, plotCMDS
from sklearn import manifold

#################################################################################"
#       SMACOF ALGORITHM
###################################################################################

def SMACOF(
    X,
    metric=True,
    n_components=2,
    proximity ="euclidean", 
    init=None, 
    n_init=8, 
    n_jobs=None, 
    max_iter=300, 
    verbose=0, 
    eps=0.001, 
    random_state=None, 
    return_n_iter=False,
):
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:
    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.
    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        
    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.
    
    Returns
    -------
    coord : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.
    
    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)
    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)
    """

        
    if proximity == "euclidean":
        dissimilarities = squareform(pdist(X,metric="euclidean"))
    elif proximity == "precomputed":
        dissimilarities = check_symmetric(X.values, raise_exception=True)
    elif proximity == "similarity":
        dissimilarities = sim_dist(X)
    
    smacof_res = manifold.smacof(
            dissimilarities = dissimilarities, 
            metric = metric,
            n_components = n_components, 
            init = init,
            n_init = n_init, 
            n_jobs = n_jobs, 
            max_iter = max_iter, 
            verbose = verbose, 
            eps = eps, 
            random_state = random_state, 
            return_n_iter = return_n_iter,
            normalized_stress = "auto")
    
    if return_n_iter:
        return smacof_res[0], smacof_res[1], smacof_res[2]
    else:
        return smacof_res[0], smacof_res[1]



            


################################################################################
#       CLASSICAL MULTIDIMENSIONAL SCALING (CMDSCALE)
###############################################################################

class CMDSCALE(BaseEstimator,TransformerMixin):
    """Classic Muldimensional Scaling (CMDSCALE)
    
    """
    def __init__(self,
                n_components=None,
                labels = None,
                proximity="euclidean",
                normalized_stress=True,
                graph=True,
                figsize=None):
        self.n_components = n_components
        self.labels = labels
        self.proximity = proximity
        self.normalized_stress = normalized_stress
        self.graph = graph
        self.figsize = figsize
    
    def fit(self,X,y=None):
        """Fit the model to X
        
        Parameters
        ----------
        X : DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored
        
        Returns:
        --------
        self : object
                Returns the instance itself
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        self.nobs_ = X.shape[0]
        self.centering_matrix_ = np.identity(self.nobs_) - (1/self.nobs_)*np.ones(shape=(self.nobs_,self.nobs_))
        
        self._compute_stats(X)

        if self.graph:
            fig, axe = plt.subplots(figsize=self.figsize)
            plotCMDS(self,repel=True,ax=axe)
        
        return self
    
    def _is_euclidean(self,X):
        """
        
        """
        self.dist_ = squareform(pdist(X,metric="euclidean"))
        B = np.dot(np.dot(self.centering_matrix_,X),np.dot(self.centering_matrix_,X).T)
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _is_precomputed(self,X):
        """
        
        """
        self.dist_ = check_symmetric(X.values, raise_exception=True)
        A = -0.5*np.multiply(self.dist_,self.dist_)
        B = np.dot(self.centering_matrix_,np.dot(A,self.centering_matrix_))
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _is_similarity(self,X):
        """
        
        """
        D = sim_dist(X)
        self.dist_ = check_symmetric(D, raise_exception=True)
        A = -0.5*np.multiply(self.dist_,self.dist_)
        B = np.dot(self.centering_matrix_,np.dot(A,self.centering_matrix_))
        value, vector = np.linalg.eig(B)
        return np.real(value), np.real(vector)
    
    def _compute_stats(self,X):
        """
        
        """
        if X.shape[0] == X.shape[1] and self.proximity != "precomputed":
            raise warnings.warn(
                "The ClassicMDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``dissimilary='precomputed'``."
            )

        # Compute euclidean
        if self.proximity == "euclidean":
            eigen_value, eigen_vector = self._is_euclidean(X)
        elif self.proximity == "precomputed":
            eigen_value, eigen_vector = self._is_precomputed(X)
        elif self.proximity == "similarity" :
            eigen_value, eigen_vector = self._is_similarity(X)
        else:
            raise ValueError("You must pass a valid 'proximity'.")
        
        proportion = 100*eigen_value/np.sum(eigen_value)
        difference = np.insert(-np.diff(eigen_value),len(eigen_value)-1,np.nan)
        cumulative = np.cumsum(proportion)
        
        # Set n_components
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = (eigen_value > 1e-16).sum()
        elif not self.n_components_:
            self.n_components_ = self.n_components_
        elif self.n_components_ > self.nobs_:
            raise ValueError("Error : You must pass a valid 'n_components'.")
        
        self.eig_ = np.array([eigen_value[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])
        
        self.eigen_vector_ = eigen_vector[:,:self.n_components_]

        self.coord_ = self.eigen_vector_*np.sqrt(eigen_value[:self.n_components_])

        self.res_dist_ = squareform(pdist(self.coord_,metric="euclidean"))

        #calcul du stress 
        if self.normalized_stress:
            self.stress_ = np.sqrt(np.sum((self.res_dist_-self.dist_)**2)/np.sum(self.dist_**2))
        else:
            self.stress_ = np.sum((self.res_dist_-self.dist_)**2)

        # Inertie 
        inertia = np.sum(self.dist_**2)/(2*self.nobs_**2)

        self.inertia_ = inertia
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        # Set labels
        self.labels_ = self.labels
        if self.labels_ is None:
            self.labels_ = [f"label_" + str(i+1) for i in np.arange(0,self.nobs_)]
        
        self.model_ = "cmds"

    def transform(self,X,y=None):
        """Apply the Multidimensional Scaling reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        raise NotImplementedError("Error : This method is not implemented yet.")


    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        
        self.fit(X)

        return self.coord_
    

##############################################################################
#           METRIC & NON METRIC MULTIDIMENSIONAL SCALING (MDS)
##############################################################################

class MDS(BaseEstimator,TransformerMixin):
    """Metric and Non - Metric Multidimensional Scaling (MDS)
    
    """
    def __init__(self,
                n_components=2,
                proximity ='euclidean',
                metric=True,
                n_init=4,
                max_iter=300,
                verbose=0,
                eps=1e-3,
                n_jobs=None,
                random_state=None,
                labels = None,
                normalized_stress=True,
                graph =True,
                figsize=(10,10)):
        self.n_components = n_components
        self.proximity = proximity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.labels = labels
        self.normalized_stress =normalized_stress
        self.graph = graph
        self.figsize = figsize
    
    def fit(self,X,y=None, init=None):
        """Fit the model to X
        
        Parameters
        ----------
        X : DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored
        
        Returns:
        --------
        self : object
                Returns the instance itself
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        

        if self.proximity == "euclidean":
            self.dist_ = squareform(pdist(X,metric="euclidean"))
        elif self.proximity == "precomputed":
            self.dist_ = check_symmetric(X.values, raise_exception=True)
        elif self.proximity == "similarity":
            self.dist_ = sim_dist(X)

        #Set Labels
        self.labels_ = self.labels
        if self.labels_ is None:
            self.labels_ = ["label_"+str(i+1) for i in range(0,X.shape[0])]
        
        if self.metric:
            self.title_ = "Metric multidimensional scaling (mMDS)"
        else:
            self.title_ = "Non-metric multidimensional scaling (NMDS)"

        self.fit_transform(X,init=init)

        self.res_dist_ = squareform(pdist(self.coord_,metric="euclidean"))

        #calcul du stress 
        if self.normalized_stress:
            self.stress_ = np.sqrt(np.sum((self.res_dist_-self.dist_)**2)/np.sum(self.dist_**2))
        else:
            self.stress_ = np.sum((self.res_dist_-self.dist_)**2)

        self.model_ = "mds"

        if self.graph:
            fig, axe = plt.subplots(figsize=self.figsize)
            plotMDS(self,repel=True,ax=axe)

        return self
    
    def fit_transform(self,X, y=None, init=None):
        """
        
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        

        self.coord_, self.stress_, self.n_iter_ = SMACOF(
            X=X,
            metric=self.metric,
            n_components=self.n_components,
            proximity = self.proximity, 
            init=init,
            n_init=self.n_init, 
            n_jobs=self.n_jobs, 
            max_iter=self.max_iter, 
            verbose=self.verbose,
            eps=self.eps, 
            random_state=self.random_state, 
            return_n_iter=True,
            )
        #set n_compoents
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = self.coord_.shape[1]
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        return self.coord_
    
    def transform(self,X,y=None):
        """Apply the Multidimensional Scaling reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        raise NotImplementedError("Error : This method is not implemented yet.")
