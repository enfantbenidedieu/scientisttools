# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import pdist,squareform
from sklearn.utils import check_symmetric
from sklearn.base import BaseEstimator, TransformerMixin

from .smacof import SMACOF
from .sim_dist import sim_dist

class MDS(BaseEstimator,TransformerMixin):
    """
    Metric and Non - Metric Multidimensional Scaling (MDS)
    ------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    This is a metric and non - metric multidimensional scaling

    Performs metric and non - metric Multidimensional Scaling (MDS) 
    with supplementary rows points.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.
    
    proximity :  {'euclidean','precomputed','similarity'}, default = 'euclidean'
        Dissmilarity measure to use :
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset
        
        - 'precomputed':
            Pre-computed dissimilarities are passed disrectly to ``fit`` and ``fit_transform``.
        
        - `similarity`:
            Similarity matrix is transform to dissimilarity matrix before passed to ``fit`` and ``fit_transform``.

    metric : bool, default=True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    var_sup : an integer or a list/tuple indicating the indexes of the supplementary variables

    normalized_stress : bool, default=True
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default.

    Return
    ------
    call_ : additional informations

    result_ : 
            - 'coord' for the position of the dataset in the embedding space
            - 'dist' for the Pairwise dissimilarities/distances between the points.
            - 'res_dist' for the restitue Pairwise dissimilarities/distances between the points
            - 'stress' for the The final value of the stress/normalized stress
    
    model_ : string
        The model fitted = 'mds'
    
    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
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
                ind_sup = None,
                normalized_stress=True):
        self.n_components = n_components
        self.proximity = proximity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ind_sup = ind_sup
        self.normalized_stress =normalized_stress
    
    def fit(self,X,y=None, init=None):
        """
        Fit the model to X
        ------------------
        
        Parameters
        ----------
        X : pandas/polars DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored
        
        Returns:
        --------
        self : object
                Returns the instance itself
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ############################
        # Check is supplementary columns
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple))  and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        ####################################### Save the base in a new variable
        # Store data
        Xtot = X

        ####################################### Drop supplementary individuals ########################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        #######################################################################################################################
        if self.proximity == "euclidean":
            dist = squareform(pdist(X,metric="euclidean"))
        elif self.proximity == "precomputed":
            dist = check_symmetric(X.values, raise_exception=True)
        elif self.proximity == "similarity":
            dist = sim_dist(X)
        
        #set n_compoents
        if self.n_components is None:
            n_components = 2
        else:
            n_components = self.n_components
        
        dist = pd.DataFrame(dist,index=X.index,columns=X.index)
        
        coord , stress = SMACOF(
            X=dist,
            metric=self.metric,
            n_components=n_components,
            proximity = "precomputed", 
            init=init,
            n_init=self.n_init, 
            n_jobs=self.n_jobs, 
            max_iter=self.max_iter, 
            verbose=self.verbose,
            eps=self.eps, 
            random_state=self.random_state, 
            return_n_iter=False,
            )
        
        # Set title
        if self.metric:
            title = "Metric multidimensional scaling (mMDS)"
        else:
            title = "Non-metric multidimensional scaling (NMDS)"

        res_dist = squareform(pdist(coord,metric="euclidean"))
        
        #calcul du stress 
        if self.normalized_stress:
            stress = np.sqrt(stress/np.sum(dist**2))
        
        # 
        self.call_ = {"n_components" : n_components,
                      "proximity" : self.proximity,
                      "metric" : self.metric,
                      "normalized_stress" : self.normalized_stress,
                      "X" : X,
                      "Xtot" : Xtot,
                      "title" : title}
        
        coord = pd.DataFrame(coord,index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])
       
        res_dist = pd.DataFrame(res_dist,index=X.index,columns=X.index)

        # Inertie 
        inertia = np.sum(dist**2)/(2*(dist.shape[0]**2))

        # Store informations
        self.result_ = {"coord" : coord, "dist" : dist, "res_dist" : res_dist,"stress" : stress,"inertia" : inertia}

        self.model_ = "mds"

        return self
    
    def fit_transform(self,X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------
        
        Parameters
        ----------
        X : pandas/polars DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        return self.result_["coord"]
    
    def transform(self,X,y=None):
        """
        Apply the Multidimensional Scaling reduction on X
        -------------------------------------------------

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
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        raise NotImplementedError("This method is not implemented yet")