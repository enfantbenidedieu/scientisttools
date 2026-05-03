# -*- coding: utf-8 -*-
from numpy import array, ones, ndarray, sum, reshape, linalg, real, diff, insert, c_, cumsum, sqrt,nan
from collections import namedtuple, OrderedDict
from functools import reduce
from pandas import DataFrame, Series, concat
from scipy.spatial.distance import pdist,squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..onetable._pcoa import PCoA 
from ..functions.get_sup_label import get_sup_label
from ..functions.cov2corr import cov2corr
from ..functions.utils import check_is_all_numeric_dtype, check_is_dataframe

class DISTATIS(BaseEstimator,TransformerMixin):
    """
    Analysis of Multiple Distance Matrices (DISTATIS)

    Performs the Analysis of Multiple Distance Matrices (DISTATIS) in the sense of `Abdi, H. and al <https://personal.utdallas.edu/~herve/abdi-distatis2005.pdf>`_, which is a 3-Way Multidimensional Scaling (MDS) on the STATIS optimization procedure.
    :class:`scientisttools.DISTATIS` is a generalization of classical multidimensional scaling (PCoA) whose goal is to analyze a single distance matrix.

    Parameters
    ----------
    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : list, tuple
        The number of variables in each group.

    name_group : list, tuple, default = None
        The name of the groups. If ``None``, the group are named Gr1, Gr2 and so on.

    option : str, default = "lambda1"
        A string for the weightings of the variables.

        * 'inertia': weighting of group :math:`k` by the inverse of the total inertia of the group :math:`k`.
        * 'lambda1': weighting of group :math:`k` by the first eigen value of the group
        * 'uniform': weighting of group :math: by one.

    metric :  str or callablse, default = 'euclidean'
        Metric touse for dissimilarity computation. Default is "euclidean".

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
        An object containing all the results for the summary called parameters, with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Scod : OrderedDict
            The separate cross-product matrix.
        S : OrderedDict
            The separate normalized cross-product matrix.
        Y : DataFrame of shape (n_rows*n_rows, n_groups)
            The complete data matrix where each column is a vec for the normalized cross-product matrix.
        Z : DataFrame of shape (n_rows, n_rows)
            The compromise matrix.
        alpha : Series of shape (n_groups,)
            The weight of the each group after separate analyses.
        beta : Series of shape (n_groups,)
            The weight of each group in compromise space.
        row_w : Series of shape (n_rows,)
            The weights of the individuals.
        ncp : int
            The number of components kepted.
        group : list
            The number of columns in each group.
        name_group : list
            The name of the groups.
        ind_sup : None, list
            The names of the supplementary individuals.

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    evd_ : evdResult
        An object containing all the results for the eigen values decomposition, with the following attributes:

        V : 2D array-like of shape (n_samples, maxcp)
            Eigen vectors.

        d : 1D array-like of shape (maxcp,)
            Eigen values.

    group_ : group
        An object containing all the results for the groups, with the following attributes:

        traceRV : DataFrame of shape (n_groups, n_groups)
            The trace \emph{RV} coefficients.
        RV : DataFrame of shape (n_groups,n_groups)
            The \emph{RV} coefficients.
        eig : DataFrame of shape (n_groups, 4)
            The eigen values of the RV matrix.
        coord : DataFrame of shape (n_groups, n_groups)
            The coordinates of the groups.
        contrib : DataFrame of shape (n_groups, n_groups)
            The relative contributions of the groups.
        cos2 : DataFrame of shape (n_groups, n_groups)
            The square cosinus of the groups.
        dist2 : Series of shape (n_groups,)
            The square euclidean distance of the groups.

    ind_ : ind
        An object containing all the results for the individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, ncp)
            The coordinates of the individuals.
        contrib : DataFrame of shape (n_samples, ncp)
            The contributions of the individuals.
        cos2 : DataFrame of shape (n_samples, ncp)
            The square cosinus of the individuals.
        dist2 : Series of shape (n_samples,)
            The square euclidean distance of the individuals.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates of the individuals.

    ind_sup_ : ind_sup
        An object containing all the results for the individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_sup, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_rows_sup, ncp)
            The square cosinus of the supplementary individuals.
        dist2 : Series of shape (n_rows_sup,)
            The square euclidean distance of the supplementary individuals.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates of the supplementary individuals.

    separate_analyses_ : OrderedDict
        The results for the separate principal coordinates analysis.

    References
    ----------
    [1] Abdi, H., Valentin, D., O'Toole, A.J., & Edelman, B. (2005). DISTATIS: The analysis of multiple distance matrices. Proceedings of the IEEE Computer Society: International Conference on Computer Vision and Pattern Recognition. (San Diego, CA, USA). pp. 42-47.

    [2] Abdi, H., Valentin, D., Chollet, S., & Chrea, C. (2007). Analyzing assessors and products in sorting tasks: DISTATIS, theory and applications. Food Quality and Preference, 18, 627-640.

    [3] Abdi, H., Dunlop, J.P., & Williams, L.J. (2009). How to compute reliability estimates and display confidence and tolerance intervals for pattern classifiers using the Bootstrap and 3-way multidimensional scaling (DISTATIS). NeuroImage, 45, 89-95.

    [4] Abdi, H., Williams, L.J., Valentin, D., & Bennani-Dosse, M. (2012). STATIS and DISTATIS: Optimum multi-table principal component analysis and three way metric multidimensional scaling. Wiley Interdisciplinary Reviews: Computational Statistics, 4, 124-167. 

    [5] Abdi, H. (2007). RV coefficient and congruence coefficient. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics. Thousand Oaks (CA): Sage. pp. 849-853.

    [6] Abdi, H. (2010). Congruence: Congruence coefficient, RV coefficient, and Mantel Coefficient. In N.J. Salkind, D.M., Dougherty, & B. Frey (Eds.): Encyclopedia of Research Design. Thousand Oaks (CA): Sage. pp. 222-229.

    See Also
    --------
    :class:`scientisttools.save`
        Print results for general factor analysis model in an Excel sheet.
    :class:`scientisttools.sprintf`
        Print the analysis results.
    :class:`scientisttools.summary`
        Printing summaries of general factor analysis model.

    Examples
    --------
    >>> from scientisttools.datasets import distalgo, wine
    >>> from scientitstools import DISTATIS
    >>> from numpy import where, sqrt
    >>> # DISTATIS with precomputed distance
    >>> data = distalgo.data.apply(lambda x : where(x > 0, sqrt(x),x))
    >>> clf = DISTATIS(group=distalgo.group,name_group=distalgo.name)
    >>> clf.fit(data)
    DISTATIS(group=(6, 6, 6, 6),name_group=('Pixels','Measures','Ratings','Pairwise'))
    >>> # DISTATIS with euclidean distance
    >>> clf = DISTATIS(group=wine.group[1:-1],name_group=wine.name[1:-1],metric="euclidean")
    >>> clf.fit(wine.data.iloc[:,2:29])
    """
    def __init__(
            self, ncp=5, group=None, name_group=None, option="lambda1",metric = "precomputed", metric_params=None, row_w=None, ind_sup=None, tol = 1e-7
    ):
        self.ncp = ncp
        self.group = group
        self.name_group = name_group
        self.option = option
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
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif not isinstance(self.group, (list,tuple,ndarray,Series)):
            raise ValueError("'group' must be a 1d array-like with the number of variables in each group")
        else:
            group = [int(x) for x in self.group]

        #check if group definition
        if sum(group) != X.shape[1]:
            raise TypeError("Not convenient group definition")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if any group has only one /columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if any(x == 1 for x in group):
            raise ValueError("groups should have at least two columns")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if option is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.option in ("lambda1","inertia","uniform")):
            raise ValueError("'option' must be one of 'lambda1', 'inertia', 'uniform'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all values are numerics - all columns are continuous
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_all_numeric_dtype(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set nmber of rows
        n_rows = X.shape[0]
        if self.name_group is None:
            name_group = [f"Gr{x+1}" for x in range(len(group))]
        elif not isinstance(self.name_group,(list,tuple)):
            raise TypeError("'name_group' must be a list or a tuple with name of group")
        else:
            name_group = [x for x in self.name_group]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group name to label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_dict, k = OrderedDict(), 0
        for i, g in zip(range(len(group)),name_group):
            group_dict[g] = list(X.columns[k:(k+group[i])])
            k += group[i]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None:
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)):
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows:
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else:
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate principal coordinates analysis (PCoA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate principal coordinates analysis (PCoA)
        model = OrderedDict()
        for g, cols in group_dict.items():
            model[g] = PCoA(ncp=self.ncp,metric=self.metric,metric_params=self.metric_params,row_w=self.row_w,tol=self.tol).fit(X[cols])

        #store separate principal coordinates analysis
        self.separate_analyses_ = model

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #cross-product matrices
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Scod = OrderedDict({g : model[g].call_.S for g in name_group})
        #set groups weights
        if self.option == "lambda1":
            alpha = Series([1/model[g].eig_.iloc[0,0] for g in name_group],index=name_group,name="alpha")
        elif self.option == "inertia":
            alpha = Series([1/sum(model[g].eig_.iloc[:,0]) for g in name_group],index=name_group,name="alpha")
        else:
            alpha = Series(ones(len(name_group)),index=name_group,name="alpha")

        #normalized cross-product matrices
        S = OrderedDict({g : alpha[g]*Scod[g] for g in name_group})

        #vec of each normalized cross-matrix
        Y = concat((DataFrame(reshape(S[g],shape=(-1,1),order="F"),columns=[g]) for g in name_group),axis=1)
        #trace RV coefficients
        traceRV = Y.T.dot(Y)
        #RV coefficients
        RV = cov2corr(traceRV)

        #eigen decomposition of RV (singular value decomposition of hermittian)
        rv_evd = linalg.svd(RV,hermitian=True)
        #convert to real if any complex
        rv_evdvals, rv_evdvects = real(rv_evd[1]), real(rv_evd[0])
        rv_evdvects[:,0] = abs(rv_evdvects[:,0])
        #maximum number of components
        rv_rank = sum(rv_evdvals/rv_evdvals[0] > self.tol)
        #update with rank
        rv_eigvals, rv_eigvects = rv_evdvals[:rv_rank], rv_evdvects[:,:rv_rank]

        #RV eigen values informations
        rv_eigdiff, rv_eigprop = insert(-diff(rv_eigvals),len(rv_eigvals)-1,nan), 100*rv_eigvals/sum(rv_eigvals)
        #convert to DataFrame
        rv_eig = DataFrame(c_[rv_eigvals,rv_eigdiff,rv_eigprop,cumsum(rv_eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(rv_rank)])

        #coordinates of the group
        group_coord = DataFrame(rv_eigvects*sqrt(rv_eigvals),index=name_group,columns=rv_eig.index)
        #squared euclidean distance
        group_sqdist = (group_coord**2).sum(axis=1)
        #cos2 of the group
        group_cos2 = ((group_coord**2).T/group_sqdist).T
        #contributions of the group
        group_ctr = (group_coord**2)/rv_eigvals
        #convert to ordered dictionary
        group_ = OrderedDict(traceRV=traceRV,RV=RV,eig=rv_eig,evd=namedtuple("evdResult",["V","d"])(rv_eigvects,rv_eigvals),coord=group_coord,contrib=group_ctr,cos2=group_cos2,dist2=group_sqdist)
        #convert to namedtuple
        self.group_ = namedtuple("group",group_.keys())(*group_.values())
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compromise matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #beta
        beta = Series(rv_eigvects[:,0]/sum(rv_eigvects[:,0]),index=name_group,name="beta")
        #compromise matrix - weighted s
        Z = reduce(lambda x, y : x + y , [beta[g]*S[g] for g in name_group])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen vales decomposition (EVD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        evd = linalg.eigh(a=Z)
        #convert to real if complex
        value, vector = real(evd[0][::-1]), real(evd[1][:,::-1])
        #maximum number of components
        rank = sum(value/value[0] > self.tol)
        #update with rank
        eigvals, eigvects = value[:rank], vector[:,:rank]

        # Set number of components
        if self.ncp is None:
            ncp = rank
        elif self.ncp < 1:
            raise TypeError("ncp must be positive")
        else:
            ncp = min(self.ncp,rank)

        #convert to namedtuple
        self.evd_ = namedtuple("evdResult",["V","d","rank","ncp"])(eigvects,eigvals,rank,ncp)

        #call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Scod=Scod,S=S,Y=Y,Z=Z,alpha=alpha,beta=beta,row_w=row_w,ncp=ncp,group=group,name_group=name_group,ind_sup=ind_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for eigenvalues
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #proportion and difference
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = ["Dim"+str(x+1) for x in range(rank)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals: coordinates, partiel coordinates, contributions, squared euclidean distance and squared cosinus
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates of the individuals in compromise space
        ind_coord = DataFrame(eigvects[:,:ncp]*sqrt(eigvals[:ncp]),index=Z.index,columns=self.eig_.index[:ncp])
        #partiel coordinares of the individuals
        ind_coord_partiel = OrderedDict({g : DataFrame(S[g].values.dot(eigvects[:,:ncp]/sqrt(eigvals[:ncp])),index=Z.index,columns=self.eig_.index[:ncp]) for g in name_group})
        #convert to namedtuple
        ind_coord_partiel = namedtuple("coord_partiel",ind_coord_partiel.keys())(*ind_coord_partiel.values())
        #squared euclidean distance of the individuals
        ind_sqdist = Series(((eigvects*sqrt(eigvals))**2).sum(axis=1),index=Z.index,name="Sq. Dist.")
        #cos2 of the individuals
        ind_cos2 = ((ind_coord**2).T/ind_sqdist).T
        #contributions of the individuals
        ind_ctr = (ind_coord**2)/eigvals[:ncp]
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord,coord_partiel=ind_coord_partiel,contrib=ind_ctr,cos2=ind_cos2,dist2=ind_sqdist)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals: coordinates, partiel coordinates, square euclidean distance and square cosinus
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            Scod_sup = OrderedDict()
            for g, cols in group_dict.items():
                if self.metric == "precomputed":
                    dist_sup = X_ind_sup[cols]
                    dist_sup.columns = model[g].call_.X.index
                else:
                    n_rows_sup = len(ind_sup_label)
                    dist_sup = DataFrame(squareform(pdist(concat((X_ind_sup[cols],model[g].call_.X),axis=0),metric=self.metric,**(self.metric_params if self.metric_params is not None else {})))[:n_rows_sup,n_rows_sup:],
                                        index=ind_sup_label,columns=model[g].call_.X.index)
                #square distance matrix
                D_sup = dist_sup**2
                #double centering
                d1_sup = D_sup.sum(axis=1)
                B = (-0.5*(((D_sup.T - d1_sup).T - model[g].call_.d2) + model[g].call_.d3))
                Scod_sup[g] = B
            
            #multiply by alpha
            S_sup = OrderedDict({g : alpha[g]*Scod_sup[g] for g in name_group})
            #multiply by beta and get the sum
            Z_sup = reduce(lambda x, y : x + y, [beta[g]*S_sup[g] for g in name_group])

            #coordinates of the supplementary individuals
            ind_sup_coord = DataFrame(Z_sup.values.dot(eigvects[:,:ncp]/sqrt(eigvals[:ncp])),index=ind_sup_label,columns=self.eig_.index[:ncp])
            #partial coordinates of the supplementary individuals
            ind_sup_coord_partiel = OrderedDict({g : DataFrame(S_sup[g].values.dot(eigvects[:,:ncp]/sqrt(eigvals[:ncp])),index=ind_sup_label,columns=self.eig_.index[:ncp]) for g in name_group})
            #convert to namedtuple
            ind_sup_coord_partiel = namedtuple("coord_partiel",ind_sup_coord_partiel.keys())(*ind_sup_coord_partiel.values())
            #squared euclidean distance of the spplementary individuals
            ind_sup_sqdist = Series(((Z_sup.values.dot(eigvects/sqrt(eigvals)))**2).sum(axis=1),index=ind_sup_label,name="Sq. Dist.")
            #cos2 of the supplementary individuals
            ind_sup_cos2 = ((ind_sup_coord**2).T/ind_sup_sqdist).T
            #convert to ordered dictionary
            ind_sup_ = OrderedDict(coord=ind_sup_coord,coord_partiel=ind_sup_coord_partiel,cos2=ind_sup_cos2,dist2=ind_sup_sqdist)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` is the number of rows and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_components)
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


        model = self.separate_analyses_
        name_group = list(self.group_.coord.index)
        group_dict = OrderedDict({g : list(model[g].call_.X.columns) for g in name_group})

        Scod = OrderedDict()
        for g, cols in group_dict.items():
            if self.metric == "precomputed":
                dist =  X[cols]
                dist.columns = model[g].call_.X.index
            else:
                n_rows_sup = X.shape[0]
                dist = DataFrame(squareform(pdist(concat((X[cols],model[g].call_.X),axis=0),metric=self.metric,**(self.metric_params if self.metric_params is not None else {})))[:n_rows_sup,n_rows_sup:],
                                    index=X.index,columns=model[g].call_.X.index)
            
            #square distance matrix
            D = dist**2
            #double centering
            d1 = D.sum(axis=1)
            B = (-0.5*(((D.T - d1).T - model[g].call_.d2) + model[g].call_.d3))
            Scod[g] = B
        
        #multiply by alpha
        S = OrderedDict({g : self.call_.alpha[g]*Scod[g] for g in name_group})
        #multiply by beta and get the sum
        Z = reduce(lambda x, y : x + y, [self.call_.beta[g]*S[g] for g in name_group])

        #coordinates of the new rows
        coord = DataFrame(Z.values.dot(self.evd_.V[:,:self.evd_.ncp]/sqrt(self.evd_.d[:self.evd_.ncp])),index=X.index,columns=self.eig_.index[:self.evd_.ncp])
        return coord