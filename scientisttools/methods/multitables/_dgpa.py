# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, linalg, zeros, diff, insert, cumsum, c_,nan, diag, sum,sqrt
from pandas import DataFrame, Series, concat, CategoricalDtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd
from ..functions.utils import check_is_bool, is_all_numeric_dtype, is_all_object_or_category_dtype, check_is_dataframe
from ..others._disjunctive import disjunctive
from ..others._splitgroup import splitgroup, RVstats

class DGPA(BaseEstimator,TransformerMixin):
    """
    Dual Generalized Procustes Analysis (DGPA)

    Performns Dual Generalized Procustes Analysis (DGPA) with supplementary individuals.

    Parameters
    ----------
    scale_unit : bool, default = True
        If ``True``, then the data are scaled to unit variance.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : int, str
        The indexe or name of the categorical variable which allows to make the group of individuals.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.
    
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.

    Returns
    -------
    call_ : call
        An object with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        x : DataFrame of shape (n_rows, n_columns - 1)
            The Data
        y : Series of shape (n_rows,)
            The vector of factors associated with group structure
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        dummies : DataFrame of shape (n_rows, n_levels)
            Disjunctive table.
        M : DataFrame of shape (n_groups, n_levels)
            The 1-proportion of levels associated to each group.
        Zcod : DataFrame of shape (n_rows, n_columns)
            The concatenated standardized data
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        C : DataFrame of shape (n_columns, n_columns)
            The compromise loadings.
        center : DataFrame of shape (n_groups, n_columns)
            The concatenated variables weighted average.
        scale : DataFrame of shape (n_groups, n_columns)
            The concatenate variables standard deviation.
        z_center : Series of shape (n_columns,)
            The weighted average of concatenate standardized data.
        z_scale : Series of shape (n_columns,)
            The weighted standard deviation of concatenate standardized data.
        ncp : int, default = 5
            The number of dimensions kept in the results.
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        group : list
            The name of the group variables used to make the group of individuals.
        ind_sup : None, list
            The names of the supplementary individuals.

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    group_ : group
        An object containing all the results for the groups, with the following attributes:

        eig : DataFrame of shape (maxcp_rv, 4)
            The eigenvalue of RV matrix, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

        coord : DataFrame of shape (n_groups, n_groups)
            The coordinates of the groups.

        traceRV : DataFrame of shape (n_groups, n_groups)
            The trace RV between groups.

        RV : DataFrame of shape (n_groups, n_groups)
            The RV coefficient between groups.

        infos : DataFrame of shape (n_groups, 3)
            Additionals informations (weight, inertia and percentage of inertia) of the groups.

        lambd : DataFrame of shape (n_groups, ncp)
            The specific variances of groups.

        expl_var : DataFrame of shape (n_groups, ncp)
            Percentages of total variance recovered associated with each dimension.

    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows, ncp)
            The coordinates of the individuals.

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.

    quanti_var_ : quanti_var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
    
    separate_analyses_ : dict
        The results for the separates Principal Component Analysis.

    svd_ : svdResult
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

    References
    ----------
    [1] J. Gower (1975). Generalized procrustes analysis. \emph{Psychometrika}, 40(1), 3-51.

    [2] A. Eslami, E. M. Qannari, A. Kohler and S. Bougeard (2013). General overview of methods of analysis of multi-group datasets, \emph{Revue des Nouvelles Technologies de l'Information}, 25, 108-123.

    [3] A. Eslami, E. M. Qannari, A. Kohler and S. Bougeard (2013). Analyses factorielles de donnees structurees en groupes d'individus, \emph{Journal de la Societe Francaise de Statistique}, 154(3), 44-57.
    
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
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DGPA
    >>> clf = DGPA(group=4)
    >>> clf.fit(iris)
    DGPA(group=4)
    """
    def __init__(
            self, scale_unit = True, ncp = 5,  group = None, row_w = None, col_w = None, ind_sup = None, tol = 1e-7
    ):  
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.group = group
        self.row_w = row_w
        self.col_w = col_w
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
        #check if scale_unit is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.scale_unit)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.group,(int,str)):
            raise TypeError("'group' must be either an objet of type int or str")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_label, ind_sup_label = get_sup_label(X=X, indexes=self.group, axis=1), get_sup_label(X=X,indexes=self.ind_sup,axis=0)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #dual multiple factor analysis (DMFA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into x and y
        y, x = X[group_label[0]], X.drop(columns=group_label)

        #check if all columns are either continuous or categorical.
        if not (is_all_numeric_dtype(x) or is_all_object_or_category_dtype(x)):
            raise TypeError("Not applied to mixed data") 

        #unique element in y
        uq_classe = sorted(list(y.unique()))
        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

        #number of rows and number of columns
        n_rows, n_vars = x.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None:
            row_w = Series(ones(n_rows)/n_rows,index=x.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)):
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows:
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else:
            row_w = Series(array(self.row_w)/sum(self.row_w),index=x.index,name="weight")

        #set variables weights
        if self.col_w is None:
            var_w = Series(ones(n_vars),index=x.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)):
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_vars:
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_vars},).")
        else:
            var_w = Series(array(self.col_w),index=x.columns,name="weight")

        #group index
        group_dict = OrderedDict({k : list(y[y==k].index) for k in uq_classe})
     
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set variables xcod - reorder 
        Xcod, col_w, dummies, M = x.copy(), var_w.copy(), None, None
        if is_all_object_or_category_dtype(x):
            dummies = disjunctive(x)
            M = concat(((1 - ((dummies.loc[rows,:].T * row_w[rows]/sum(row_w[rows])).sum(axis=1))).to_frame(g) for g, rows in group_dict.items()),axis=1).T    
            Xcod = dummies*M.loc[y.values,:].values
            col_w = Series([x*y for x,y in zip(ones(dummies.shape[1]),list(chain(*[repeat(i,k) for i, k in zip(var_w,[x[j].nunique() for j in x.columns])])))],index=dummies.columns,name="weight")
        #set number of columns
        n_cols = Xcod.shape[1]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #run separate general factor analysis
        model = splitgroup(X=Xcod,y=y,scale_unit=self.scale_unit,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
            
        #store separate analysis
        self.separate_analyses_ = model

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #scale_unitd data
        Zcod = concat((model[g].call_.Z for g in list(model.keys())),axis=0,ignore_index=False).loc[y.index,:]
        #weighted average
        center, scale = concat((model[g].call_.center.to_frame(g) for g in list(model.keys())),axis=1).T, concat((model[g].call_.scale.to_frame(g) for g in list(model.keys())),axis=1).T
        #number of rows in each groups
        nb_rows = Series([len(group_dict[g]) for g in list(group_dict.keys())],index=list(group_dict.keys()))
        #max number of rows
        maxnrows = max(nb_rows)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization according to normed principal components analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        z_center, z_scale = wmean(X=Zcod,w=row_w), wstd(X=Zcod,w=row_w)
        #standardization : z_ik = (x_ik - m_k)/s_k
        Z = (Zcod - z_center)/z_scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #iterative algorithm
        # finding: C  compromis:   geometrical centroid of the transformed matrices c=sum(Xm* Hm)/M by minimizing the sum(norm(Xm* Hm- C)^2)
        # (Xm* Hm) is considered as transformed matrices
        # the solution is achived by applying an iterative algorithm
        #   Step1: define the inical centroid C
        #   Step2: finding the similarity taransformation matrices (Hm)
        #   Step3: after the calculation of (Xm* Hm), iterative updating of the centroid C 
        # until the global convergence 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #internal procrustes function
        def procrustes(
                X,Y
        ):
            svd = linalg.svd(X.T.dot(Y))        # svd(G) = USV
            rank = sum((svd[1]/svd[1][1])>self.tol)
            H = svd[0][:,:rank].dot(svd[2][:rank,:])              # rotation
            d = ((X.dot(H) - Y)**2).sum().sum() # dist|AH-B|=trace((AH-B)'AH-B)=sum((AH-B)^2)
            return H, d
        
        # Transposed dataset and divided by sqrt(n_m)
        tab = OrderedDict()
        for g, rows in group_dict.items():
            Z_g, nrows_g =  model[g].call_.Z/sqrt(nb_rows[g]), nb_rows[g]
            if nrows_g < maxnrows:
                difrows = maxnrows-nrows_g
                Z_g = concat((Z_g,DataFrame(zeros((difrows,n_cols)),columns=Z_g.columns,index=[f"row{x+1}_{g}" for x in range(difrows)])),axis=0)
            tab[g] = Z_g.T

        #Initialize
        C0, I0 = tab[uq_classe[0]].values, sum([sum(diag(model[g].call_.Vb)) for g in uq_classe])
        max_iter, threshold = 10, 1e-10
        while max_iter > threshold:
            C = zeros((n_cols,maxnrows))
            for g in uq_classe:
                Hg = procrustes(tab[g],C0)[0]
                C += tab[g].values.dot(Hg)

            C = C/len(uq_classe)
            #update criterion
            d0 = sum([procrustes(tab[g],C)[1] for g in uq_classe])
            max_iter = I0 - d0
            I0 = d0
            C0 = C
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen decomposition - finding common loadings of CC' 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen decomposition
        svd = linalg.svd(C.dot(C.T),hermitian=True)
        #set maximum number of components
        rank = sum(svd[1]/svd[1][0] > self.tol)
        
        #set number of components
        if self.ncp is None:
            ncp = rank
        elif self.ncp < 1: 
            raise ValueError("'ncp' must be equal or greater than 1.")
        else: 
            ncp = min(self.ncp,rank)

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,x=x,y=y,Xcod=Xcod,dummies=dummies,M=M,C=C,Zcod=Zcod,Z=Z,center=center,scale=scale,z_center=z_center,z_scale=z_scale,
                            row_w=row_w,var_w=var_w,col_w=col_w,ncp=ncp,group=group_label,ind_sup=ind_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #convert to namedtuple
        self.svd_ = namedtuple("svdResult",["V","vs","ncp","rank"])(svd[0],sqrt(svd[1][:rank]),ncp,rank)

        #eigen values informations
        eigvals = svd[1][:rank]
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(rank)])  

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables in compromises spaces
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #columns coordinates
        quanti_var_coord = DataFrame(self.svd_.V[:,:ncp]*self.svd_.vs[:ncp],index=Xcod.columns,columns=self.eig_.index[:self.ncp])
        #convert to ordered dictionary
        quanti_var_ = OrderedDict(coord=quanti_var_coord)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #groups informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_ = RVstats(model=model,tol=self.tol)
        # variance of each loading (lambd = V'*(Σ_m)*V)
        lambd =  concat((Series(diag(self.svd_.V[:,:ncp].T.dot(model[g].call_.Vb).dot(self.svd_.V[:,:ncp])),index=self.eig_.index[:ncp]).to_frame(g) for g in uq_classe),axis=1).T
        #add to group
        group_["lambd"] = lambd
        #explained variance
        group_["expl_var"] = concat((100*lambd.loc[g,:]/sum(diag(model[g].call_.Vb)) for g in uq_classe),axis=1).T
        #store all group informations
        self.group_ = namedtuple("group",group_.keys())(*group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals in compromises spaces
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = (Z *col_w).dot(self.svd_.V[:,:ncp])
        ind_coord.columns = self.eig_.index[:ncp]
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split in x and y
            y_ind_sup, X_ind_sup = X_ind_sup[group_label[0]], X_ind_sup.drop(columns=group_label)

            Xcod_ind_sup = X_ind_sup
            if is_all_object_or_category_dtype(X_ind_sup):
                Xcod_ind_sup = disjunctive(X_ind_sup,cols=dummies.columns) * M.loc[y_ind_sup.values,:].values
            
            #standardization
            Z_ind_sup = (((Xcod_ind_sup - center.loc[y_ind_sup.values,:].values)/scale.loc[y_ind_sup.values,:].values) - z_center)/z_scale
            #coordinates for supplementary individuals
            ind_sup_coord = (Z_ind_sup * col_w).dot(self.svd_.V[:,:self.svd_.ncp])
            ind_sup_coord.columns = self.eig_.index[:ncp]
            #convert to ordered dictionary
            ind_sup_ = OrderedDict(coord=ind_sup_coord)
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

        #split in x and y
        y, X = X[self.call_.group[0]], X.drop(columns=self.call_.group)

        Xcod = X
        if is_all_object_or_category_dtype(X):
            Xcod = disjunctive(X,cols=self.call_.dummies.columns) * self.call_.M.loc[y.values,:].values
        
        #standardization
        Z = (((Xcod - self.call_.center.loc[y.values,:].values)/self.call_.scale.loc[y.values,:].values) - self.call_.z_center)/self.call_.z_scale
        #coordinates for supplementary individuals
        coord = (Z * self.call_.col_w).dot(self.svd_.V[:,:self.svd_.ncp])
        coord.columns = self.eig_.index[:self.svd_.ncp]
        return coord