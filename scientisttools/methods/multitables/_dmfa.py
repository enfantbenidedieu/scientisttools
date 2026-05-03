# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, outer, diag, sum, dot, sqrt, linalg, empty
from pandas import DataFrame, Series, concat, CategoricalDtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd, wcorr, func_groupby
from ..functions.gfa import gFA
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..functions.utils import check_is_bool, is_all_numeric_dtype, is_all_object_or_category_dtype, check_is_dataframe
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix
from ..others._splitgroup import splitgroup, RVstats

class DMFA(BaseEstimator,TransformerMixin):
    """
    Dual Multiple Factor Analysis (DMFA)
    
    Performs Dual Multiple Factor Analysis (DMFA) in the sense of `Pagès and Le Dien <https://hal.science/hal-00704553v1>`_ with supplementary individuals and/or supplementary variables (continuous and/or categorical).

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

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (quantitative and/or qualitative).
    
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
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    group_ : group
        An object containing all the results for the groups, with the following attributes:

        eig : DataFrame of shape (maxcp_rv, 4)
            The eigenvalue of RV matrix, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

        coord : DataFrame of shape (n_groups, n_groups)
            The coordinates of the groups.

        coord_n : DataFrame of shape (n_groups, n_groups)
            The normalied coordinates of the groups.

        cos2 : DataFrame of shape (n_groups, n_groups)
            The sqared cosinus of the groups.

        traceRV : DataFrame of shape (n_groups, n_groups)
            The trace RV between groups.

        RV : DataFrame of shape (n_groups, n_groups)
            The RV coefficient between groups.

    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows,ncp)
            The coordinates of the individuals.
        cos2 : DataFrame of shape (n_rows, ncp)
            Thesquared cosinus of the individuals.
        contrib : DataFrame of shape (n_rows, ncp) 
            The relative contributions of the individuals.
        infos : DataFrame of shape (n_rows, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    ind_sup_ : ind_sup
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_rows_plus, ncp)
            The squared cosinus of the supplementary individuals.
        dist2 : Series of shape (n_rows_plus,)
            The squared distance to origin of the supplementary individuals.

    levels_sup_ : levels_sup 
        An object containing all the results for the supplementary levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the supplementary levels.
        dist2 : Series of shape (n_levels,)
            The squared distance to origin of the supplementary levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the supplementary levels.
        
    quali_var_sup_ : quali_var_sup 
        An object containing all the results for the supplementary qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary qualitative variables. The squared correlation ratio of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension

    quanti_var_ : quanti_var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.
        coord_partiel : coord_partiel
            An object containing the partiel coordinates of the variables for each group.

    quanti_var_sup_ : quanti_var_sup
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates of the supplementary quantitative variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus of the supplementary quantitative variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin of the supplementary quantitative variables.
        coord_partiel : coord_partiel
            An object containing the partiel coordinates of the supplementary qantitative variables for each group.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, ncp) or (n_groups, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.
    
    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    [2] Lê, S. & Pagès J. (2003). Deux extensions de l'Analyse Factorielle Multiple, thèse de doctorat.

    [3] Lê, S. & Pagès J. (2010). DMFA: dual multiple factor analysis. Communications in Statistics - Theory and Methods, 2010, 39 (3), pp.483-492. ⟨10.1080/03610920903140114⟩. ⟨hal-00704553⟩

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
    >>> from scientisttools import DMFA
    >>> clf = DMFA(group=4)
    >>> clf.fit(D)
    DMFA(group=4)
    """
    def __init__(
            self, scale_unit = True, ncp=5,  group = None, row_w = None, col_w = None, ind_sup = None, sup_var = None, tol = 1e-7
    ):  
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.group = group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
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
        group_label, ind_sup_label, sup_var_label = get_sup_label(X=X, indexes=self.group, axis=1), get_sup_label(X=X,indexes=self.ind_sup,axis=0), get_sup_label(X=X,indexes=self.sup_var,axis=1)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary variables
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
            if self.ind_sup is not None: 
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

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
        model = splitgroup(X=Xcod,y=y,scale_unit=self.scale_unit,ncp=self.ncp,row_w=row_w,col_w=col_w)
            
        #store separate analysis
        self.separate_analyses_ = model

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #scale_unitd data
        Zcod = concat((model[g].call_.Z for g in uq_classe),axis=0).loc[y.index,:]
        #weighted average
        center, scale = concat((model[g].call_.center.to_frame(g) for g in uq_classe),axis=1).T, concat((model[g].call_.scale.to_frame(g) for g in uq_classe),axis=1).T

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization according to normed principal components analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        z_center, z_scale = wmean(X=Zcod,w=row_w), wstd(X=Zcod,w=row_w)
        #standardization : z_ik = (x_ik - m_k)/s_k
        Z = (Zcod - z_center)/z_scale
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=Z,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
        #extract elements
        self.svd_, self.eig_, self.quanti_var_ = fit_.svd, fit_.eig, namedtuple("quanti_var",fit_.col.keys())(*fit_.col.values())
        #number of components kepted
        ncp = self.svd_.ncp

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,x=x,y=y,Xcod=Xcod,dummies=dummies,M=M,Zcod=Zcod,Z=Z,center=center,scale=scale,z_center=z_center,z_scale=z_scale,row_w=row_w,var_w=var_w,col_w=col_w,
                            ncp=ncp,group=group_label,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: coordinates, cos2
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to ordered dictionary - reorderd index
        ind_ = OrderedDict({k : fit_.row[k].loc[y.index,:] for k in list(fit_.row.keys())}) 
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables informations: partiel coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partiel coordinates for quantitative variables
        var_partiel = OrderedDict()
        for g, rows in group_dict.items():
            coord = wcorr(concat((model[g].call_.Z,self.ind_.coord.loc[rows,:]),axis=1),w=model[g].call_.row_w).iloc[:n_cols,n_cols:]
            coord.columns = self.eig_.index[:ncp]
            var_partiel[g] = coord
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group informations : coordinates, cos2
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates of the groups
        group_coord, group_eigvals, sum_sqeigvals = DataFrame(index=uq_classe,columns=self.eig_.index[:ncp]).astype(float), empty((len(uq_classe),),dtype=float), empty((len(uq_classe),),dtype=float)
        for i, g in enumerate(uq_classe):
            V = model[g].call_.R if self.scale_unit else model[g].call_.V
            evd = linalg.svd(V,hermitian=True)
            group_eigvals[i], sum_sqeigvals[i] = evd[1][0], sum(evd[1]**2)
            for j, d in enumerate(self.eig_.index[:ncp]):
                group_coord.loc[g,d] = sum(diag(outer(fit_.col["coord"].iloc[:,j],dot(fit_.col["coord"].iloc[:,j],V))))/self.eig_.iloc[j,0]
        #normalized coordinates and cos2 of the groups
        group_coord_n, group_sqcos = (group_coord.T/group_eigvals).T, 100*((group_coord**2).T/sum_sqeigvals).T
        #group contributions
        group_ctr = concat((self.ind_.contrib.loc[rows,:].sum(axis=0).to_frame(g) for g, rows in group_dict.items()),axis=1).T

        #convert to ordered dictionary
        group_ = OrderedDict(coord=group_coord,coord_n=group_coord_n,contrib=group_ctr,cos2=group_sqcos)
        #RV statistics
        rvstats = RVstats(model=model,tol=self.tol)
        #update group_ informations
        group_ = OrderedDict({**group_, **OrderedDict({k : rvstats[k] for k in ("traceRV","RV","eig")})})
        #store all group informations
        self.group_ = namedtuple("group",group_.keys())(*group_.values()) 

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split in x and y
            y_ind_sup, X_ind_sup = X_ind_sup[group_label[0]], X_ind_sup.drop(columns=group_label)

            Xcod_ind_sup = X_ind_sup.copy()
            if is_all_object_or_category_dtype(X_ind_sup):
                Xcod_ind_sup = disjunctive(X_ind_sup,cols=dummies.columns)*M.loc[y_ind_sup.values,:].values
            
            #standardization
            Z_ind_sup = (((Xcod_ind_sup - center.loc[y_ind_sup.values,:].values)/scale.loc[y_ind_sup.values,:].values) - z_center)/z_scale
            #statistics for supplementary individuals
            ind_sup_ = func_predict(X=Z_ind_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_var_sup > 0:
                #conditional weighted average
                center_sup = func_groupby(X=X_quanti_var_sup,by=y,func="mean",w=row_w)
                #conditional weighted standard deviation
                if self.scale_unit:
                    scale_sup = func_groupby(X=X_quanti_var_sup,by=y,func="std",w=row_w,ddof=1)
                else:
                    scale_sup = DataFrame(ones((center_sup.shape[0],n_quanti_var_sup)),columns=X_quanti_var_sup.columns,index=center.index)
                #standardization: z_ikl = (x_ikl - m_kl)/s_kl
                Zcod_quanti_var_sup = (X_quanti_var_sup - center_sup.loc[y.values,:].values)/scale_sup.loc[y.values,:].values
                #standardization: z_ik = (x_ik - m_k)/s_k
                z_quanti_var_sup_center, z_quanti_var_sup_scale = wmean(X=Zcod_quanti_var_sup,w=row_w), wstd(X=Zcod_quanti_var_sup,w=row_w)
                Z_quanti_var_sup = (Zcod_quanti_var_sup - z_quanti_var_sup_center)/z_quanti_var_sup_scale
                #statistics for supplementary quantitative variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
    
                #partiel coordinates for supplementary quantitative variables
                for g, rows in group_dict.items():
                    coord = wcorr(concat((Z_quanti_var_sup.loc[rows,:],self.ind_.coord.loc[rows,:]),axis=1),w=model[g].call_.row_w).iloc[:n_quanti_var_sup,n_quanti_var_sup:]
                    coord.columns = self.eig_.index[:ncp]
                    var_partiel[g] = concat((var_partiel[g],coord),axis=0)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_var_sup > 0:
                #create new qualitative columns
                X_quali_var_sup_new = concat((concat((X_quali_var_sup[x],y),axis=1).apply(lambda x: ''.join(x),axis=1) for x in X_quali_var_sup.columns),axis=1)
                X_quali_var_sup_new.columns = [f"{x}_{group_label[0]}" for x in X_quali_var_sup.columns]
                #concatenate
                X_quali_var_sup = concat((X_quali_var_sup,X_quali_var_sup_new),axis=1)
                #compute conditional weighted average
                Z_levels_sup = (func_groupby(X=Zcod,by=X_quali_var_sup,func="mean",w=row_w) - z_center)/z_scale
                #statistics for supplementary levels
                levels_sup_ = func_predict(X=Z_levels_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
                #proportion of supplementary levels
                p_k_sup = (disjunctive(X_quali_var_sup).T*row_w).sum(axis=1)
                #vtest for the supplementary levels
                levels_sup_["vtest"] = (levels_sup_["coord"].T*sqrt((n_rows-1)/((1/p_k_sup)-1))).T/fit_.svd.vs[:ncp]
                #convert to namedtuple
                self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

                ##statistics for supplementary qualitative variables
                #coordinates for the supplementary qualitative variables - Eta-squared
                quali_var_sup_coord = func_eta2(X=self.ind_.coord,by=X_quali_var_sup,w=row_w,excl=None)
                #convert to ordered dictionary
                quali_var_sup_ = OrderedDict(coord=quali_var_sup_coord)
                #convert to namedtuple
                self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())

        # convert to namedtuple
        self.var_partiel_ = namedtuple("var_partiel",var_partiel.keys())(*var_partiel.values())

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