# -*- coding: utf-8 -*-
from numpy import ndarray, array, zeros, ones, sqrt, linalg, log, flip,cumsum,mean, fill_diagonal, nan, cov
from pandas import DataFrame, Series, concat, CategoricalDtype
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2
from collections import namedtuple,OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.concat_empty import concat_empty
from ..functions.wlsreg import wlsreg
from ..functions.statistics import wmean, wstd, wcov, func_groupby
from ..functions.utils import check_is_bool
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..functions.cov2corr import cov2corr
from ..others._kaisermsa import kaisermsa
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)

    Performs Principal Component Analysis (PCA) and its derivatives with supplementary individuals, supplementary variables (continuous and/or categorical). 
    Missing values on continuous variables are replaced by the column mean. Missing values on categorical variables are replaced by the most frequent categories in columns.
    :class:`scientisttools.PCA` performns:

        1. Principal Component Analysis (PCA)
        2. Principal Component Analysis with partial correlation matrix (PCApartial)
        3. Principal Component Analysis with instrumental variables (PCAiv)
        4. Principal Component Analysis with orthogonal instrumental variables (PCAoiv)
        5. Between-class Principal Component Analysis (bcPCA)
        6. Within-class Principal Component Analysis (wcPCA)
    
    Parameters
    ----------
    scale_unit : bool, default = True
        If ``True``, then the data are scaled to unit variance.

    iv : int, list, tuple or range, default = None
        The indexes or names of the instrumental (explanatory) variables (quantitative and/or qualitative).

    ortho : bool, default = False
        If ``True``, then the principal analysis with orthogonal instrumental variables (PCAoiv) is performed.

    partiel : int, str, list, tuple or range, default = None
        The indexes or the names of the partial variables (quantitative and/or qualitative).

    ncp : int, default = 5
        The number of dimensions kept in the results.

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
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        bary : None or DataFrameof shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        x_center : Series of shape (n_columns,)
            The columns weighted average.
        x_scale : Series of shape (n_columns)
            The columns weighted standard deviation.
        center : Series of shape (n_columns,)
            The variables weighted average.
        scale : Series of shape (n_columns,)
            The variables standard deviation:

            - If `scale_unit = True`, then standard deviation are computed using variables weighted standard deviation
            - If `scale_unit = False`, then standard deviation are a vector of ones with length number of variables.
        ind_w : Series of shape (n_rows,) 
            The individuals weights.
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        ncp : int
            The number of components kepted.
        features : None, list
            The names of the explanatory variables (instrumental variables) or partial variables.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).
        zcod : DataFrame of shape (n_rows, n_features), optional
            The standardized features data.
        z_center : Series of shape (n_feature_quanti_var,), optional
            The weighted average of features variables.
        z_scale : Series of shape (n_feature_quanti_var,), optional
            The weighted standard deviation of features variables

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

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

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_rows_plus, ncp)
            The squared cosinus of the supplementary individuals.
        dist2 : Series of shape (n_rows_plus,)
            The squared distance to origin of the supplementary individuals.

    levels_sup_ : levels_sup, optional
        An object containing all the results for the supplementary levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the supplementary levels.
        dist2 : Series of shape (n_levels,)
            The squared distance to origin of the supplementary levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the supplementary levels.
        
    quali_var_sup_ : quali_var_sup, optional
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

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates of the supplementary quantitative variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus of the supplementary quantitative variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin of the supplementary quantitative variables.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, maxcp) or (n_groups, maxcp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, maxcp)
            The right singular vectors.

    References
    ----------
    [1] Bry X. (1996), Analyses factorielles multiple, Economica

    [2] Bry X. (1999), Analyses factorielles simples, Economica

    [3] Escofier B., Pagès J. (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    [4] Saporta G. (2006). Probabilites, Analyse des données et Statistiques. Technip

    [5] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    [6] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    [6] Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    [7] Rakotomalala, R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    [8] Tenenhaus, M. (2006). Statistique : Méthodes pour décrire, expliquer et prévoir. Dunod.
    
    See also
    --------
    :class:`scientisttools.save`
        Print results for general factor analysis model in an Excel sheet.
    :class:`scientisttools.sprintf`
        Print the analysis results.
    :class:`scientisttools.summary`
        Printing summaries of general factor analysis model.
    
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA
    >>> clf = PCA(ind_sup=range(41,46), sup_var=(10,11,12))
    >>> clf.fit(decathlon)
    PCA(ind_sup=range(41,46), sup_var=(10,11,12))
    """
    def __init__(
            self, scale_unit=True, iv=None, ortho=False, partial=None, group=None, option="between", ncp=5, row_w=None, col_w=None, ind_sup=None, sup_var=None, tol = 1e-7
    ):
        self.scale_unit = scale_unit
        self.iv = iv
        self.ortho = ortho
        self.partial = partial
        self.group = group
        self.option = option
        self.ncp = ncp
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
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.

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
        #check if partial and iv are both not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if all(x is not None for x in (self.partial,self.iv)): 
            raise ValueError("At least one should be None between 'iv' and 'partial'.")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if ortho is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is not None: 
            check_is_bool(self.ortho)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set scale_unit
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.scale_unit = True if self.partial is not None else False if self.iv is not None else self.scale_unit

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is not None and not isinstance(self.group,(int,str)):
            raise TypeError("'group' must be either an objet of type int or str")
        if self.group is not None and not self.option in ("between","within"):
            raise ValueError("'option' should be one of 'between', 'within'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get the features labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        features_label = get_sup_label(X=X,indexes=self.partial,axis=1) if self.partial is not None else get_sup_label(X=X, indexes=self.iv, axis=1) if self.iv is not None else None

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
            if self.group is not None:
                y_ind_sup, X_ind_sup = X_ind_sup[group_label[0]], X_ind_sup.drop(columns=group_label)
            if (self.iv is not None) or (self.partial is not None):
                z_ind_sup, X_ind_sup = X_ind_sup.loc[:,features_label], X_ind_sup.drop(columns=features_label)

        #drop features (partial or instrumental) variables
        if (self.iv is not None) or (self.partial is not None):
            z, X = X.loc[:,features_label], X.drop(columns=features_label)
        
        #extract group disribution
        if self.group is not None:
            y, X = X[group_label[0]], X.drop(columns=group_label)
            #unique element in y
            uq_classe = sorted(list(y.unique()))
            #convert y to categorical data type
            y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal components analysis (PCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #number of rows and number of columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and columns weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None: 
            ind_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'row_w' must be a 1d array-like of rows weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"'row_w' must be 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
        
        #set columns weights
        if self.col_w is None: 
            var_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'col_w' must be a 1d array-like of columns weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
        else: 
            var_w = Series(array(self.col_w),index=X.columns,name="weight")

        #compute weighted average and standard deviation of dependent variables
        x_center, x_scale = wmean(X=X,w=ind_w), wstd(X=X,w=ind_w)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Xcod, col_w = X.copy(), var_w.copy()
        if any(x is not None for x in (self.partial,self.iv)):
            #set target
            x = (X - x_center)/x_scale if self.iv is not None else X.copy()

            #split z (features)
            split_z = splitmix(z)
            z_quanti_var, z_quali_var, nz_quanti_var, nz_quali_var = split_z.quanti, split_z.quali, split_z.k1, split_z.k2
        
            zcod = None
            #add qualitative features
            if nz_quali_var > 0: 
                zcod = concat_empty(zcod,z_quali_var,axis=1)
            #add quantitative features
            if nz_quanti_var > 0:
                if self.partial is not None: 
                    zcod = concat_empty(zcod,z_quanti_var,axis=1)
                else:
                    z_center, z_scale = wmean(X=z_quanti_var,w=ind_w), wstd(X=z_quanti_var,w=ind_w)
                    zcod = concat_empty(zcod, (z_quanti_var - z_center)/z_scale,axis=1)
            #reorder columns 
            zcod = zcod[z.columns]
    
            #separate weighted least squared model
            self.separate_analyses_ = wlsreg(X=zcod,Y=x,w=ind_w)

            #set variables
            if (self.partial is not None) or (self.iv is not None and self.ortho): 
                Xcod = concat((self.separate_analyses_[k].resid.to_frame(k) for k in x.columns),axis=1)
            if self.iv is not None and not self.ortho: 
                Xcod = concat((self.separate_analyses_[k].fittedvalues.to_frame(k) for k in x.columns),axis=1)
            #set columns weight
            col_w = Series(ones(Xcod.shape[1]),index=Xcod.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and weighted standard deviation
        center, scale = wmean(X=Xcod,w=ind_w), wstd(X=Xcod,w=ind_w) if self.scale_unit else Series(ones(n_cols),index=Xcod.columns,name="scale")
        #standardization: z_ik = (x_ik - m_k)/s_k
        Z = (Xcod - center)/scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #class analysis (None/between/within)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set tab, row_w and bary
        tab, row_w, bary = Z.copy(), ind_w.copy(), None
        if self.group is not None:
            #update bary
            bary = func_groupby(X=Z,by=y,func="mean",w=ind_w).loc[uq_classe,:]
            #update tab and row_w
            if self.option == "between":
                tab, row_w = bary.copy(), Series([ind_w.loc[y[y==k].index].sum() for k in uq_classe],index=uq_classe,name="weight")
            else:
                tab, row_w = Z - bary.loc[y.values,:].values, ind_w.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)

        #extract elements
        self.svd_, self.eig_, ncp, self.quanti_var_ = fit_.svd, fit_.eig, fit_.ncp, namedtuple("quanti_var",fit_.col.keys())(*fit_.col.values())

        #convert to ordered dictionary - call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Xcod=Xcod,Z=Z,bary=bary,tab=tab,x_center=x_center,x_scale=x_scale,center=center,scale=scale,ind_w=ind_w,row_w=row_w,var_w=var_w,col_w=col_w,ncp=ncp,
                            features=features_label,group=group_label,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #add features
        if any(x is not None for x in (self.iv, self.partial)):
            call_["zcod"] = zcod
        #add weighted average and standard deviation for instrumental variables
        if (self.iv is not None) and any(is_numeric_dtype(z[x]) for x in z.columns): 
            call_ = {**call_, **OrderedDict(z_center=z_center,z_scale=z_scale)}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals and/or groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_ = fit_.row 
        if self.group is not None:
            #ratio - percentage of between-class/within-class inertia
            res_ = gSVD(X=Z,ncp=self.ncp,row_w=ind_w,col_w=col_w)
            if self.option == "between":
                group_, ind_ = fit_.row, func_predict(X=Z,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            else:
                group_ = func_predict(X=bary,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            self.ratio_, self.group_ = sum(self.eig_.iloc[:,0])/sum(res_.vs**2), namedtuple("group",group_.keys())(*group_.values())
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #set Xcod for supplementary individuals
            Xcod_ind_sup = X_ind_sup.copy()
            if any(x is not None for x in (self.iv,self.partial)): 
                #set x for supplementary individuals
                x_ind_sup = (X_ind_sup - x_center)/x_scale if self.iv is not None else X_ind_sup.copy()

                #split z_ind_sup
                split_z_ind_sup = splitmix(z_ind_sup)
                #extract elements
                z_ind_sup_quanti_var, z_ind_sup_quali_var, nz_ind_sup_quanti_var, nz_ind_sup_quali_var = split_z_ind_sup.quanti, split_z_ind_sup.quali, split_z_ind_sup.k1, split_z_ind_sup.k2
            
                zcod_ind_sup = None
                #Add qualitative variables
                if nz_ind_sup_quali_var > 0: 
                    if nz_ind_sup_quali_var != nz_quali_var: 
                        raise ValueError("Not convenient qualitative variable")
                    zcod_ind_sup = concat_empty(zcod_ind_sup,z_ind_sup_quali_var,axis=1)
                #add quantitative variables
                if nz_ind_sup_quanti_var > 0:
                    if  nz_ind_sup_quanti_var != nz_quanti_var: 
                        raise ValueError("Not convenient quantitative variable")
                    zcod_ind_sup = concat_empty(zcod_ind_sup,z_ind_sup_quanti_var,axis=1) if self.partial is not None else concat_empty(zcod_ind_sup,z_ind_sup_quanti_var.sub(z_center,axis=1).div(z_scale,axis=1),axis=1)
                
                #reorder columns
                zcod_ind_sup = zcod_ind_sup[z_ind_sup.columns]
                #predicted values
                Xcod_ind_sup = concat((self.separate_analyses_[k].predict(zcod_ind_sup).to_frame(k) for k in x_ind_sup.columns),axis=1)
                #residuals for PCApartial or PCAoiv
                if (self.partial is not None) or (self.iv is not None and self.ortho): 
                    Xcod_ind_sup = x_ind_sup.sub(Xcod_ind_sup.values)

            #standardization: z_ik = (x_ik - m_k)/s_k
            Z_ind_sup = (Xcod_ind_sup - center)/scale

            #within class analysis - suppress within effect
            if self.group is not None and self.option == "within":
                Z_ind_sup = Z_ind_sup - bary.loc[y_ind_sup.values,:].values

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
                #set Xcod_quanti_var_sup
                Xcod_quanti_var_sup = X_quanti_var_sup.copy()
                if any(x is not None for x in (self.iv,self.partial)):
                    x_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=ind_w))/wstd(X=X_quanti_var_sup,w=ind_w) if self.iv is not None else X_quanti_var_sup.copy()
                    #update dictionary
                    self.separate_analyses_ = {**self.separate_analyses_, **wlsreg(X=zcod,Y=x_quanti_var_sup,w=ind_w)}
                    #set variables
                    if (self.partial is not None) or (self.iv is not None and self.ortho): 
                        Xcod_quanti_var_sup = concat((self.separate_analyses_[k].resid.to_frame(k) for k in x_quanti_var_sup.columns),axis=1)
                    if self.iv is not None and not self.ortho: 
                        Xcod_quanti_var_sup = concat((self.separate_analyses_[k].fittedvalues.to_frame(k) for k in x_quanti_var_sup.columns),axis=1)

                #compute weighted average for supplementary quantitative variables
                center_sup, scale_sup = wmean(X=Xcod_quanti_var_sup,w=ind_w), wstd(X=Xcod_quanti_var_sup,w=ind_w) if self.scale_unit else ones(n_quanti_var_sup)
                #standardization: z_ik = (x_ik - m_k)/s_k
                Z_quanti_var_sup = (Xcod_quanti_var_sup - center_sup)/scale_sup

                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    Z_quanti_var_sup = bary_quanti_var_sup if self.option == "between" else Z_quanti_var_sup - bary_quanti_var_sup.loc[y.values,:].values

                #statistics for supplementary quantitative variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_var_sup > 0:
                #conditional mean - Barycenter of original data
                X_levels_sup = func_groupby(X=Xcod,by=X_quali_var_sup,w=ind_w,func="mean")
                #standardization: z_ik = (x_ik - m_k)/s_k
                Z_levels_sup = (X_levels_sup - center)/scale
                #statistics for supplementary levels
                levels_sup_ = func_predict(X=Z_levels_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
                #proportion of supplementary levels
                p_k_sup = (disjunctive(X_quali_var_sup).T * ind_w).sum(axis=1)
                #vtest for the supplementary levels
                levels_sup_["vtest"] = (levels_sup_["coord"].T * sqrt((n_rows-1)/((1/p_k_sup) - 1))).T/fit_.svd.vs[:self.call_.ncp]
                #convert to namedtuple
                self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

                #coordinates for the supplementary qualitative variables - Eta-squared
                quali_var_sup_coord = func_eta2(X=self.ind_.coord,by=X_quali_var_sup,w=ind_w,excl=None)
                #convert to ordered dictionary
                quali_var_sup_ = OrderedDict(coord=quali_var_sup_coord)
                #convert to namedtuple
                self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())
                
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
        X_new : DataFrame of shape (n_rows, a)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
def statsPCA(
        obj
):
    """
    Statistics with Principal Component Analysis

    Performs statistics with principal component analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`.

    Returns
    -------
    result : statPCAResult
        A object with the following attributes

        corr_ : corr
            An object containing all the results for the correlation with the following attributes:  

            corrcoef: DataFrame of shape (n_columns, n_columns) 
                The pearson correlation coefficient matrix.
            pcorrcoef: DataFrame of shape (n_columns, n_columns) 
                The partial pearson correlation coefficient matrix
            reconst: DataFrame of shape (n_columns, n_columns) 
                The reconstitution pearson correlation coefficient matrix
            residual: DataFrame of shape (n_columns, n_columns) 
                The residual correlation matrix

        others_ : others
            An object with the following attributes:

            threshold : DataFrame of shape (1,2)
                Eigen values threshold: kaiser, kaiser proportion and KSS (Karlis - Saporta - Spinaki).
            bartlett: DataFrame of shape (1,4)
                The Bartlett's test of Spericity.
            broken: Series of shape (max_components, 2)
                The broken's stick threshold.
            msa: Series of shape (n_columns + 1,)
                The Kaiser measure of sampling adequacy.
            
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class PCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCA":
        raise TypeError("'obj' must be an object of class PCA")

    #set number of rows and columns
    n_rows, n_cols = obj.call_.X.shape
    maxncp, colnames = obj.eig_.shape[0], obj.call_.Z.columns

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #correlation matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #covariance/correlation of Z and reconst
    M = wcov(obj.call_.Z,w=obj.call_.row_w,ddof=0)
    #inverse of M
    try: 
        inv_M = DataFrame(linalg.inv(M),index=colnames,columns=colnames)
    except linalg.LinAlgError: 
        inv_M = DataFrame(linalg.pinv(M),index=colnames,columns=colnames)
    #weighted partial correlation matrix and reconst covariance/correlation
    partial_M, reconst_M = -1*cov2corr(inv_M), (obj.quanti_var_.coord.T * obj.call_.col_w).T.dot(obj.quanti_var_.coord.T)
    for c in partial_M.columns:
        partial_M.loc[c,c] = 1
    #residual covariance/correlation
    resid_M = M - reconst_M.values
    for c in resid_M.columns:
        resid_M.loc[c,c] = nan
    #convert to ordered dictionary
    corr_ = OrderedDict(corrcoef=M,pcorrcoef=partial_M,reconst=reconst_M,resid=resid_M)
    #convert to namedtuple
    corr_ = namedtuple("corr",corr_.keys())(*corr_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #bartlett's test
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Bartlett - statistics
    bartlett_stats, bs_dof = -(n_rows-1-(2*n_cols+5)/6)*sum(log(obj.eig_.iloc[:,0])), n_cols*(n_cols-1)/2
    bs_pvalue = chi2.sf(bartlett_stats,df=bs_dof)
    bartlett = DataFrame([[linalg.det(M),bartlett_stats,bs_dof,bs_pvalue]],columns=["|CORR.MATRIX|","CHISQ","dof","p-value"],index=["Bartlett's test"])
    bartlett["dof"] = bartlett["dof"].astype(int)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #others informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Karlis - Saporta - Spinaki threshold
    kss_th =  1 + 2*sqrt((maxncp-1)/(n_rows-1))
    #eigen value threshold
    eig_th = DataFrame([[mean(obj.eig_.iloc[:,0]),kss_th]],columns=["Kaiser-Guttman","Karlis-Saporta-Spinaki"],index=["Critical values"])
    
    #broken-stick crticial values
    broken = Series(cumsum([1/x for x in range(maxncp,0,-1)])[::-1],name="Broken-stick critical values",index=[f"Dim{x+1}" for x in range(maxncp)])
    broken = concat((obj.eig_.iloc[:,0],broken),axis=1)
    #convert to ordered dictionary
    others_ = OrderedDict(threshold=eig_th,bartlett=bartlett,broken=broken,msa=kaisermsa(X=obj.call_.X,w=obj.call_.ind_w))
    #convert to namedtuple
    others_ = namedtuple("others",others_.keys())(*others_.values())
    return namedtuple("statsPCAResult",["corr_","others_"])(corr_,others_)