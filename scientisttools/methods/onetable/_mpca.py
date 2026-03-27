# -*- coding: utf-8 -*-
from numpy import array,ndarray,sqrt,linalg,ones,diag
from pandas import concat, Series, CategoricalDtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from ._pca import PCA
from ._mca import MCA
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..functions.statistics import wmean, wstd, func_groupby
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class MPCA(BaseEstimator,TransformerMixin):
    """
    Mixed Principal Component Analysis (MPCA)
    
    Mixed principal component analusis performs principal component analysis of a set of individuals (observations) described by a mixture of qualitative and quantitative variables with supplementary individuals, supplementary variables (continuous and/or categorical).
    :class:`scientisttools.MPCA` also performns Between-class Mixed Principal Component Analysis (bcMPCA) and within-class Detrended Correspondence Analysis (wcMPCA).

    Parameters
    ----------
    group : int, str
        The indexe or name of the categorical variable which allows for between-class or within-class analysis.

    option : str, default = "between"
        Which class analysis should be performns.

        - 'between' for between-class analysis.
        - 'within' for within-class analysis.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional individuals weights. The weights are given only for the active individuals.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional variables weights. The weights are given only for the active variables.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (continuous and/or categorical).
    
    Returns
    -------
    call_ : call
        An object with the following attributes
    
    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    group_ : group, optional
        An object containing all the results for the groups with the following attributes:

        coord : DataFrame of shape (n_groups, ncp)
            The coordinates for the groups.
        cos2 : DataFrame of shape (n_groups, ncp)
            The squared cosinus for groups.
        contrib : DataFrame of shape (n_groups, ncp), optional
            The relative contributions for the groups.
        dist2 : Series of shape (n_groups,), optional
            The squared distance to origin for the groups.
        infos : DataFrame of shape (n_groups, 4), optional
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the groups.

    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows, ncp)
            The coordinates of the individuals.
        cos2 : DataFrame of shape (n_rows, ncp)
            Thesquared cosinus of the individuals.
        contrib : DataFrame of shape (n_rows, ncp), optional
            The relative contributions of the individuals.
        dist2 : Series of shape (n_rows,), optional
            The squared distance to origin for the rows.
        infos : DataFrame of shape (n_rows, 4), optional
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    ind_sup_ : ind_sup
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_rows_plus, ncp)
            The squared cosinus of the supplementary individuals.
        dist2 : Series of shape (n_rows_plus,)
            The squared distance to origin of the supplementary individuals.

    levels_ : levels
        An object containing all the results for the active levels, with the following attributes:
        
        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the levels.
        contrib : DataFrame of shape (n_levels, ncp)
            The relative contributions of the levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test of the levels.

    levels_sup_ : levels_sup_
        An object containing all the results for the supplementary levels, with the following attributes:
        
        coord : DataFrame of shape (n_levels_sup, ncp)
            The coordinates of the supplementary levels.
        coord_n : DataFrame of shape (n_levels_sup, ncp)
            The barycenter coordinates of the supplementary levels.
        cos2 : DataFrame of shape (n_levels_sup, ncp)
            The squared cosinus of the supplementary levels.
        dist2 : Series of shape (n_levels_sup,)
            the squared distance to origin of the supplementary levels.
        vtest : DataFrame of shape (n_levels_sup, ncp)
            The value-test of the supplementary levels.

    quali_var_ : quali_var
        An object containing all the results for the active qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var, ncp)
            The coordinates of the qualitative variables, which is eta2, the square correlation coefficient between a qualitative variable and a dimension.
        contrib : DataFrame of shape (n_quali_var, ncp)
            The contributions of the qualitative variables.

    quali_var_sup_ : quali_var_sup 
        An object containing all the results for the supplementary qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var_sup, ncp)
            The coordinates of the supplementary qualitative variables. The squared correlation ratio of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension

    quanti_var_ : quanti_var
        An object containing all the results for the active quantitative variables, with the following attributes:

        coord : DataFrame of shape (n_quanti_var + n_levels, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_quanti_var + n_levels, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_quanti_var + n_levels, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_quanti_var + n_levels, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.

    quanti_var_sup_ : quanti_var_sup
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates of the supplementary quantitative variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus of the supplementary quantitative variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin of the supplementary quantitative variables.
    
    ratio_ : float, optional
        The inertia (between-class/within-class) percentage.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, maxcp) or (n_groups, maxcp)
            The left singular vectors.
        V : 2d numpy array of shape (n_quanti_var + n_levels, maxcp)
            The right singular vectors.

    var_ : var
        An object containing all the results for the active variables (quantitative and qualitative), with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the variables.
    
    References
    ----------
    [1] Abdesselam R. (2006), Analyse en Composantes Principales Mixtes, CREM UMR CNRS 6211
    
    [2] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    [3] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    [4] Pages J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    [5] Rakotomalala, R (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

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
    >>> from scientisttools.datasets import autos1990, tea
    >>> from scientisttools import MPCA
    >>> #mixed principal component analysis (MPCA)
    >>> clf = MPCA()
    >>> clf.fit(autos1990)
    MPCA()
    >>> #between-class MPCA (bcMPCA)
    >>> clf = MPCA(group=20,ind_sup=range(200,300),sup_var=range(21,tea.shape[1]))
    >>> clf.fit(tea)
    MPCA(group=20,ind_sup=range(200,300),sup_var=range(21,tea.shape[1]))
    >>> #within-class MPCA (wcMPCA)
    >>> clf = MPCA(group=20,option='within',ind_sup=range(200,300),sup_var=range(21,tea.shape[1]))
    >>> clf.fit(tea)
    MPCA(group=20,ind_sup=range(200,300),option='within',sup_var=range(21,tea.shape[1]))
    """
    def __init__(
            self, group=None, option="between", ncp=5, row_w=None, col_w=None, ind_sup=None, sup_var=None
    ):
        self.group = group
        self.option = option
        self.ncp = ncp
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var

    def fit(self,X, y=None):
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
        #get labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_label, ind_sup_label, sup_var_label = get_sup_label(X=X, indexes=self.group, axis=1), get_sup_label(X=X, indexes=self.ind_sup, axis=0), get_sup_label(X=X, indexes=self.sup_var, axis=1)

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

        #extract group disribution
        if self.group is not None:
            y, X = X[group_label[0]], X.drop(columns=group_label)
            #unique element in y
            uq_classe = sorted(list(y.unique()))
            #convert y to categorical data type
            y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #mixed principal component analysis (MPCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X
        split_X = splitmix(X)
        #extract all elements
        X_quanti, X_quali, n_rows, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

        #check if mixed data
        if any(x == 0 for x in (n_quanti, n_quali)): 
            raise TypeError("MPCA require both continuous and categorical variables.")
        n_cols = n_quanti + n_quali

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None: 
            ind_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")

        #set variable weights
        if self.col_w is None: 
            var_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'col_w' must be a 1d array-like of variables weights")
        elif len(self.col_w) != n_cols: 
            raise TypeError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
        else: 
            var_w = Series(array(self.col_w),index=X.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate analyses
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        xmodel = PCA(scale_unit=True,ncp=self.ncp,row_w=ind_w,col_w=var_w[X_quanti.columns],sup_var=list(X_quali.columns)).fit(X)
        ymodel = MCA(ncp=self.ncp,row_w=ind_w,col_w=var_w[X_quali.columns],sup_var=list(X_quanti.columns)).fit(X)
        self.separate_analyses_ = OrderedDict({"PCA" : xmodel, "MCA" : ymodel})
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center numerics variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average mean and standard deviation
        center1 = wmean(X=X_quanti,w=ind_w)
        #center quantitatives variables
        X1c = X_quanti - center1

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #treatment of categorics variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #recode categorical variables
        dummies = disjunctive(X=X_quali)
        #covariance matrix between X and between Y and X
        Vx, Vyx = (X1c.T * ind_w).dot(X1c), (dummies.T * ind_w).dot(X1c)
        #compute the mean
        center2 = Series(linalg.multi_dot([Vyx,linalg.pinv(Vx,hermitian=True).T,X1c.T,diag(ind_w),ones(n_rows)]),index=dummies.columns,name="center")
        #center the disjunctive table
        X2c = dummies - center2
        #duplicate according to number of levels
        nb_moda = Series([X_quali[j].nunique() for j in X_quali.columns],index=X_quali.columns)
        #levels weights
        levels_w = Series(list(chain(*[repeat(i,k) for i, k in zip(var_w.loc[X_quali.columns],nb_moda)])),index=dummies.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #concatenate
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Xcod, Xc, center, col_w = concat((X_quanti,dummies),axis=1), concat((X1c,X2c),axis=1), concat((center1,center2),axis=0), concat((var_w.loc[X_quanti.columns],levels_w),axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization according to normed PCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        xc_center, xc_scale = wmean(X=Xc,w=ind_w), wstd(X=Xc,w=ind_w)
        #standardization: z_ik = (x_ik - m_k)/s_k
        Z = (Xc - xc_center)/xc_scale

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
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w)
        #extract elements
        self.svd_, self.eig_, ncp, self.quanti_var_ = fit_.svd, fit_.eig, fit_.ncp, namedtuple("quanti_var",fit_.col.keys())(*fit_.col.values())

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Xcod=Xcod,Xc=Xc,Z=Z,bary=bary,tab=tab,center=center,xc_center=xc_center,xc_scale=xc_scale,k1=n_quanti,k2=n_quali,ind_w=ind_w,row_w=row_w,var_w=var_w,
                            levels_w=levels_w,col_w=col_w,ncp=ncp,ind_sup=ind_sup_label,sup_var=sup_var_label)
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
        #statistics for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the levels as barycenter of individuals
        levels_coord = func_groupby(X=self.ind_.coord,by=X_quali,w=ind_w,func="mean")
        #contribution for the levels
        levels_ctr = fit_.col["contrib"].loc[levels_coord.index,:]
        #proportion for the levels
        p_k = (dummies.T * row_w).sum(axis=1)
        #vtest for the levels
        levels_vtest = (levels_coord.T * sqrt((n_rows-1)/((1/p_k) - 1))).T/fit_.svd.vs[:ncp]
        #convert to ordered dictionary
        levels_ = OrderedDict(coord=levels_coord, contrib=levels_ctr, vtest=levels_vtest)
        #convert to namedtuple
        self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eta2 for the qualitative variables
        quali_var_coord = func_eta2(X=self.ind_.coord,by=X_quali,w=ind_w,excl=None)
        #contrib of the qualitative variables
        quali_var_ctr = concat((levels_ctr.loc[levels_ctr.index.isin(list(X_quali[j].unique())),:].sum(axis=0).to_frame(j) for j in X_quali.columns),axis=1).T
        #convert to dictionary
        quali_var_ = OrderedDict(coord=quali_var_coord,contrib=quali_var_ctr)
        #convert to namedtuple
        self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to ordered dictionary
        var_= OrderedDict(coord=concat((fit_.col["cos2"].loc[X_quanti.columns,:],quali_var_coord),axis=0),contrib=concat((fit_.col["contrib"].loc[X_quanti.columns,:],quali_var_ctr),axis=0))
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split data
            split_X_ind_sup = splitmix(X_ind_sup)
            #extract elements
            X_ind_sup_quanti, X_ind_sup_quali = split_X_ind_sup.quanti, split_X_ind_sup.quali
            #disjunctive table for supplementary individuals
            dummies_ind_sup = disjunctive(X=X_ind_sup_quali,cols=dummies.columns)
            #concatenate
            Xcod_ind_sup = concat((X_ind_sup_quanti,dummies_ind_sup),axis=1)
            #standardize the data
            Z_ind_sup = ((Xcod_ind_sup - center) - xc_center)/xc_scale

            #within class analysis - suppress within effect
            if self.group is not None and self.option == "within":
                Z_ind_sup = Z_ind_sup - bary.loc[y_ind_sup.values,:].values

            #statistics for supplementary individuals
            ind_sup_ = func_predict(X=Z_ind_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            #extract elements
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_var_sup > 0:
                #standardization : zc_ik = (xc_ik - mc_k)/sc_k
                Z_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=ind_w))/wstd(X=X_quanti_var_sup,w=ind_w)

                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    Z_quanti_var_sup = bary_quanti_var_sup if self.option == "between" else Z_quanti_var_sup - bary_quanti_var_sup.loc[y.values,:].values

                #statistics for supplementary quantitative variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())
            
            #statistics for qualitative variables/levels
            if n_quali_var_sup > 0:
                #disjunctive table for supplementary levels
                dummies_sup = disjunctive(X_quali_var_sup)
                #covariance matrix
                Vyx_sup = (dummies_sup.T*ind_w).dot(X1c)
                #compute the average
                center2_sup = Series(linalg.multi_dot([Vyx_sup,linalg.pinv(Vx,hermitian=True).T,X1c.T,diag(ind_w),ones(n_rows)]),index=dummies_sup.columns,name="center")
                #center disjunctive table
                Xc_levels_sup = dummies_sup - center2_sup
                #standardization : zc_ik = (xc_ik - mc_k)/sc_k
                Z_levels_sup = (Xc_levels_sup - wmean(Xc_levels_sup,w=ind_w))/wstd(X=Xc_levels_sup,w=ind_w)

                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_levels_sup = func_groupby(X=Z_levels_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    Z_levels_sup = bary_levels_sup if self.option == "between" else Z_levels_sup - bary_levels_sup.loc[y.values,:].values

                #statistics for supplementary quantitative variables
                levels_sup_ = func_predict(X=Z_levels_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #coordinates for the supplementary levels as barycenter of individuals
                levels_sup_["coord_n"] = func_groupby(X=self.ind_.coord,by=X_quali_var_sup,w=ind_w,func="mean")
                #proportion for the supplementary levels
                p_k_sup = (dummies_sup.T *row_w).sum(axis=1)
                #vtest for the supplementary levels
                levels_sup_["vtest"] = (levels_sup_["coord_n"].T *sqrt((n_rows-1)/((1/p_k_sup) - 1))).T/fit_.svd.vs[:ncp]
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
        X_new : DataFrame of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord