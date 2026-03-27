# -*- coding: utf-8 -*-
from numpy import array, ndarray, sqrt, ones
from pandas import Series, CategoricalDtype, concat
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
from ..functions.concat_empty import concat_empty
from ..functions.statistics import wmean, wstd, func_groupby
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class FAMD(BaseEstimator,TransformerMixin):
    """
    Factor Analysis of Mixed Data (FAMD)
    
    Performs Factor Analysis of Mixed Data (FAMD) and its derivatives with supplementary individuals, supplementary variables (continuous and/or categorical).
    Missing values on continuous variables are replaced by the column mean. Missing values on categorical variables are replaced by the most frequent categories in columns.
    
    .. note:: 
        Its includes standard Principal Component Analysis (PCA) and Multiple Correspondence Analysis (MCA) as special cases. If all variables are quantitative, standard PCA is performed.
        If all variables are qualitative, then standard MCA is performed. When all the variable are categorical, the factor coordinates of the individuals are equal to the factor scores
        of standard MCA times squares root of :math::`J` (the number of categorical variables) and the eigenvalues are then equal to the usual eigenvalues of MCA times :math::`J`.
        When all the variables are quantitative, FAMD gives exactly the same results as normed PCA.

    :class::`scientisttools.FAMD` performns:

        1. Normed Principal Component Analysis (PCA)
        2. Standard Multiple Correspondence Analysis (MCA)
        3. Factor Analysis of Mixed Data (FAMD)
        4. Between-class Normed Principal Component Analysis (bcPCA)
        5. Between-class Standard Multiple Correspondence Analysis (bcMCA)
        6. Between-class Factor Analysis of Mixed Data (bcFAMD)
        7. Within-class Normed Principal Component Analysis (wcPCA)
        8. Within-class Standard Multiple Correspondence Analysis (wcMCA)
        9. Within-class Factor Analysis of Mixed Data (wcFAMD)

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
        dummies : DataFrame of shape (n_rows, n_levels)
            Disjunctive table.
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        Zcod : DataFrame of shape (n_rows, n_columns)
            Standadized recoded data.
        Z : DataFrame of shape (n_rows, columns)
            Standardized data
        bary : None or DataFrameof shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        k1 : int
            The number of continuous variables.
        k2 : int
            The number of categorical variables.
        ind_w : Series of shape (n_rows,)
            The individuals weights.
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        var_w : Series of shape (n_quali_var,)
            The categorical variables weights.
        col_w : Series of shape (n_columns,)
            The categories weights.
        center : Series of shape (n_columns,)
            The columns average.
        scale : Series of shape (n_columns,)
            The standard deviation of the columns.
        z_center : Series of shape (n_columns,)
            The weighted average of standardized recoded data.
        ncp : int
            The number of components kepted.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical)

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
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the levels.
        contrib : DataFrame of shape (n_levels, ncp)
            The relative contributions of the levels.
        dist2 : Series of shape (n_levels,)
            the squared distance to origin of the levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test of the levels.

    levels_sup_ : levels_sup_
        An object containing all the results for the supplementary levels, with the following attributes:
        
        coord : DataFrame of shape (n_levels_sup, ncp)
            The coordinates of the supplementary levels.
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

        coord : DataFrame of shape (n_quanti_var, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_quanti_var, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_quanti_var, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_quanti_var, 4)
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
        U : 2d numpy array of shape (n_rows, ncp) or (n_groups, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_quanti_var + n_levels, ncp)
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
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    [2] Pagès J. (2004). <Analyse factorielle de donnees mixtes https://www.numdam.org/article/RSA_2004__52_4_93_0.pdf>_. Revue Statistique Appliquee. LII (4). pp. 93-111.

    [3] Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    [4] Rakotomalala, Ricco (2020), <Pratique des méthodes factorielles avec Python https://hal.science/hal-04868625v1>_. Université Lumière Lyon 2, Version 1.0

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
    >>> from scientisttools.datasets import autos2005, decathlon, canines, tea
    >>> from scientisttools import FAMD
    >>> #PCA with FAMD function
    >>> clf = FAMD(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> clf.fit(decathlon)
    FAMD(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> #MCA with FAMD function
    >>> clf = FAMD(ind_sup=range(27,33),sup_var=(6,7))
    >>> clf.fit(canines)
    FAMD(ind_sup=range(27,33),sup_var=(6,7))
    >>> #Mixed Data with FAMD function
    >>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> clf.fit(autos2005)
    FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> #between-class FAMD (bcFAMD)
    >>> clf = FAMD(group=20,ind_sup=range(200,300),sup_var=range(21,tea.shape[1]))
    >>> clf.fit(tea)
    FAMD(group=20,ind_sup=range(200,300),sup_var=range(21,tea.shape[1]))
    >>> #within-class FAMD (wcFAMD)
    >>> clf = FAMD(group=20,option='within',ind_sup=range(200,300),sup_var=range(21,tea.shape[1]))
    >>> clf.fit(tea)
    FAMD(group=20,ind_sup=range(200,300),option='within',sup_var=range(21,tea.shape[1]))
    """
    def __init__(
            self, group=None, option="between", ncp=5, row_w=None, col_w=None, ind_sup=None, sup_var=None, tol = 1e-7
    ):
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
        #factor analysis of mixed data (FAMD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #et number of rows and columns
        n_rows, n_cols = X.shape

        #split X
        split_X = splitmix(X)
        X_quanti, X_quali, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.k1, split_X.k2

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights and columns weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals
        if self.row_w is None: 
            ind_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")

        #set variables weights
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
        if all(x > 0 for x in (n_quali,n_quanti)):
            xmodel = PCA(scale_unit=True,ncp=self.ncp,row_w=ind_w,col_w=var_w[X_quanti.columns],sup_var=list(X_quali.columns)).fit(X)
            ymodel = MCA(ncp=self.ncp,row_w=ind_w,col_w=var_w[X_quali.columns],sup_var=list(X_quanti.columns)).fit(X)
            self.separate_analyses_ = OrderedDict({"PCA" : xmodel, "MCA" : ymodel})
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #initialization
        Xcod, center, scale, col_w, dummies = None, None, None, None, None
        #concatenate
        if n_quanti > 0:
            Xcod, col_w, scale = concat_empty(Xcod,X_quanti,axis=1), concat_empty(col_w,var_w[X_quanti.columns],axis=0), concat_empty(scale,wstd(X=X_quanti,w=ind_w),axis=0)
        if n_quali > 0:
            #disjunctive table
            dummies = disjunctive(X=X_quali)
            #set number of categorics by qualitative variable
            nb_moda = array([X_quali[j].nunique() for j in X_quali.columns])
            #set levels weights
            levels_w = Series(array(list(chain(*[repeat(i,k) for i, k in zip(var_w[X_quali.columns],nb_moda)]))),index=dummies.columns,name="weight")
            #concatenate
            Xcod, col_w, scale = concat_empty(Xcod,dummies,axis=1), concat_empty(col_w,levels_w,axis=0), concat_empty(scale,Series(sqrt(dummies.mul(ind_w,axis=0).sum(axis=0)),index=dummies.columns,name="scale"),axis=0)
        
        #weighted average
        center = wmean(X=Xcod,w=ind_w)
        #standardization: z_ik = (x_ik - m_k)/s_k
        Zcod = (Xcod - center)/scale
        #centering according to non-normed principal components analysis
        z_center = wmean(X=Zcod,w=ind_w)
        Z = Zcod - z_center

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #class analysis (None/between/within)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            tab, row_w, bary = Z.copy(), ind_w.copy(), None
        else:
            bary = func_groupby(X=Z,by=y,func="mean",w=ind_w).loc[uq_classe,:]
            if self.option == "between":
                tab, row_w = bary.copy(), Series([ind_w.loc[y[y==k].index].sum() for k in uq_classe],index=uq_classe,name="weight")
            else:
                tab, row_w = Z - bary.loc[y.values,:].values, ind_w.copy()
                
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
        #extract elements
        self.svd_, self.eig_, ncp = fit_.svd, fit_.eig, fit_.ncp
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Xcod=Xcod,Zcod=Zcod,Z=Z,tab=tab,bary=bary,k1=n_quanti,k2=n_quali,ind_w=ind_w,row_w=row_w,var_w=var_w,col_w=col_w,center=center,scale=scale,z_center=z_center,ncp=ncp,
                            group=group_label,ind_sup=ind_sup_label,sup_var=sup_var_label)
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals and/or classes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            self.ind_ = namedtuple("ind",fit_.row.keys())(*fit_.row.values())
        else:
            #ratio - percentage of class inertia
            res_ = gSVD(X=Z,ncp=self.ncp,row_w=ind_w,col_w=col_w)
            if self.option == "between":
                group_, ind_ = fit_.row, func_predict(X=Z,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            else:
                ind_, group_ = fit_.row, func_predict(X=bary,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            self.ratio_, self.ind_, self.group_ = sum(self.eig_.iloc[:,0])/sum(res_.vs**2), namedtuple("ind",ind_.keys())(*ind_.values()), namedtuple("group",group_.keys())(*group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for active quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if n_quanti > 0:
            #convert to dictionary
            quanti_var_ = {x : fit_.col[x].iloc[:n_quanti,:] for x in list(fit_.col.keys())}
            #convert to namedtuple
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for active levels - active qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if n_quali > 0:
            #proportion for the leves
            p_k = center.iloc[n_quanti:]
            #standardization: z_gk = (x_gk - m_k)
            Z_levels = func_groupby(X=Zcod,by=X_quali,w=ind_w,func="mean").sub(z_center,axis=1)
            #statistics for the levels
            levels_ = func_predict(X=Z_levels,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #contributions of the levels
            levels_["contrib"] = fit_.col["contrib"].iloc[n_quanti:,:]
            #vtest of the levels
            levels_["vtest"] = (levels_["coord"].T * sqrt((n_rows - 1)/((1/p_k) - 1))).T/fit_.svd.vs[:ncp]
            #convert to namedtuple
            self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

            #coordinates for the qualitative variables  - Eta-squared
            quali_var_coord = func_eta2(X=self.ind_.coord,by=X_quali,w=ind_w,excl=None)
            #contributions of qualitative variables
            quali_var_ctr =  concat((self.levels_.contrib.loc[self.levels_.contrib.index.isin(list(X_quali[j].unique())),:].sum(axis=0).to_frame(j) for j in X_quali.columns),axis=1).T
            #convert to ordered dictionary
            quali_var_ = OrderedDict(coord=quali_var_coord,contrib=quali_var_ctr)
            #convert to namedtuple
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if all(x > 0 for x in (n_quanti,n_quali)):
            #convert to ordered dictionary
            var_= OrderedDict(coord=concat_empty(quanti_var_["cos2"],quali_var_coord,axis=0),contrib=concat_empty(quanti_var_["contrib"],quali_var_ctr,axis=0),
                              cos2=concat_empty(quanti_var_["cos2"]**2,((quali_var_coord**2).T/(nb_moda-1)).T,axis=0))
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

            #initialize the data
            Xcod_ind_sup = None
            if n_quanti > 0:
                #concatenate
                Xcod_ind_sup = concat_empty(Xcod_ind_sup,X_ind_sup_quanti,axis=1)
            if n_quali > 0:
                #concatenate
                Xcod_ind_sup = concat_empty(Xcod_ind_sup,disjunctive(X_ind_sup_quali,cols=dummies.columns),axis=1)

            #standardization: z_ik = (x_ik - m_k)/s_k) - m_zk
            Z_ind_sup = ((Xcod_ind_sup - center)/scale) - z_center

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
            split_X_sup_var = splitmix(X_sup_var)
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_var_sup > 0:
                #standardization: z_ik = (x_ik - m_k)/s_k
                Z_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=ind_w))/wstd(X=X_quanti_var_sup,w=ind_w)

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
                #standardization: z_gk = (x_gk - m_k)
                Z_levels_sup = func_groupby(X=Zcod,by=X_quali_var_sup,w=ind_w,func="mean").sub(z_center,axis=1)
                #statistics for supplementary individuals
                levels_sup_ = func_predict(X=Z_levels_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
                #proportion of supplementary levels
                p_k_sup = (disjunctive(X=X_quali_var_sup).T * ind_w).sum(axis=1)
                #vtest of the supplementary levels
                levels_sup_["vtest"] = (levels_sup_["coord"].T * sqrt((n_rows - 1)/((1/p_k_sup) - 1))).T/fit_.svd.vs[:ncp]
                #convert to namedtuple
                self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

                #cordinates for the supplementary qualitative variables - Eta-squared
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