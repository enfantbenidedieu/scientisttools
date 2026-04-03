# -*- coding: utf-8 -*-
from numpy import array,ndarray,sqrt,linalg,ones,diag
from pandas import concat, Series, CategoricalDtype, DataFrame
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ._pca import PCA
from ._mca import MCA
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.model_matrix import model_matrix
from ..functions.wlsreg import wlsreg
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..functions.statistics import wmean, wstd, wcorr, func_groupby
from ..functions.utils import check_is_bool, check_is_dataframe
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class MPCA(BaseEstimator,TransformerMixin):
    """
    Mixed Principal Component Analysis (MPCA)
    
    Performns Mixed Principal Component Analysis (MPCA) and its derivatives with supplementary individuals, supplementary variables (continuous and/or categorical).
    Missing values on continuous variables are replaced by the column mean. Missing values on categorical variables are replaced by the most frequent categories in columns.
    Mixed Principal Component Analysis (MPCA) performs Principal Component Analysis of a set of individuals (observations) described by a mixture of categorical and continuous variables.
    
    Parameters
    ----------
    iv : int, list, tuple or range, default = None
        The indexes or names of the instrumental (explanatory) variables (continuous and/or categorical).

    ortho : bool, default = False
        If ``True``, then the mixed principal component analysis with orthogonal instrumental variables (MPCAoiv) is performed, else 
        mixed principal component analysis with instrumental variables (MPCAiv).

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
        An object containing the summary called parameters with the following attributes:

         Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        dummies : DataFrame of shape (n_rows, n_levels)
            Disjunctive table.
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        Xc : DataFrame of shape (n_rows, n_columns)
            Centered data.
        Zcod : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data (MPCA) or fitted values (MPCAiv) or residuals values (MPCAoiv).
        bary : None or DataFrameof shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        center : Series of shape (n_columns,)
            The weighted average of the columns.
        xc_center : Series of shape (n_columns,)
            The weighted average of centered data.
        xc_scale : Series of shape (n_columns,)
            The weighted standard deviation of centered data.
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
        level_w : Series of shape (n_levels,)
            The weight of the levels.
        col_w : Series of shape (n_columns,)
            The weights of the columns.
        ncp : int
            The number of components kepted.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical)
        z : DataFrame of shape (n_rows, n_iv), optional
            Instrumental variables.
        zcod : DataFrame of shape (n_rows, n_zcod), optional
            Recoded instrumental variables.
        zs : DataFrame of shape (n_rows, n_zcod), optional
            Standardized recoded instrumental variables.
        z_center : Series of shape (n_zcod,), optional
            The weighted average of recoded instrumental variables.
        z_scale : Series of shape (n_zcod,), optional
            The weighted standard deviation of recoded instrumental variables.
        model : OrderedDict, optional
            The separate weighted least square regression between standardized data and standardized recoded instrumental variables.
        y : Series of shape (n_rows,), optional
            The group distribution.
    
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

    iv_ : iv, optional
        An object containing all the results for the instrumental variables with the following attributes:

        coord : DataFrame of shape (n_zcod, ncp)
            The coordinates of the instrumental variables.
        cos2 : DataFrame of shape (n_zcod, ncp)
            The squared cosinus of the instrumental variables.

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
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

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
            self, iv=None, ortho=False, group=None, option="between", ncp=5, row_w=None, col_w=None, ind_sup=None, sup_var=None
    ):
        self.iv = iv
        self.ortho = ortho
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
        #check if ortho is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is not None: 
            check_is_bool(self.ortho)

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
        #get the instrumental variables labels and group labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        iv_label, group_label = get_sup_label(X=X, indexes=self.iv, axis=1), get_sup_label(X=X, indexes=self.group, axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label, sup_var_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0), get_sup_label(X=X, indexes=self.sup_var, axis=1)

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
            if self.iv is not None:
                z_ind_sup, X_ind_sup = X_ind_sup.loc[:,iv_label], X_ind_sup.drop(columns=iv_label)
            if self.group is not None:
                y_ind_sup, X_ind_sup = X_ind_sup[group_label[0]], X_ind_sup.drop(columns=group_label)

        #drop features (partial or instrumental) variables
        if self.iv is not None:
            z, X = X.loc[:,iv_label], X.drop(columns=iv_label)

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
        #center continuous variables
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
        Zcod = (Xc - xc_center)/xc_scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #mixed principal component analysis with (orthogonal) instrumental variables (MPCAiv/MPCAoiv)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Z = Zcod.copy()
        if self.iv is not None:
            #recode categorical variable into disjunctive and drop first
            zcod = model_matrix(X=z)
            z_center, z_scale = wmean(X=zcod,w=ind_w), wstd(X=zcod,w=ind_w)
            #standardization
            zs = (zcod - z_center)/z_scale
            #separate weighted least squared model
            model = wlsreg(X=zs,Y=Zcod,w=ind_w)
            #residuals (MPCAoiv) or fitted values (MPCAiv)
            if self.ortho:
                Z = concat((model[k].resid.to_frame(k)  for k in Zcod.columns),axis=1)
            else:
                Z = concat((model[k].fittedvalues.to_frame(k) for k in Zcod.columns),axis=1)

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
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Xcod=Xcod,Xc=Xc,Zcod=Zcod,Z=Z,bary=bary,tab=tab,center=center,xc_center=xc_center,xc_scale=xc_scale,k1=n_quanti,k2=n_quali,ind_w=ind_w,row_w=row_w,var_w=var_w,
                            levels_w=levels_w,col_w=col_w,ncp=ncp,iv=iv_label,group=group_label,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #add features
        if self.iv is not None:
            call_ = {**call_, **OrderedDict(z=z,zcod=zcod,zs=zs,z_center=z_center,z_scale=z_scale,model=model)}
        #add group distribution
        if self.group is not None:
            call_ = {**call_, **OrderedDict(y=y)}
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
        #statistics for instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is not None and self.ortho is False:
            nzcod = self.call_.zcod.shape[1]
            #coordinates for the instrumental variables
            iv_coord = wcorr(X=concat((self.call_.zcod,self.ind_.coord),axis=1),w=ind_w).iloc[:nzcod,nzcod:]
            #convert to ordered dictionary
            iv_ = OrderedDict(coord=iv_coord,cos2=iv_coord**2)
            #convert to namedtuple
            self.iv_ = namedtuple("iv",iv_.keys())(*iv_.values())

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
            Zcod_ind_sup = ((Xcod_ind_sup - center) - xc_center)/xc_scale

            #mixed principal component analysis with (orthogonal) instrumental variables (MCAiv/MCAoiv)
            Z_ind_sup = Zcod_ind_sup.copy()
            if self.iv is not None:
                #split z_ind_sup
                split_z_ind_sup = splitmix(z_ind_sup)
                #extract elements
                z_ind_sup_quanti_var, z_ind_sup_quali_var, nz_ind_sup_quanti_var, nz_ind_sup_quali_var = split_z_ind_sup.quanti, split_z_ind_sup.quali, split_z_ind_sup.k1, split_z_ind_sup.k2
                #initialization
                zcod_ind_sup = DataFrame(index=ind_sup_label,columns=self.call_.zcod.columns).astype(float)
                #check if numerics variables
                if nz_ind_sup_quanti_var > 0:
                    #replace with numerics columns
                    zcod_ind_sup.loc[:,z_ind_sup_quanti_var.columns] = z_ind_sup_quanti_var
                #check if categorical variables      
                if nz_ind_sup_quali_var > 0:
                    #active categorics
                    categorics = [x for x in self.call_.zcod.columns if x not in self.call_.z.columns]
                    #replace with dummies
                    zcod_ind_sup.loc[:,categorics] = disjunctive(X=z_ind_sup_quali_var,cols=categorics,prefix=True,sep="")
                #standardization: z_ik = (x_ik - m_k)/s_k
                zs_ind_sup = (zs_ind_sup - self.call_.z_center)/self.call_.z_scale
                #insert constant to features
                zs_ind_sup.insert(0,"const",1)

                #predicted values (MCAiv)
                Z_ind_sup = concat((self.call_.model[k].predict(zs_ind_sup).to_frame(k) for k in Zcod_ind_sup.columns),axis=1)
                #residuals (MPCAoiv)
                if self.ortho:
                    Z_ind_sup = Zcod_ind_sup - Z_ind_sup.values

            #within class analysis - suppress within effect
            if self.group is not None and self.option == "within":
                Z_ind_sup = Z_ind_sup - bary.loc[y_ind_sup.values,:].values

            #statistics for supplementary individuals
            ind_sup_ = func_predict(X=Z_ind_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (continuous and/or categorical)
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            #extract elements
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary continuous variables
            if n_quanti_var_sup > 0:
                #standardization : zc_ik = (xc_ik - mc_k)/sc_k
                Zcod_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=ind_w))/wstd(X=X_quanti_var_sup,w=ind_w)

                #mixed principal component analysis with (orthogonal) instrumental variables (MCAiv/MCAoiv)
                Z_quanti_var_sup = Zcod_quanti_var_sup.copy()
                if self.iv:
                    #separate weighted least squared regression
                    model_quanti_var_sup = wlsreg(X=self.call_.zs,Y=Zcod_quanti_var_sup,w=ind_w)
                    #residuals (MPCAoiv) or fitted values (MCAoiv)
                    if self.ortho:
                        Z_quanti_var_sup = concat((model_quanti_var_sup[k].resid.to_frame(k) for k in Zcod_quanti_var_sup.columns),axis=1)
                    else:
                        Z_quanti_var_sup = concat((model_quanti_var_sup[k].fittedvalues.to_frame(k) for k in Zcod_quanti_var_sup.columns),axis=1)
    
                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    Z_quanti_var_sup = bary_quanti_var_sup if self.option == "between" else Z_quanti_var_sup - bary_quanti_var_sup.loc[y.values,:].values

                #statistics for supplementary continuous variables
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
                Zcod_levels_sup = (Xc_levels_sup - wmean(Xc_levels_sup,w=ind_w))/wstd(X=Xc_levels_sup,w=ind_w)

                #mixed principal component analysis with (orthogonal) instrumental variables (MCAiv/MCAoiv)
                Z_levels_sup = Zcod_levels_sup.copy()
                if self.iv:
                    #separate weighted least squared regression
                    model_levels_sup = wlsreg(X=self.call_.zs,Y=Zcod_levels_sup,w=ind_w)
                    #residuals (MPCAoiv) or fitted values (MCAoiv)
                    if self.ortho:
                        Z_levels_sup = concat((model_levels_sup[k].resid.to_frame(k) for k in Zcod_levels_sup.columns),axis=1)
                    else:
                        Z_levels_sup = concat((model_levels_sup[k].fittedvalues.to_frame(k) for k in Zcod_levels_sup.columns),axis=1)

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
        #get elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is not None:
            y, X = X[self.call_.group[0]], X.drop(columns=self.call_.group)
        if self.iv is not None:
            z, X = X.loc[:,self.call_.iv], X.drop(columns=self.call_.iv)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X contains original columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not set(self.call_.X.columns).issubset(X.columns): 
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the {} result".format(self.__class__.__name__))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split data
        split_X = splitmix(X)
        #extract elements
        X_quanti, X_quali = split_X.quanti, split_X.quali
        #disjunctive table for supplementary individuals
        dummies = disjunctive(X=X_quali,cols=self.call_.dummies.columns)
        #concatenate
        Xcod = concat((X_quanti,dummies),axis=1)
        #standardize the data
        Zcod = ((Xcod - self.call_.center) - self.call_.xc_center)/self.call_.xc_scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal component analysis with instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Z = Zcod.copy()
        if self.iv is not None:
            #split z
            split_z = splitmix(z)
            #extract elements
            z_quanti_var, z_quali_var, nz_quanti_var, nz_quali_var = split_z.quanti, split_z.quali, split_z.k1, split_z.k2
            #initialization
            zcod = DataFrame(index=X.index,columns=self.call_.zcod.columns).astype(float)
            #check if numerics variables
            if nz_quanti_var > 0:
                #replace with numerics columns
                zcod.loc[:,z_quanti_var.columns] = z_quanti_var
            #check if categorical variables      
            if nz_quali_var > 0:
                #active categorics
                categorics = [x for x in self.call_.zcod.columns if x not in self.call_.z.columns]
                #replace with dummies
                zcod.loc[:,categorics] = disjunctive(X=z_quali_var,cols=categorics,prefix=True,sep="")
            #standardization: z_ik = (x_ik - m_k)/s_k
            zs = (zcod - self.call_.z_center)/self.call_.z_scale
            #insert constant to features
            zs.insert(0,"const",1)

            #predicted values (MPCAiv)
            Z = concat((self.call_.model[k].predict(zs).to_frame(k) for k in Zcod.columns),axis=1)
            #residuals values (MPCAoiv)
            if self.ortho:
                Z = Zcod - Z.values

        #within class analysis - suppress within effect
        if self.group is not None and self.option == "within":
            Z = Z - self.call_.bary.loc[y.values,:].values
        #coordinates for the new nrows
        coord = (Z * self.call_.col_w).dot(self.svd_.V[:,:self.svd_.ncp])
        coord.columns = self.eig_.index[:self.svd_.ncp]
        return coord