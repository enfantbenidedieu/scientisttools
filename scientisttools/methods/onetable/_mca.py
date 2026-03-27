# -*- coding: utf-8 -*-
from numpy import array, ones, ndarray, c_, cumsum, sqrt
from pandas import CategoricalDtype, DataFrame, Series, concat
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..functions.statistics import wmean, wstd, func_groupby
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)

    Performs Multiple Correspondence Analysis (MCA) or Specific Multiple Correspondence Analysis (speMCA) with supplementary individuals, supplementary variables (quantitative and/or qualitative).
    :class:`scientisttools.MCA` performns:

        1. Multiple Correspondence Analysis (MCA)
        2. Specific Multiple Correspondence Analysis (speMCA) 

    It also performns:

        1. Between/Within - class Multiple Correspondence Analysis (bcMCA/wcMCA)
        2. Between/Within - class Specific Multiple Correspondence Analysis (bcspeMCA/wcspeMCA)

    Parameters
    ----------
    excl : None, list, default = None
        The "junk" categories. It can be a list or a tuple of the names of the categories or a list or a tuple of the indexes in the active disjunctive table.

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

    excl_sup : None, list, default = None
        The supplementary "junk" categories. It can be a list or a tuple of the names of the supplementary categories or a list or a tuple of the indexes in the supplementary disjunctive table.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (quantitative and/or qualitative).

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.
    
    Returns
    -------
    call_ : class
        An object with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        dummies : DataFrame of shape (n_rows, n_levels)
            Disjunctive table.
        Z : DataFrame of shape (n_rows, n_levels)
            Standardized data
        bary : None or DataFrameof shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
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
        ncp : int
            The number of components kepted.
        excl : None, list
            The name of the excluded categories.
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
            The squared cosinus of the individuals.
        contrib : DataFrame of shape (n_rows, ncp)
            The relative contributions of the individuals.
        infos : DataFrame of shape (n_rows, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    ind_sup_ : ind_sup
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_sup, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_rows_sup, ncp)
            The squared cosinus of the supplementary individuals.
        dist2 : Series of shape (n_rows_sup,)
            The squared distance to origin of the supplementary individuals.

    levels_ : levels
        An object containing all the results for the active levels, with the following attributes:
        
        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the levels.
        coord_n : DataFrame of shape (n_levels, ncp)
            The normalized coordinates of the levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the levels.
        contrib : DataFrame of shape (n_levels, ncp)
            The relative contributions of the levels.
        infos : DataFrame of shape (n_levels, 4)
            the additionnal informations (weight, squared distance to origin, inertia and percentage of inertia) of the levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test of the levels.

    levels_sup_ : levels_sup
        An object containing all the results for the supplementary levels, with the following attributes:
        
        coord : DataFrame of shape (n_levels_sup, ncp)
            The coordinates of the supplementary levels.
        coord_n : DataFrame of shape (n_levels_sup, ncp)
            The normalized coordinates of the supplementary levels.
        cos2 : DataFrame of shape (n_levels_sup, ncp)
            The squared cosinus of the supplementary levels.
        vtest : DataFrame of shape (n_levels_sup, ncp)
            The value-test of the supplementary levels.
        dist2 : Series of shape (n_levels_sup,)
            The squared distance to origin of the supplementary levels.

    quali_var_ : quali_var
        An object containing all the results for the active qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var, ncp)
            The coordinates of the qualitative variables, which is eta2, the square correlation corefficient between a qualitative variable and a dimension.
        contrib : DataFrame of shape (n_quali_var, ncp)
            The contributions of the qualitative variables.
        infos : DataFrame of shape (n_quali_var, 3)
            the additionnal informations (weight, inertia and percentage of inertia) of the qualitatve variables.

    quali_var_sup_ : quali_var_sup
        An object containing all the results for the supplementary qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var_sup, ncp)
            The coordinates of the qualitative variables, which is eta2, the square correlation corefficient between a qualitative variable and a dimension.
    
    quanti_var_sup_: quanti_var_sup
        An object containing all the results for the supplementary quantitative variables, with the following attributes:

        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates of the supplementary quantitative variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus of the supplementary quantitative variables.
        dist2 : Series of shape (n_quanti_var_sup, ncp)
            The squared distance to origin of the supplementary quantitative variables.
    
    ratio_ : float, optional
        The inertia (between-class/within-class) percentage.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, ncp) or (n_groups, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.

    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    [2] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    [3] Le Roux B. and Rouanet H., Geometric Data Analysis: From Correspondence Analysis to Stuctured Data Analysis, Kluwer Academic Publishers, Dordrecht (June 2004).

    [4] Le Roux B. and Rouanet H., Multiple Correspondence Analysis, SAGE, Series: Quantitative Applications in the Social Sciences, Volume 163, CA:Thousand Oaks (2010).

    [5] Le Roux B. and Jean C. (2010), Développements récents en analyse des correspondances multiples, Revue MODULARD, Numéro 42

    [6] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    [7] Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

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
    >>> from scientisttools.datasets import poison, tea
    >>> from scientisttools import MCA
    >>> #multiple correspondence analysis (MCA)
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison)
    MCA(sup_var=range(4))
    >>> #specific multiple correspondence analysis (speMCA)
    >>> clf = MCA(excl=(0,2),sup_var = (0,1,13,14))
    >>> clf.fit(poison)
    MCA(excl=(0,2),sup_var = (0,1,13,14))
    >>> #between-class multiple correspondence analysis (bcMCA)
    >>> clf = MCA(group=20,ind_sup=range(200,300),sup_var=[18,19,*list(range(21,tea.shape[1]))])
    >>> clf.fit(tea)
    MCA(group=20,ind_sup=range(200,300),sup_var=[18,19,*list(range(21,tea.shape[1]))])
    >>> #within-class multiple correspondence analysis (wcMCA)
    >>> clf = MCA(group=20,option="within",ind_sup=range(200,300),sup_var=[18,19,*list(range(21,tea.shape[1]))])
    >>> clf.fit(tea)
    MCA(group=20,ind_sup=range(200,300),option="within",sup_var=[18,19,*list(range(21,tea.shape[1]))])
    """
    def __init__(
            self, excl=None, group=None, option="between", ncp=5, row_w=None, col_w=None, excl_sup=None, ind_sup=None, sup_var=None, tol = 1e-7
    ):
        self.excl = excl
        self.group = group
        self.option = option
        self.ncp = ncp
        self.row_w = row_w
        self.col_w = col_w
        self.excl_sup = excl_sup
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
        #group validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is not None and not isinstance(self.group,(int,str)):
            raise TypeError("'group' must be either an objet of type int or str")
        if self.group is not None and not self.option in ("between","within"):
            raise ValueError("'option' should be one of 'between', 'within'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)
        
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
        #multiple correspondence analysis (MCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #disjunctive table
        dummies = disjunctive(X)
        
        #number of rows/columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None: 
            ind_w = Series(ones(n_rows)/n_rows, index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
            
        #set variables weights
        if self.col_w is None: 
            var_w = Series(ones(n_cols)/n_cols,index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
        else: 
            var_w = Series(array(self.col_w)/sum(self.col_w),index=X.columns,name="weight")

        #number of levels, count and proportion
        p_k = (dummies.T * ind_w).sum(axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization : z_ik = (y_ik - p_k)/p_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z_ik = (y_ik/p_k) - 1
        Z = (dummies/p_k) - 1

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get exclusion labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        excl_label = get_sup_label(X=Z,indexes=self.excl,axis=1)

        #get exclusion position
        if self.excl is not None:
            excl_idx = [list(Z.columns).index(x) for x in excl_label]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set columns weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        nb_moda = array([X[j].nunique() for j in X.columns])
        col_w = Series([x*y for x,y in zip(p_k,array(list(chain(*[repeat(i,k) for i, k in zip(var_w,nb_moda)]))))],index=Z.columns,name="weight")

        #replace excluded categories weights by 1e-15
        if self.excl is not None: 
            col_w.iloc[excl_idx] = 1e-15

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
        #fit generalized principal components analysis and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
        #extract elements
        self.svd_, self.eig_, ncp = fit_.svd, fit_.eig, fit_.ncp

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Z=Z,bary=bary,tab=tab,ind_w=ind_w,row_w=row_w,var_w=var_w,col_w=col_w,center=p_k,scale=p_k,ncp=ncp,
                            excl=excl_label,group=group_label,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #replace nan or inf by 1e-15
        if self.excl is not None: 
            self.svd_.V[excl_idx,:] = 1e-15

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
        #Eigenvalues 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.excl is not None:
            lambd = self.eig_.iloc[:,0][self.eig_.iloc[:,0]>(1/n_cols)]
            # Add modified rated and cumulative modified rates
            self.eig_["modified rates"], self.eig_["cumulative modified rates"] = 0.0, 100.0
            pseudo = (n_cols/(n_cols-1)*(lambd-1/n_cols))**2
            self.eig_.iloc[:len(lambd),4] = 100*pseudo/sum(pseudo)
            self.eig_.iloc[:,5] = cumsum(self.eig_.iloc[:,4])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #levels additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        levels_ = fit_.col
        #add proportions
        levels_["infos"].insert(0,"Proportion",p_k)
        #normalized levels coordinates - conditional weighted mean
        levels_["coord_n"] = func_groupby(X=self.ind_.coord,by=X,func="mean",w=ind_w)
        #vtest for the levels
        levels_["vtest"] = (levels_["coord"].T * sqrt((n_rows - 1)/((1/p_k) - 1))).T
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.excl is not None: 
            levels_ = {k : levels_[k].drop(index=excl_label) for k in list(levels_.keys())} 
        #convert to namedtuple
        self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #infos for the qualitative variables
        quali_var_infos = concat((self.levels_.infos.loc[self.levels_.infos.index.isin(list(X[j].unique())),["Inertia","Inertia (%)"]].sum(axis=0).to_frame(j) for j in X.columns),axis=1).T
        quali_var_infos.insert(0,"Weight",var_w)
        #coordinates for the qualitative variables - Eta-squared
        quali_var_coord = func_eta2(X=self.ind_.coord,by=X,w=ind_w,excl=excl_label)
        #contributions for the qualitative variables
        quali_var_ctr = concat((self.levels_.contrib.loc[self.levels_.contrib.index.isin(list(X[j].unique())),:].sum(axis=0).to_frame(j) for j in X.columns),axis=1).T
        #convert to namedtuple
        self.quali_var_ = namedtuple("quali_var",["coord","contrib","infos"])(quali_var_coord,quali_var_ctr,quali_var_infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #disjunctive table
            dummies_ind_sup = disjunctive(X=X_ind_sup,cols=dummies.columns)
            #standardize the data
            Z_ind_sup = (dummies_ind_sup/p_k) - 1

            #within class analysis - suppress within effect
            if self.group is not None and self.option == "within":
                Z_ind_sup = Z_ind_sup - bary.loc[y_ind_sup.values,:].values

            #statistics for supplementary individuals
            ind_sup_ = func_predict(X=Z_ind_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary qualitative variables (qualitative and/or quantitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_var_sup > 0:
                #standardization : z_ik = (x_ik - m_k)/s_k
                Z_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=ind_w))/wstd(X=X_quanti_var_sup,w=ind_w)

                #between/within-class analysis effects
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    Z_quanti_var_sup = bary_quanti_var_sup if self.option == "between" else Z_quanti_var_sup - bary_quanti_var_sup.loc[y.values,:].values

                #statistics for supplementary quantitative variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_var_sup > 0:
                #recode supplementary qualitative variables
                dummies_sup = disjunctive(X_quali_var_sup)
                #get supplementary exclusion labels
                excl_sup_label = get_sup_label(X=dummies_sup,indexes=self.excl_sup,axis=1)

                #proportion of supplementary levels
                p_k_sup = (dummies_sup.T * ind_w).sum(axis=1)
                #standardization : z_ik = (y_ik/p_k) - 1
                Z_levels_sup = (dummies_sup/p_k_sup) - 1

                #between/within-class analysis effects
                if self.group is not None:
                    bary_levels_sup = func_groupby(X=Z_levels_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    Z_levels_sup = bary_levels_sup if self.option == "between" else Z_levels_sup - bary_levels_sup.loc[y.values,:].values
                
                #statistics for supplementary levels
                levels_sup_ = func_predict(X=Z_levels_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #normalized coordinates of the supplementary levels
                levels_sup_["coord_n"] = func_groupby(X=self.ind_.coord,by=X_quali_var_sup,func="mean",w=ind_w)
                #vtest of the supplementary levels
                levels_sup_["vtest"] = (levels_sup_["coord"].T * sqrt((n_rows-1)/((1/p_k_sup) - 1))).T

                #remove supplementary excluded labels
                if self.excl_sup is not None: 
                    levels_sup_ = {k : levels_sup_[k].drop(index=excl_sup_label) for k in list(levels_sup_.keys())} 
                #convert to namedtuple
                self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

                #coordinates of the supplementary qualitative variables - Eta-squared
                quali_var_sup_coord = func_eta2(X=self.ind_.coord,by=X_quali_var_sup,w=ind_w,excl=excl_sup_label)
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
        #fit model
        self.fit(X)
        return self.ind_.coord
    
def statsMCA(
        obj
):
    """
    Statistics with Multiple Correspondence Analysis

    Performs statistics with multiple correspondence analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    Returns
    -------
    result : statMCAResult
        A object with the following attributes

        correction_ : correction
            An object containing eigenvalues correction, with the following attributes:

            benzecri : DataFrame of shape (..., 3)
                The benzecri correction.
            greenacre : DataFrame of shape (..., 3)
                The greenacre correction.

        others_ : others
            An object of others statistics, with the following attributes:

            inertia : float
                The global multiple correspondence analysis inertia.
            kaiser : DataFrame of shape (1,2)
                The kaiser threshold.

    References
    ----------
    [1] Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, statsMCA
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison.data)
    MCA(sup_var=range(4))
    >>> #statistics with multiple correspondence analysis
    >>> stats = statsMCA(clf)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class MCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MCA": 
        raise TypeError("'obj' must be an object of class MCA")

    #set number of qualitative variables and number of levels
    n_cols, n_levels = obj.quali_var_.coord.shape[0], obj.levels_.coord.shape[0]

    #save eigen value grather than threshold
    kaiser_threshold = 1/n_cols
    lambd = obj.eig_.iloc[:,0][obj.eig_.iloc[:,0]>kaiser_threshold]

    #benzecri correction
    lambd_tilde = ((n_cols/(n_cols-1))*(lambd - kaiser_threshold))**2
    s_tilde = 100*(lambd_tilde/sum(lambd_tilde))
    benzecri = DataFrame(c_[lambd_tilde,s_tilde,cumsum(s_tilde)],columns=["Eigenvalue","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])
    #greenacre correction
    s_tilde_tilde = n_cols/(n_cols-1)*(sum(obj.eig_.iloc[:,0]**2)-(n_levels - n_cols)/(n_cols**2))
    tau = 100*(lambd_tilde/s_tilde_tilde)
    greenacre = DataFrame(c_[lambd_tilde,tau,cumsum(tau)],columns=["Eigenvalue","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])
    #convert to namedtuple
    correction_ = namedtuple("correction",["benzecri","greenacre"])(benzecri,greenacre)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #multiple correspondence analysis additionals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #inertia
    inertia = (n_levels/n_cols) - 1
    #eigenvalue threshold
    kaiser_proportion_threshold = 100/inertia
    #eigen value threshold
    kaiser = DataFrame([[kaiser_threshold,kaiser_proportion_threshold]],columns=["threshold","proportion"],index=["Kaiser critical values"])
    #convert to namedtuple
    others_ = namedtuple("others",["inertia","kaiser"])(inertia,kaiser)
    return namedtuple("statsMCAResult",["correction_","others_"])(correction_,others_)