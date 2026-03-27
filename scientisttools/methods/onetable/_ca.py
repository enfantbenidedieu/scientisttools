# -*- coding: utf-8 -*-
from numpy import sqrt, ones, nan, inf
from scipy.stats import chi2_contingency
from pandas import DataFrame, Series, CategoricalDtype
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd, func_groupby
from ..functions.func_predict import func_predict
from ..functions.func_eta2 import func_eta2
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    
    Performs Correspondence Analysis (CA) and its derivatives with supplementary points (rows and/or columns), supplementary variables (continuous and/or categorical).
    :class:`scientisttools.CA` performns:

        1. Simple Correspondence Analysis (CA)
        2. Detrended Correspondence Analysis (DCA) 
        3. Non-symmetric Correspondence Analysis (nsCA) 
        4. Between-class Correspondence Analysis (bcCA)
        5. Between-class Detrended Correspondence Analysis (bcDCA)
        6. Between-class non-symmetric Correspondence Analysis (bcnsCA)
        7. Within-class Correspondence Analysis (wcCA)
        8. Within-class Detrended Correspondence Analysis (wcDCA)
        9. Within-class non-symmetric Correspondence Analysis (wcnsCA)

    Parameters
    ----------
    symmetric: bool, default = True
        if ``True``, then we performns symmetric Correspondence Analysis (CA, DCA, bcCA, wcCA, bcDCA, wcDCA), else non-symmetric Correspondence Analysis (nsCA, bcnsCA, wcnsCA).
    
    ref : int, str, default = None
        The indexe or name of the reference distribution. Only for Detrended Correspondence Analysis (DCA).

    group : int, str
        The indexe or name of the categorical variable which allows for between-class or within-class analysis.

    option : str, default = "between"
        Which class analysis should be performns.

        - 'between' for between-class analysis.
        - 'within' for within-class analysis.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary rows points.

    col_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary columns points.

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
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        bary : None or DataFrameof shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        total : int
            The sum of elements in ``X``.
        row_s : Series of shape (n_rows,)
            The rows sums.
        col_s : Series of shape (n_columns,)
            The columns sums.
        row_m : Series of shape (n_rows,)
            The rows marging
        col_m : Series of shape (n_columns,)
            The columns marging
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        ncp : int
            The number of components kepted.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ref : None, list
            The name of the reference distribution used for detrended correspondence analysis.
        row_sup : None, list
            The names of the supplementary rows.
        col_sup : None, list
            The names of the supplementary columns.
        sup_var : None, list
            The names of the supplementary variables (quantitative and/or qualitative)
    
    col_ : col
        An object containing all the results for the active columns with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates for the active columns.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus for the active columns.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions for the active columns.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the active columns.

    col_sup_ : col_sup, optional
        An object containing all the results for the supplementary columns with the following attributes:

        coord : DataFrame of shape (n_columns_sup, ncp)
            The coordinates for the supplementary columns.
        cos2 : DataFrame of shape (n_columns_sup, ncp)
            The squared cosinus for the supplementary columns.
        dist2 : Series of shape (n_columns_sup,)
            The squared distance to origin for the supplementary columns.

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

    levels_sup_ : levels_sup
        An object containing all the results for the supplementary levels with the following attributes:

        coord : DataFrame of shape (n_levels_sup, ncp)
            The coordinates for supplementary levels.
        cos2 : DataFrame of shape (n_levels_sup, ncp)
            The squared cosinus for supplementary levels.
        vtest : DataFrame of shape (n_levels_sup, ncp)
            The value-test for supplementary levels.
        dist2 : Series of shape (n_levels_sup,)
            The squared distance to origin for supplementary levels.

    quali_var_sup_ : quali_var_sup
        An object containing all the results for the supplementary qualitative variables with the following attributes:

        coord : DataFrame of shape (n_quali_var_sup, ncp)
            The coordinates for supplementary levels, which is the squared correlation ratio

    quanti_var_sup_ : quanti_var_sup
        An object containing all the results for the supplementary quantitative variables with the following attributes:

        coord : DataFrame of shape (n_quanti_sup, n_columns)
            The coordinates for the supplementary quantitative variables.
        cos2 : DataFrame of shape (n_quanti_sup, n_columns)
            The squared cosinus for the supplementary quantitative variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin for supplementary quantitative variables.

    ratio_ : float, optional
        The inertia (between-class/within-class) percentage.

    row_ : row
        An object containing all the results for the active rows with the following attributes:

        coord : DataFrame of shape (n_rows, ncp) 
            The coordinates for the active rows.
        cos2 : DataFrame of shape (n_rows, ncp)
            The squared cosinus for the active rows.
        contrib : DataFrame of shape (n_rows, ncp), optional
            The relative contributions for the active rows.
        dist2 : Series of shape (n_rows,), optional
            The squared distance to origin for the active rows.
        infos : DataFrame of shape (n_rows, 4), optional
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the active rows.

    row_sup_ : row_sup
        An object containing all the results for the supplementary rows with the following attributes:

        coord : DataFrame of shape (n_rows_sup, ncp)
            The coordinates for the supplementary rows.
        cos2 : DataFrame of shape (n_rows_sup, ncp)
            The squared cosinus for the supplementary rows.
        dist2 : Series of shape (n_rows_sup,)
            The squared distance to origin for the supplementary rows.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, ncp) or (n_groups, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.

    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod
    
    [2] Husson, F., Le, S. and Pages, J. (2009). Analyse de donnees avec R, Presses Universitaires de Rennes.

    [3] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    [4] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle, Dunod, Paris 4ed.

    [5] Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    [5] Rakotomalala R. (2020), `Pratique des méthodes factorielles avec Python <https://hal.science/hal-04868625v1>`_, Université Lumière Lyon 2, Version 1.0

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
    >>> from scientisttools.datasets import housetasks, children, ichtyo, cultural
    >>> from scientisttools import CA
    >>> #with supplementary rows, supplementary columns and supplementary qualitative variables
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    CA(col_sup=(5,6,7),row_sup=range(14,18),sup_var=8)
    >>> #with supplementary rows, supplementary variables (quantitative and qualitative)
    >>> clf = CA(row_sup=range(14,18),sup_var=range(5,9))
    >>> clf.fit(children)
    CA(row_sup=range(14,18),sup_var=range(5,9))
    >>> #detrended correspondence analysys
    >>> clf = CA(ref=9,sup_var=10)
    >>> clf.fit(ichtyo)
    CA(ref=9,sup_var=10)
    >>> Non-symmetric correspondence analysis (nsCA)
    >>> clf = CA(symmetric=False)
    >>> clf.fit(housetasks)
    CA(symmetric=False)
    >>> #between-class correspondence analysis (bcCA)
    >>> clf = CA(group=0,row_sup=range(20,26),col_sup=range(9,12),sup_var=range(12,18))
    >>> clf.fit(cultural)
    CA(col_sup=range(9,12),group=0,row_sup=range(20,26),sup_var=range(12,18))
    >>> #within-class correspondence analysis
    >>> clf = CA(group=0,option="within",row_sup=range(20,26),col_sup=range(9,12),sup_var=range(12,18))
    >>> clf.fit(cultural)
    CA(col_sup=range(9,12),group=0,option="within",row_sup=range(20,26),sup_var=range(12,18))
    """
    def __init__(
            self, symmetric=True, ref=None, group=None, option="between", ncp=5, row_sup=None, col_sup=None, sup_var=None, tol = 1e-7
    ):
        self.symmetric = symmetric
        self.ref = ref
        self.group = group
        self.option = option
        self.ncp = ncp
        self.row_sup = row_sup
        self.col_sup = col_sup
        self.sup_var = sup_var
        self.tol = tol
        
    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns),
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.
            ``X`` is a contingency table containing absolute frequencies.

        y : None
            y is ignored.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is not None and not isinstance(self.group,(int,str)):
            raise TypeError("'group' must be either an objet of type int or str")
        if self.group is not None and not self.option in ("between","within"):
            raise ValueError("'option' should be one of 'between', 'within'")
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #reference validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ref is not None and not isinstance(self.ref,(int,str)):
            raise TypeError("'ref' must be either an objet of type int or str")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_label, ref_label = get_sup_label(X=X, indexes=self.group, axis=1), get_sup_label(X=X, indexes=self.ref, axis=1)
        row_sup_label, col_sup_label, sup_var_label = get_sup_label(X=X, indexes=self.row_sup, axis=0), get_sup_label(X=X, indexes=self.col_sup, axis=1), get_sup_label(X=X, indexes=self.sup_var, axis=1)
        
        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary variables
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
            if self.row_sup is not None: 
                X_sup_var = X_sup_var.drop(index=row_sup_label)

        #drop supplementary columns
        if self.col_sup is not None:
            X_col_sup, X = X.loc[:,col_sup_label],  X.drop(columns=col_sup_label)
            if self.row_sup is not None: 
                X_col_sup = X_col_sup.drop(index=row_sup_label)
        
        #drop supplementary rows
        if self.row_sup is not None: 
            X_row_sup, X = X.loc[row_sup_label,:], X.drop(index=row_sup_label)
            if self.group is not None:
                y_row_sup, X_row_sup = X_row_sup[group_label[0]], X_row_sup.drop(columns=group_label)
            if self.ref is not None:
                X_row_sup = X_row_sup.drop(columns=ref_label)

        #extract reference distribution
        if self.ref is not None:
            z, X = X[ref_label[0]],  X.drop(columns=ref_label)

        #extract group disribution
        if self.group is not None:
            y, X = X[group_label[0]], X.drop(columns=group_label)
            #unique element in y
            uq_classe = sorted(list(y.unique()))
            #convert y to ordered categorical data type
            y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correspondence analysis (CA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if negative entries
        if X[X<0].any().any():
            raise ValueError("negative entries in X")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize the data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of columns and columns sums
        n_cols, col_s = X.shape[1], X.sum(axis=0)

        #rows sums and total
        if self.ref is None: 
            row_s, total = X.sum(axis=1), int(X.sum(axis=0).sum())
        else: 
            row_s, total = z.copy(), int(z.sum())
        row_s.name, col_s.name = "ni.", "n.j"
        if total == 0: 
            raise ValueError("all frequencies are zero")
    
        #columns and rows margins
        col_m, row_m = col_s/total, row_s/total
        row_m.name, col_m.name = "fi.", "f.j"

        #set columns weights and standardization
        if self.symmetric:
            col_w, Z = col_m.copy(), ((X.T/ row_s).T/col_m) - 1
        else:
            col_w, Z = Series(ones(n_cols)/n_cols,index=X.columns), (X.apply(lambda x : x/sum(x) if sum(x)!=0 else col_m,axis=1) - col_m)* n_cols
        
        #fill NA, +/-inf if 1e-15
        Z = Z.replace([nan,inf,-inf], 1e-15)  

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #class analysis (None/between/within)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set tab, row_w and bary
        tab, row_w, bary = Z.copy(), row_m.copy(), None    
        if self.group is not None:
            #update bary
            bary = func_groupby(X=Z,by=y,func="mean",w=row_m).loc[uq_classe,:]
            #update tab and row_w
            if self.option == "between":
                tab, row_w = bary.copy(), Series([row_m.loc[y[y==k].index].sum() for k in uq_classe],index=uq_classe,name="weight")
            else:
                tab, row_w = Z - bary.loc[y.values,:].values, row_m.copy()
        col_w.name, row_w.name = "weight", "weight"
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)

        #extract elements
        self.svd_, self.eig_, self.col_, ncp = fit_.svd, fit_.eig, namedtuple("col",fit_.col.keys())(*fit_.col.values()), fit_.ncp

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,bary=bary,tab=tab,total=total,row_s=row_s,col_s=col_s,row_m=row_m,col_m=col_m,row_w=row_w,col_w=col_w,ncp=ncp,
                            group=group_label,ref=ref_label,row_sup=row_sup_label,col_sup=col_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for rows and/or groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        row_ = fit_.row
        if self.group is not None:
            #ratio - percentage of between-class/within-class inertia
            res_ = gSVD(X=Z,ncp=self.ncp,row_w=row_m,col_w=col_w)
            if self.option == "between":
                group_, row_ = fit_.row, func_predict(X=Z,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            else:
                group_ = func_predict(X=bary,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            self.ratio_, self.group_ = sum(self.eig_.iloc[:,0])/sum(res_.vs**2), namedtuple("group",group_.keys())(*group_.values())
        self.row_ = namedtuple("row",row_.keys())(*row_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            #standardization
            if self.symmetric:
                #frequencies of supplementary rows
                P_row_sup = X_row_sup/total
                #margins for supplementary rows
                row_sup_m = P_row_sup.sum(axis=1)
                #standardization: z_ij = (f_ij/(f_i.*f_.j)) - 1
                Z_row_sup = ((P_row_sup.T / row_sup_m).T/col_m) - 1
            else: 
                Z_row_sup = (X_row_sup.apply(lambda x : x/sum(x) if sum(x)!=0 else col_m,axis=1) - col_m) * n_cols

            #within class analysis - suppress within effect
            if self.group is not None and self.option == "within":
                Z_row_sup = Z_row_sup - bary.loc[y_row_sup.values,:].values

            #fill NA, +/-inf if 1e-15
            Z_row_sup = Z_row_sup.replace([nan,inf,-inf], 1e-15)  
            
            #statistics for supplementary rows
            row_sup_ = func_predict(X=Z_row_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",row_sup_.keys())(*row_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.col_sup is not None:
            #frequencies of supplementary columns
            P_col_sup = X_col_sup/total
            #margins for supplementary columns
            col_sup_m = P_col_sup.sum(axis=0)
            #standardization
            if self.symmetric: 
                # z_ij = (f_ij/(f_i.*f_.j)) - 1
                Z_col_sup = ((P_col_sup.T/row_m).T/col_sup_m) - 1
            else: 
                Z_col_sup = (X_col_sup.apply(lambda x : x/sum(x) if sum(x)!=0 else col_sup_m, axis=1) - col_sup_m) * len(col_sup_label)

            #between/within-class effect
            if self.group is not None:
                bary_col_sup = func_groupby(X=Z_col_sup,by=y,func="mean",w=row_m).loc[uq_classe,:]
                Z_col_sup = bary_col_sup if self.option == "between" else Z_col_sup - bary_col_sup.loc[y.values,:].values
    
            #statistics for supplementary columns
            col_sup_ = func_predict(X=Z_col_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
            #convert to namedtuple
            self.col_sup_ = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_var_sup > 0:
                #standardization: z_ik =  (x_ik - m_k)/s_k
                Z_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=row_m))/wstd(X=X_quanti_var_sup,w=row_m)

                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=row_m).loc[uq_classe,:]
                    Z_quanti_var_sup = bary_quanti_var_sup if self.option == "between" else Z_quanti_var_sup - bary_quanti_var_sup.loc[y.values,:].values

                #statistics for supplementary quantitative variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_var_sup > 0:
                #conditional sum accross levels
                X_levels_sup = func_groupby(X=X,by=X_quali_var_sup,func="sum")

                #standardization
                if self.symmetric:
                    #frequencies of supplementary levels
                    P_levels_sup = X_levels_sup/total
                    #margins for supplementary levels
                    levels_sup_m = P_levels_sup.sum(axis=1)
                    #standardization: z_ij = (f_ij/(f_i.*f_.j)) - 1
                    Z_levels_sup = ((P_levels_sup.T / levels_sup_m).T/col_m) - 1
                else: 
                    Z_levels_sup = (X_levels_sup.apply(lambda x : x/sum(x) if sum(x)!=0 else col_m,axis=1) - col_m) * n_cols

                #statistics for supplementary levels
                levels_sup_ = func_predict(X=Z_levels_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
                #proportion of supplementary levels
                p_k_sup = (disjunctive(X_quali_var_sup).T * row_m).sum(axis=1)
                #vtest for supplementary levels
                levels_sup_["vtest"] = (levels_sup_["coord"].T * sqrt((total - 1) / ((1/p_k_sup) - 1))).T
                #convert to namedtuple
                self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

                #coordinates for the supplementary qualitative variables - Eta-squared
                quali_var_sup_coord = func_eta2(X=self.row_.coord,by=X_quali_var_sup,w=row_m,excl=None)
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
        `X`: DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.
            ``X`` is a contingency table containing absolute frequencies.

        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.row_.coord
    
def statsCA(
        obj
):
    """
    Statistics with Correspondence Analysis

    Performs statistics with correspondence analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    Returns
    -------
    result : statCAResult
        A object with the following attributes:

        goodness_ : goodness
            An object with the following attributes:

            test : DataFrame of shape (2, 3)
                The pearson's chi-squared test and The log-likelihood ratio (i.e the "G-test").    
            association : DataFrame of shape (6, 2)
                The degree of association between two nominal variables ("cramer", "tschuprow", "pearson").

        residual_ : residual
            An object with the following attributes:

            resid : DataFrame of shape (n_rows, n_columns) 
                The model residuals.
            resid_std : DataFrame of shape (n_rows, n_columns) 
                The standardized residuals.
            resid_adj : DataFrame of shape (n_rows, n_columns) 
                The adjusted residuals.
            contrib : DataFrame of shape (n_rows, n_columns) 
                The contribution to chi-squared.
            att_rep_ind : DataFrame of shape (n_rows, n_columns)  
                The attraction repulsion index.

        kaiser_ : DataFrame of shape (1,2)
            The kaiser threshold.

    References
    ----------
    [1] Ricco Rakotomalala (2025), `Etude des dépendances - Variables qualitatives: Tableau de contingence et mesures d'association <https://eric.univ-lyon2.fr/ricco/cours/cours/Dependance_Variables_Qualitatives.pdf>`_, Université Lumière Lyon 2, version 2.2.
    
    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, statsCA
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children.data)
    CA(col_sup=(5,6,7),row_sup=range(14,18),sup_var=8)
    >>> #statistics with correspondence analysis
    >>> stats = statsCA(clf)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class CA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CA":
        raise TypeError("'obj' must be an object of class CA")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #diagnostics tests - multivariate goodness of fit tests
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set dimensions
    n_rows, n_cols = obj.call_.X.shape
    #chi-squared statistics
    chi2_stat, chi2_pvalue, dof, expected_freq = chi2_contingency(obj.call_.X,lambda_=None,correction=False)
    #log - likelihood test (G - test)
    g_stat, g_pvalue = chi2_contingency(obj.call_.X, lambda_="log-likelihood")[:2]
    #convert to DataFrame
    test = DataFrame([[chi2_stat,dof,chi2_pvalue],[g_stat,dof,g_pvalue]],columns=["statistic","dof","pvalue"],index=["Pearson's Chi-Square Test","log-likelihood (G-test)"])
    #association test
    phi2, phi_max, chi2_max = chi2_stat/obj.call_.total, sqrt(min(n_rows - 1, n_cols - 1)), obj.call_.total*min(n_rows - 1, n_cols - 1)
    #Cramer's V
    cramer_v = sqrt(phi2 / min(n_cols - 1, n_rows - 1))
    #Tschuprow's T
    tschuprow_t, tschuprow_max = sqrt(phi2 / sqrt((n_rows - 1) * (n_cols - 1))), (min(n_rows - 1, n_cols - 1)/max(n_rows - 1, n_cols - 1))**(1/4)
    #Pearson's C
    pearson_c, pearson_max = sqrt(phi2 / (1 + phi2)), sqrt((min(n_rows, n_cols) - 1)/min(n_rows, n_cols))
    #Corrected pearson's C
    pearson_c_n = pearson_c/pearson_max
    #convert to pd.DataFrame
    association = DataFrame([[chi2_stat,chi2_max],[sqrt(phi2),phi_max],[cramer_v,1],[tschuprow_t,tschuprow_max],[pearson_c,pearson_max],[pearson_c_n,1]],
                            columns = ["statistic","upper bound"],index = ["Chi-squared","Phi","Cramer's V","Tschuprow's T","Pearson's C","Norm. Pearson's C"])
    #convert to ordered dictionary
    goodness_ = OrderedDict(test=test,association=association)
    #convert to namedtuple
    goodness = namedtuple("goodness",goodness_.keys())(*goodness_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #residuals
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #absolute residuals and attraction repulsion index
    resid, att_rep_ind = obj.call_.X - expected_freq,  obj.call_.X / expected_freq
    #standardized residuals
    resid_std = resid /sqrt(expected_freq)
    #adjusted residuals and chi2 contributions
    resid_adj, chi2_ctr = (resid_std.T / sqrt(1-obj.call_.row_w)).T / sqrt(1-obj.call_.col_w), (resid_std**2)/chi2_stat
    #convert to ordered dictionary
    residuals_ = OrderedDict(resid=resid,resid_std=resid_std,resid_adj=resid_adj,contrib=chi2_ctr,att_rep_ind=att_rep_ind)
    #convert to namedtuple
    residuals = namedtuple("residuals",residuals_.keys())(*residuals_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #compute others indicators 
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #kaiser threshold
    kaiser = DataFrame([[obj.eig_.iloc[:,0].mean(),100/obj.eig_.shape[0]]],columns=["threshold","proportion"],index=["Kaiser"])

    #convert to namedtuple
    return namedtuple("statsCAResult",["goodness","residuals","kaiser"])(goodness,residuals,kaiser)