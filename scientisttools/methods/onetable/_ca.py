# -*- coding: utf-8 -*-
from numpy import sqrt, ones, nan, inf
from scipy.stats import chi2_contingency
from pandas import DataFrame, Series, CategoricalDtype, concat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.model_matrix import model_matrix
from ..functions.wlsreg import wlsreg
from ..functions.statistics import wmean, wstd, wcorr, func_groupby
from ..functions.func_predict import func_predict
from ..functions.func_eta2 import func_eta2
from ..functions.utils import check_is_bool, check_is_dataframe
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    
    Performs Correspondence Analysis (CA) and its derivatives with supplementary points (rows and/or columns), supplementary variables (continuous and/or categorical).
    
    Parameters
    ----------
    symmetric: bool, default = True
        if ``True``, then we performns symmetric Correspondence Analysis (CA, DCA, bcCA, wcCA, bcDCA, wcDCA), else non-symmetric Correspondence Analysis (nsCA, bcnsCA, wcnsCA).
    
    ref : int, str, default = None
        The indexe or name of the reference distribution. Only for Detrended Correspondence Analysis (DCA).
    
    iv : int, list, tuple or range, default = None
        The indexes or names of the instrumental (explanatory) variables (continuous and/or categorical).

    ortho : bool, default = False
        If ``True``, then the correspondence analysis with orthogonal instrumental variables (CAoiv) is performed.

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
        The indexes or names of the supplementary variables (continuous and/or categorical).

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Zcod : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data (CA) or fitted values (CAiv) or residuals values (CAoiv).
        bary : None or DataFrame of shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        total : int
            The sum of all elements in ``X`` or the sum of all elements in reference distribution.
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
        iv : None, list
            The names of the instrumental variables.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ref : None, list
            The name of the reference distribution used for detrended correspondence analysis.
        row_sup : None, list
            The names of the supplementary rows.
        col_sup : None, list
            The names of the supplementary columns.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical)
        t : Series of shape (n_rows,)
            The reference distribution.
        z : DataFrame of shape (n_rows, n_iv)
            Instrumental variables.
        zcod : DataFrame of shape (n_rows, n_zcod)
            Recoded instrumental variables.
        zs : DataFrame of shape (n_rows, n_zcod)
            Standardized recoded instrumental variables.
        z_center : Series of shape (n_zcod,)
            The weighted average of recoded instrumental variables.
        z_scale : Series of shape (n_zcod,)
            The weighted standard deviation of recoded instrumental variables.
        model : OrderedDict
            The separate weighted least square regression between standardized data and standardized recoded instrumental variables.
        y : Series of shape (n_rows,)
            The group distribution.
    
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

    iv_ : iv, optional
        An object containing all the results for the instrumental variables with the following attributes:

        coord : DataFrame of shape (n_zcod, ncp)
            The coordinates of the instrumental variables.
        cos2 : DataFrame of shape (n_zcod, ncp)
            The squared cosinus of the instrumental variables.

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
        An object containing all the results for the supplementary categorical variables with the following attributes:

        coord : DataFrame of shape (n_quali_var_sup, ncp)
            The coordinates for supplementary levels, which is the squared correlation ratio

    quanti_var_sup_ : quanti_var_sup
        An object containing all the results for the supplementary continuous variables with the following attributes:

        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates for the supplementary continuous variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus for the supplementary continuous variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin for supplementary continuous variables.

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
        U : 2d numpy array of shape (n_rows, maxcp) or (n_groups, maxcp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, maxcp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

    References
    ----------
    [1] Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod
    
    [2] Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle, Dunod, Paris 4ed.

    [3] Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    [4] Rakotomalala R. (2020), `Pratique des méthodes factorielles avec Python <https://hal.science/hal-04868625v1>`_, Université Lumière Lyon 2, Version 1.0

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
    >>> from scientisttools.datasets import housetasks, children, ichtyo, dune, cultural
    >>> from scientisttools import CA
    >>> #with supplementary rows, supplementary columns and supplementary categorical variables
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    CA(col_sup=(5,6,7),row_sup=range(14,18),sup_var=8)
    >>> #with supplementary rows, supplementary variables (continuous and categorical)
    >>> clf = CA(row_sup=range(14,18),sup_var=range(5,9))
    >>> clf.fit(children)
    CA(row_sup=range(14,18),sup_var=range(5,9))
    >>> #detrended correspondence analysis (DCA)
    >>> clf = CA(ref=9,sup_var=10)
    >>> clf.fit(ichtyo)
    CA(ref=9,sup_var=10)
    >>> #non-symmetric correspondence analysis (nsCA)
    >>> clf = CA(symmetric=False)
    >>> clf.fit(housetasks)
    CA(symmetric=False)
    >>> #correspondence analysis with instrumental variables (CAiv)
    >>> clf = CA(iv=range(5))
    >>> clf.fit(dune)
    CA(iv=range(5))
    >>> #correspondence analysis with orthogonal instrumental variables (CAoiv)
    >>> clf = CA(iv=range(5),ortho=True)
    >>> clf.fit(dune)
    CA(iv=range(5),ortho=True)
    >>> #between-class correspondence analysis (bcCA)
    >>> clf = CA(group=0,row_sup=range(20,26),col_sup=range(9,12),sup_var=range(12,18))
    >>> clf.fit(cultural)
    CA(col_sup=range(9,12),group=0,row_sup=range(20,26),sup_var=range(12,18))
    >>> #within-class correspondence analysis (wcCA)
    >>> clf = CA(group=0,option="within",row_sup=range(20,26),col_sup=range(9,12),sup_var=range(12,18))
    >>> clf.fit(cultural)
    CA(col_sup=range(9,12),group=0,option="within",row_sup=range(20,26),sup_var=range(12,18))
    """
    def __init__(
            self, symmetric=True, ref=None, iv=None, ortho=False, group=None, option="between", ncp=5, row_sup=None, col_sup=None, sup_var=None, tol = 1e-7
    ):
        self.symmetric = symmetric
        self.ref = ref
        self.iv = iv
        self.ortho = ortho
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
        #check if symmetric is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.symmetric)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #reference validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ref is not None and not isinstance(self.ref,(int,str)):
            raise TypeError("'ref' must be either an objet of type int or str")

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
        #get the instrumental variables labels, the group labels, and the reference labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        iv_label, group_label, ref_label = get_sup_label(X=X,indexes=self.iv,axis=1), get_sup_label(X=X,indexes=self.group,axis=1), get_sup_label(X=X,indexes=self.ref,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        row_sup_label, col_sup_label, sup_var_label = get_sup_label(X=X,indexes=self.row_sup,axis=0), get_sup_label(X=X,indexes=self.col_sup,axis=1), get_sup_label(X=X,indexes=self.sup_var,axis=1)
        
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
            if self.ref is not None:
                t_row_sup, X_row_sup = X_row_sup[ref_label[0]], X_row_sup.drop(columns=ref_label)
            if self.iv is not None:
                z_row_sup, X_row_sup = X_row_sup.loc[:,iv_label], X_row_sup.drop(columns=iv_label)
            if self.group is not None:
                y_row_sup, X_row_sup = X_row_sup[group_label[0]], X_row_sup.drop(columns=group_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop others elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop instrumental variables
        if self.iv is not None:
            z, X = X.loc[:,iv_label], X.drop(columns=iv_label)

        #extract reference distribution
        if self.ref is not None:
            t, X = X[ref_label[0]],  X.drop(columns=ref_label)

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
            row_s, total = t.copy(), int(t.sum())
        row_s.name, col_s.name = "ni.", "n.j"
        if total == 0: 
            raise ValueError("all frequencies are zero")
    
        #columns and rows margins
        col_m, row_m = col_s/total, row_s/total
        row_m.name, col_m.name = "fi.", "f.j"

        #set columns weights and standardization
        if self.symmetric:
            col_w, Zcod = col_m.copy(), ((X.T/row_s).T/col_m) - 1
        else:
            col_w, Zcod = Series(ones(n_cols)/n_cols,index=X.columns), (X.apply(lambda x : x/sum(x) if sum(x)!=0 else col_m,axis=1) - col_m) * n_cols
        
        #fill NA, +/-inf if 1e-15
        Zcod = Zcod.replace([nan,inf,-inf], 1e-15)  

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correspondance analysis with (orthogonal) instrumental variables (CAiv/CAoiv)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Z = Zcod.copy()
        if self.iv is not None:
            #recode categorical variable into disjunctive and drop first
            zcod = model_matrix(X=z)
            z_center, z_scale = wmean(X=zcod,w=row_m), wstd(X=zcod,w=row_m)
            #standardization
            zs = (zcod - z_center)/z_scale
            #separate weighted least squared model
            model = wlsreg(X=zs,Y=Zcod,w=row_m)
            #residuals (CAoiv) or fitted values (CAiv)
            if self.ortho:
                Z = concat((model[k].resid.to_frame(k) for k in Zcod.columns),axis=1)
            else:
                Z = concat((model[k].fittedvalues.to_frame(k) for k in Zcod.columns),axis=1)
   
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
        call_ = OrderedDict(Xtot=Xtot,X=X,Zcod=Zcod,Z=Z,bary=bary,tab=tab,total=total,row_s=row_s,col_s=col_s,row_m=row_m,col_m=col_m,row_w=row_w,col_w=col_w,ncp=ncp,
                            iv=iv_label,group=group_label,ref=ref_label,row_sup=row_sup_label,col_sup=col_sup_label,sup_var=sup_var_label)
        #add reference distribution
        if self.ref is not None:
            call_ = {**call_, **OrderedDict(t=t)}
        #add instrumental variables informations
        if self.iv is not None:
            call_ = {**call_, **OrderedDict(z=z,zcod=zcod,zs=zs,z_center=z_center,z_scale=z_scale,model=model)}
        #add group distribution
        if self.group is not None:
            call_ = {**call_, **OrderedDict(y=y)}
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
        #statistics for instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is not None and self.ortho is False:
            nzcod = self.call_.zcod.shape[1]
            #coordinates for the instrumental variables
            iv_coord = wcorr(X=concat((self.call_.zcod,self.row_.coord),axis=1),w=row_m).iloc[:nzcod,nzcod:]
            #convert to ordered dictionary
            iv_ = OrderedDict(coord=iv_coord,cos2=iv_coord**2)
            #convert to namedtuple
            self.iv_ = namedtuple("iv",iv_.keys())(*iv_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            #standardization
            if self.symmetric:
                #margins for supplementary rows
                if self.ref is None:
                    row_sup_m = X_row_sup.sum(axis=1)/self.call_.total
                else:
                    row_sup_m = t_row_sup/self.call_.total
                #frequencies of supplementary rows
                P_row_sup = X_row_sup/total
                #standardization: z_ij = (f_ij/(f_i.*f_.j)) - 1
                Zcod_row_sup = ((P_row_sup.T / row_sup_m).T/col_m) - 1
            else: 
                Zcod_row_sup = (X_row_sup.apply(lambda x : x/sum(x) if sum(x)!=0 else col_m,axis=1) - col_m) * n_cols

            #correspondence analysis with (orthogonal) instrumental variables
            Z_row_sup = Zcod_row_sup.copy()
            if self.iv is not None:
                #split z_row_sup
                split_z_row_sup = splitmix(z_row_sup)
                #extract elements
                z_row_sup_quanti_var, z_row_sup_quali_var, nz_row_sup_quanti_var, nz_row_sup_quali_var = split_z_row_sup.quanti, split_z_row_sup.quali, split_z_row_sup.k1, split_z_row_sup.k2
                #initialization
                zcod_row_sup = DataFrame(index=row_sup_label,columns=self.call_.zcod.columns).astype(float)
                #check if numerics variables
                if nz_row_sup_quanti_var > 0:
                    #replace with numerics columns
                    zcod_row_sup.loc[:,z_row_sup_quanti_var.columns] = z_row_sup_quanti_var
                #check if categorical variables      
                if nz_row_sup_quali_var > 0:
                    #active categorics
                    categorics = [x for x in self.call_.zcod.columns if x not in self.call_.z.columns]
                    #replace with dummies
                    zcod_row_sup.loc[:,categorics] = disjunctive(X=z_row_sup_quali_var,cols=categorics,prefix=True,sep="")
                #standardization: z_ik = (x_ik - m_k)/s_k
                zs_row_sup = (zcod_row_sup - self.call_.z_center)/self.call_.z_scale
                #insert constant to features
                zs_row_sup.insert(0,"const",1)
                #predicted values for CAiv
                Z_row_sup = concat((model[k].predict(zs_row_sup).to_frame(k) for k in Zcod_row_sup.columns),axis=1)
                #residuals for CAoiv
                if self.ortho: 
                    Z_row_sup = Zcod_row_sup - Z_row_sup.values
                
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
                Zcod_col_sup = ((P_col_sup.T/row_m).T/col_sup_m) - 1
            else: 
                Zcod_col_sup = (X_col_sup.apply(lambda x : x/sum(x) if sum(x)!=0 else col_sup_m, axis=1) - col_sup_m) * len(col_sup_label)

            #correspondence analysis with instrumental variables
            Z_col_sup = Zcod_col_sup.copy()
            if self.iv is not None:
                #separate weighted least squared model
                model_col_sup = wlsreg(X=self.call_.zs,Y=Z_col_sup,w=row_m)
                #residuals (CAoiv) or fitted variables (CAiv)
                if self.ortho:
                    Z_col_sup = concat((model_col_sup[k].resid.to_frame(k) for k in Zcod_col_sup.columns),axis=1) 
                else:
                    Z_col_sup = concat((model_col_sup[k].fittedvalues.to_frame(k) for k in Zcod_col_sup.columns),axis=1)
            
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

            #statistics for supplementary continuous variables
            if n_quanti_var_sup > 0:
                #standardization: z_ik =  (x_ik - m_k)/s_k
                Zcod_quanti_var_sup = (X_quanti_var_sup - wmean(X=X_quanti_var_sup,w=row_m))/wstd(X=X_quanti_var_sup,w=row_m)

                #correspondence analysis with (orthogonal) instrumental variables
                Z_quanti_var_sup = Zcod_quanti_var_sup.copy()
                if self.iv is not None:
                    #separate weighted least squared model
                    model_quanti_var_sup = wlsreg(X=self.call_.zs,Y=Zcod_quanti_var_sup,w=row_m)
                    #residuals (CAoiv) or fitted variables (CAiv)
                    if self.ortho:
                        Z_quanti_var_sup = concat((model_quanti_var_sup[k].resid.to_frame(k) for k in Zcod_quanti_var_sup.columns),axis=1)
                    else:
                        Z_quanti_var_sup = concat((model_quanti_var_sup[k].fittedvalues.to_frame(k) for k in Zcod_quanti_var_sup.columns),axis=1)

                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=row_m).loc[uq_classe,:]
                    Z_quanti_var_sup = bary_quanti_var_sup if self.option == "between" else Z_quanti_var_sup - bary_quanti_var_sup.loc[y.values,:].values

                #statistics for supplementary continuous variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary categorical variables/levels
            if n_quali_var_sup > 0:
                #conditional sum accross levels
                X_levels_sup = func_groupby(X=X,by=X_quali_var_sup,func="sum")

                #standardization
                if self.symmetric:
                    #margins for supplementary levels
                    if self.ref is not None:
                        levels_sup_m = func_groupby(X=t,by=X_quali_var_sup,func="sum")[ref_label[0]]/self.call_.total
                    else:
                        levels_sup_m = X_levels_sup.sum(axis=1)/self.call_.total
                    #frequencies of supplementary levels
                    P_levels_sup = X_levels_sup/total
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

                #coordinates for the supplementary categorical variables - Eta-squared
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
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.
            ``X`` is a contingency table containing absolute frequencies.

        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Transformed values, where ``n_rows`` is the number of rows and ``ncp`` is the number of the components.
        """
        self.fit(X)
        return self.row_.coord
    
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
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ref is not None:
            if self.call_.ref[0] in X.columns:
                t, X = X[self.call_.ref[0]], X.drop(columns=self.call_.ref)
            else:
                raise ValueError("X must contain reference distribution.")
        if self.group is not None:
            y, X = X[self.call_.group[0]], X.drop(columns=self.call_.group)
        if self.iv is not None:
            z, X = X.loc[:,self.call_.iv], X.drop(columns=self.call_.iv)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X contains original columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not set(self.call_.X.columns).issubset(X.columns): 
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the {} result".format(self.__class__.__name__))
        X = X[self.call_.X.columns]

        #standardization
        if self.symmetric:
            #margins for new rows
            if self.ref is None:
                row_m = X.sum(axis=1)/self.call_.total
            else:
                row_m = t/self.call_.total
            #frequencies of new rows
            P = X/self.call_.total
            #standardization: z_ij = (n_ij/(n_i.*f_.j)) - 1
            Zcod = ((P.T / row_m).T/self.call_.col_m) - 1
        else: 
            Zcod = (X.apply(lambda x : x/sum(x) if sum(x)!=0 else self.call_.col_m,axis=1) - self.call_.col_m) * X.shape[1]

        #correspondence analysis with (orthogonal) instrumental variables
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
            #fitted values for CAiv
            Z = concat((self.call_.model[k].predict(zs).to_frame(k) for k in Zcod.columns),axis=1)
            #residuals for CAoiv
            if self.ortho: 
                Z = Zcod - Z.values
        
        #within class analysis - suppress within effect
        if self.group is not None and self.option == "within":
            Z = Z - self.call_.bary.loc[y.values,:].values

        #fill NA, +/-inf if 1e-15
        Z = Z.replace([nan,inf,-inf], 1e-15) 
        #coordinates for the new nrows
        coord = (Z * self.call_.col_w).dot(self.svd_.V[:,:self.svd_.ncp])
        coord.columns = self.eig_.index[:self.svd_.ncp]
        return coord
            
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