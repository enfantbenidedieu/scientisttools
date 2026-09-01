# -*- coding: utf-8 -*-
from numpy import ndarray, array,ones, sqrt
from pandas import DataFrame, Series, concat, CategoricalDtype
from collections import namedtuple
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
from ..functions.utils import check_is_bool, check_is_dataframe
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)

    Performs Principal Component Analysis (PCA) and its derivatives with supplementary individuals, supplementary variables (continuous and/or categorical). 
    Missing values on continuous variables are replaced by the column mean. Missing values on categorical variables are replaced by the most frequent categories in columns.
    
    Parameters
    ----------
    scale_unit : bool, default = True
        If True, then the data are scaled to unit variance.

    iv : int, list, tuple or range, default = None
        The indexes or names of the instrumental (explanatory) variables (continuous and/or categorical).

    ortho : bool, default = False
        If True, then the principal component analysis with orthogonal instrumental variables (PCAoiv) is performed.

    partial : int, str, list, tuple or range, default = None
        The indexes or the names of the partial variables (continuous and/or categorical).

    group : int, str, default = None
        The indexe or name of the categorical variable which allows for between-class or within-class analysis.

    option : str, default = "between"
        Which class analysis should be performns.

        * 'between' for between-class analysis.
        * 'within' for within-class analysis.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (continuous and/or categorical).
    
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than `-tol*lambda1` where `lambda1` is the largest eigenvalue.
     
    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters with the following attributes:

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
            The variables standard deviation.
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
        iv : None, list
            The names of the instrumental variables.
        partial : None, list
            The names of the partial variables.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).
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
        z : DataFrame of shape (n_rows, n_partial), optional
            Partial variables.
        y : Series of shape (n_rows,), optional
            The group distribution.

    eig_ : DataFrame of shape (rank, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    ind_ : ind
        An object containing all the results for the active individuals with the following attributes:

        coord : DataFrame of shape (n_rows,ncp)
            The coordinates of the individuals.
        cos2 : DataFrame of shape (n_rows, ncp)
            Thesquared cosinus of the individuals.
        contrib : DataFrame of shape (n_rows, ncp) 
            The relative contributions of the individuals.
        infos : DataFrame of shape (n_rows, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals with the following attributes:

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

    levels_sup_ : levels_sup, optional
        An object containing all the results for the supplementary levels with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the supplementary levels.
        dist2 : Series of shape (n_levels,)
            The squared distance to origin of the supplementary levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the supplementary levels.
        
    quali_var_sup_ : quali_var_sup, optional
        An object containing all the results for the supplementary categorical variables, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary categorical variables. The squared correlation ratio of the supplementary categorical variables, which is the square correlation coefficient between a categorical variable and a dimension

    quanti_var_ : quanti_var
        An object containing all the results for the active variables with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing all the results for the supplementary continuous variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates of the supplementary continuous variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus of the supplementary continuous variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin of the supplementary continuous variables.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
        
        vs : 1d numpy array of shape (rank,)
            The singular values.
        d : 1d numpy array of shape (rank,)
            The eigen values (= square singular values).
        U : 2d numpy array of shape (n_rows, rank) or (n_groups, rank)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, rank)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

    References
    ----------
    [1] Escofier B, Pagès J. `Analyses Factorielles Simples et Multiples <https://cdn-cms.f-static.com/uploads/1460418/normal_5b9ba5dc15394.pdf>`_. 2008, Dunod, Paris 4ed.

    [2] Lebart L., Piron M., & Morineau A. `Statistique exploratoire multidimensionnelle <https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-12/010038111.pdf>`_, 2006, Dunod, Paris 4ed.

    [3] Pagès J. (2013). `Analyse factorielle multiple avec R : Pratique R <https://math.institut-agro-rennes-angers.fr/fr/ouvrages/analyse-factorielle-multiple-avec-r>`_. EDP sciences.

    [4] Ricco Rakotomalala. Pratique des Méthodes Factorielles avec Python. 2020. `hal-04868625 <https://hal.science/hal-04868625>`_
    
    See Also
    --------
    save : Print results for general factor analysis model in an Excel sheet
    sprintf : Print the analysis results
    summary : Printing summaries of general factor analysis model
    
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA
    >>> clf = PCA(ind_sup=range(41,46), sup_var=(10,11,12))
    >>> clf.fit(decathlon)
    PCA(ind_sup=range(41,46), sup_var=(10,11,12))
    """
    def __init__(
            self, 
            scale_unit = True, 
            iv = None, 
            ortho = False, 
            partial = None, 
            group = None, 
            option = "between", 
            ncp = 5, 
            row_w = None, 
            col_w = None, 
            ind_sup = None, 
            sup_var = None, 
            tol = 1e-7
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
        """Fit the model to X

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of rows 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if scale_unit is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.scale_unit)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if ortho is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is not None: 
            check_is_bool(self.ortho)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # group validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is not None:
            if not isinstance(self.group,(int,str)):
                raise TypeError("group must be either an objet of type int or str")
            if not (self.option in ("between","within")):
                raise ValueError("option should be one of 'between', 'within'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get instrumental variables labels, partial variables labels and grouping variable label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        iv_label      = get_sup_label(X=X,indexes=self.iv, axis=1)
        partial_label = get_sup_label(X=X,indexes=self.partial,axis=1)
        group_label   = get_sup_label(X=X,indexes=self.group,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X,indexes=self.ind_sup,axis=0)
        sup_var_label = get_sup_label(X=X,indexes=self.sup_var,axis=1)

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
            if self.iv is not None:
                z_ind_sup, X_ind_sup = X_ind_sup.loc[:,iv_label], X_ind_sup.drop(columns=iv_label)
            if self.partial is not None:
                t_ind_sup, X_ind_sup = X_ind_sup.loc[:,partial_label], X_ind_sup.drop(columns=partial_label)

        #drop instrumental variables
        if self.iv is not None:
            z, X = X.loc[:,iv_label], X.drop(columns=iv_label)
        
        #drop partiel variables
        if self.partial is not None:
            t, X = X.loc[:,partial_label], X.drop(columns=partial_label)
        
        #extract group disribution
        if self.group is not None:
            y, X = X[group_label[0]], X.drop(columns=group_label)
            #unique element in y
            uq_classe = sorted(y.unique())
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
            raise TypeError("row_w must be a 1d array-like of rows weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
        
        #set columns weights
        if self.col_w is None:
            var_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("col_w must be a 1d array-like of columns weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"col_w must be a 1d array-like of shape ({n_cols},).")
        else: 
            var_w = Series(array(self.col_w),index=X.columns,name="weight")

        #compute weighted average and standard deviation of dependent variables
        x_center, x_scale = wmean(X=X,w=ind_w), wstd(X=X,w=ind_w)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # principal components analysis with partial variables (PCApartial)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Xcod = X.copy()
        if self.partial is not None:
            #recode categorical variable into disjunctive and drop first
            tcod = model_matrix(X=t)
            #separate weighted least squared model
            model1 = wlsreg(X=tcod,Y=X,w=ind_w)
            #residuals of models
            Xcod = concat((model1[k].resid.to_frame(k) for k in X.columns),axis=1)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and weighted standard deviation
        center= wmean(X=Xcod,w=ind_w)
        if self.scale_unit:
            scale = wstd(X=Xcod,w=ind_w)
        else:
            scale = Series(ones(n_cols),index=Xcod.columns,name="scale")
        #standardization: z_ik = (x_ik - m_k)/s_k
        Zcod = (Xcod - center)/scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # principal components analysis with (orthogonal) instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Z, col_w = Zcod.copy(), var_w.copy()
        if self.iv is not None:
            #recode categorical variable into disjunctive and drop first
            zcod = model_matrix(X=z)
            #standardized instrumental variables
            z_center, z_scale = wmean(X=zcod,w=ind_w), wstd(X=zcod,w=ind_w)
            zs = (zcod - z_center)/z_scale
            #separate weighted least squared model
            model2 = wlsreg(X=zs,Y=Zcod,w=ind_w)
            #residuals or fitted variables
            if self.ortho:
                Z = concat((model2[k].resid.to_frame(k) for k in Zcod.columns),axis=1) 
            else:
                Z = concat((model2[k].fittedvalues.to_frame(k) for k in Zcod.columns),axis=1)
            #set columns weights
            col_w = Series(ones(Zcod.shape[1]),index=Zcod.columns,name="weight")

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
                tab = bary.copy()
                row_w = Series([ind_w.loc[y[y==k].index].sum() for k in uq_classe],index=uq_classe,name="weight")
            else:
                tab, row_w = Z - bary.loc[y.to_numpy(),:].to_numpy(), ind_w.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)

        #extract elements
        self.svd_, self.eig_= fit_.svd, fit_.eig
        # reset number of components
        ncp = self.svd_.ncp
        # convert to namedtuple and add to model attribute
        self.quanti_var_  = namedtuple("quanti_var",fit_.col.keys())(*fit_.col.values())

        #convert to ordered dictionary - call informations
        call_ = {"Xtot": Xtot, "X": X, "Xcod": Xcod, "Zcod": Zcod, "Z": Z, "bary": bary, "tab": tab, 
                 "x_center": x_center, "x_scale": x_scale, "center": center, "scale": scale, 
                 "ind_w": ind_w, "row_w": row_w, "var_w": var_w, "col_w": col_w, "ncp": ncp,
                 "iv": iv_label, "partial": partial_label, "group": group_label, "ind_sup": ind_sup_label, "sup_var": sup_var_label}
        #add elements
        if self.partial is not None:
            call_ = {**call_, **{"t": t, "tcod": tcod, "model1": model1}}
        if self.iv is not None:
            call_ = {**call_, **{"z": z, "zcod": zcod, "zs": zs, "z_center": z_center, "z_scale": z_scale, "model2": model2}}
        if self.group is not None:
            call_ = {**call_, **{"y": y}}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals and/or groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_ = fit_.row 
        if self.group is not None:
            #ratio - percentage of between-class/within-class inertia
            res_ = gSVD(X=Z,ncp=self.ncp,row_w=ind_w,col_w=col_w,tol=self.tol)
            if self.option == "between":
                group_, ind_ = fit_.row, func_predict(X=Z,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            else:
                group_ = func_predict(X=bary,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            self.ratio_ = sum(self.eig_.iloc[:,0])/sum(res_.d)
            self.group_ = namedtuple("group",group_.keys())(*group_.values())
        
        # convert to namedtuple and add to model attribute
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for partial variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.partial is not None:
            ntod = self.call_.tcod.shape[1]
            #coordinates for the partial variables
            partial_coord = wcorr(X=concat((self.call_.tcod,self.ind_.coord),axis=1),w=ind_w).iloc[:ntod,ntod:]
            #convert to ordered dictionary
            partial_ = {"coord": partial_coord, "cos2": partial_coord**2}
            #convert to namedtuple
            self.partial_ = namedtuple("partial",partial_.keys())(*partial_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.iv is not None and self.ortho is False:
            nzcod = self.call_.zcod.shape[1]
            #coordinates for the instrumental variables
            iv_coord = wcorr(X=concat((self.call_.zcod,self.ind_.coord),axis=1),w=ind_w).iloc[:nzcod,nzcod:]
            #convert to ordered dictionary
            iv_ = {"coord": iv_coord, "cos2": iv_coord**2}
            #convert to namedtuple
            self.iv_ = namedtuple("iv",iv_.keys())(*iv_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #set Xcod for supplementary individuals
            #principal component analysis with partial variables
            Xcod_ind_sup = X_ind_sup.copy()
            if self.partial is not None:
                #split t_ind_sup
                split_t_ind_sup = splitmix(t_ind_sup)
                #extract elements
                t_ind_sup_quanti_var, t_ind_sup_quali_var = split_t_ind_sup.quanti, split_t_ind_sup.quali
                nt_ind_sup_quanti_var, nt_ind_sup_quali_var = split_t_ind_sup.k1, split_t_ind_sup.k2
                
                # initialization
                tcod_ind_sup = DataFrame(index=ind_sup_label,columns=self.call_.tcod.columns).astype("float")
                # check if numerics variables
                if nt_ind_sup_quanti_var > 0:
                    #replace with numerics columns
                    tcod_ind_sup.loc[:,t_ind_sup_quanti_var.columns] = t_ind_sup_quanti_var
                #check if categorical variables      
                if nt_ind_sup_quali_var > 0:
                    #active categorics
                    categorics = [x for x in self.call_.tcod.columns if x not in self.call_.t.columns]
                    #replace with dummies
                    tcod_ind_sup.loc[:,categorics] = disjunctive(X=t_ind_sup_quali_var,cols=categorics,prefix=True,sep="")
                # insert constant to features
                tcod_ind_sup.insert(0,"const",1)
                # predictions
                t_pred_ind_sup = concat((self.call_.model1[k].predict(tcod_ind_sup).to_frame(k) for k in X_ind_sup.columns),axis=1)
                # residuals
                Xcod_ind_sup = X_ind_sup - t_pred_ind_sup.to_numpy()
                
            # standardization: z_ik = (x_ik - m_k)/s_k
            Zcod_ind_sup = (Xcod_ind_sup - self.call_.center)/self.call_.scale

            #principal component analysis with instrumental variables
            Z_ind_sup = Zcod_ind_sup.copy()
            if self.iv is not None:
                #split z_ind_sup
                split_z_ind_sup = splitmix(z_ind_sup)
                #extract elements
                z_ind_sup_quanti_var, z_ind_sup_quali_var= split_z_ind_sup.quanti, split_z_ind_sup.quali
                nz_ind_sup_quanti_var, nz_ind_sup_quali_var  = split_z_ind_sup.k1, split_z_ind_sup.k2
                
                #initialization
                zcod_ind_sup = DataFrame(index=ind_sup_label,columns=self.call_.zcod.columns).astype("float")
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
                zs_ind_sup = (zcod_ind_sup - self.call_.z_center)/self.call_.z_scale
                #insert constant to features
                zs_ind_sup.insert(0,"const",1)

                #predicted values
                Z_ind_sup = concat((self.call_.model2[k].predict(zs_ind_sup).to_frame(k) for k in Zcod_ind_sup.columns),axis=1)
                #residuals for orthogonal instrumental variables
                if self.ortho: 
                    Z_ind_sup = Zcod_ind_sup - Z_ind_sup.to_numpy()

            #within class analysis - suppress within effect
            if self.group is not None and self.option == "within":
                Z_ind_sup = Z_ind_sup - bary.loc[y_ind_sup.to_numpy(),:].to_numpy()

            #statistics for supplementary individuals
            ind_sup_ = func_predict(X=Z_ind_sup,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (continuous and/or categorical)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_var_sup, X_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali
            n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.k1, split_X_sup_var.k2
            
            #statistics for supplementary continuous variables
            if n_quanti_var_sup > 0:
                #set Xcod_quanti_var_sup
                Xcod_quanti_var_sup = X_quanti_var_sup.copy()
                #principal component analysis with partial variables
                if self.partial is not None:
                    #separate weighted least squared model
                    model1_sup = wlsreg(X=self.call_.tcod,Y=X_quanti_var_sup,w=ind_w)
                    #residuals of models
                    Xcod_quanti_var_sup = concat((model1_sup[k].resid.to_frame(k) for k in X_quanti_var_sup.columns),axis=1)

                #compute weighted average for supplementary continuous variables
                center_sup = wmean(X=Xcod_quanti_var_sup,w=ind_w)
                scale_sup = wstd(X=Xcod_quanti_var_sup,w=ind_w) if self.scale_unit else ones(n_quanti_var_sup)
                #standardization: z_ik = (x_ik - m_k)/s_k
                Zcod_quanti_var_sup = (Xcod_quanti_var_sup - center_sup)/scale_sup

                #principal component analysis with instrumental variables
                Z_quanti_var_sup = Zcod_quanti_var_sup.copy()
                if self.iv is not None:
                    #separate weighted least squared model
                    model2_sup = wlsreg(X=self.call_.zs,Y=Zcod_quanti_var_sup,w=ind_w)
                    #residuals or fitted variables
                    if self.ortho:
                        Z_quanti_var_sup = concat((model2_sup[k].resid.to_frame(k) for k in Zcod_quanti_var_sup.columns),axis=1)
                    else:
                        Z_quanti_var_sup = concat((model2_sup[k].fittedvalues.to_frame(k) for k in Zcod_quanti_var_sup.columns),axis=1)

                #within class analysis - suppress within effect
                if self.group is not None:
                    bary_quanti_var_sup = func_groupby(X=Z_quanti_var_sup,by=y,func="mean",w=ind_w).loc[uq_classe,:]
                    if self.option == "between":
                        Z_quanti_var_sup = bary_quanti_var_sup
                    else:
                        Z_quanti_var_sup = Z_quanti_var_sup - bary_quanti_var_sup.loc[y.to_numpy(),:].to_numpy()

                #statistics for supplementary continuous variables
                quanti_var_sup_ = func_predict(X=Z_quanti_var_sup,Y=fit_.svd.U[:,:ncp],w=row_w,axis=1)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary categorical variables/levels
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

                #coordinates for the supplementary categorical variables - Eta-squared
                quali_var_sup_coord = func_eta2(X=self.ind_.coord,by=X_quali_var_sup,w=ind_w,excl=None)
                #convert to ordered dictionary
                quali_var_sup_ = {"coord": quali_var_sup_coord}
                #convert to namedtuple
                self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())
                
        return self
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` is the number of rows 
            and ``n_columns`` is the number of columns.
        
        y : Ignored
            Ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def transform(self,X):
        """
        Apply dimensionality reduction to X

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            New data, where ``n_rows`` is the number of rows 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Projection of X in the first principal components, where ``n_rows`` is the number of rows 
            and ``ncp`` is the number of the components.
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
        if self.partial is not None:
            t, X = X.loc[:,self.call_.partial], X.drop(columns=self.call_.partial)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X contains original columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not set(self.call_.X.columns).issubset(X.columns): 
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the {} result".format(self.__class__.__name__))
        X = X[self.call_.X.columns]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # principal component analysis with partial correlation matrix (PCApartial)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Xcod = X.copy()
        if self.partial is not None:
            #split t
            split_t = splitmix(t)
            #extract elements
            t_quanti_var, t_quali_var = split_t.quanti, split_t.quali
            nt_quanti_var, nt_quali_var = split_t.k1, split_t.k2
            
            #initialization
            tcod = DataFrame(index=X.index,columns=self.call_.tcod.columns).astype("float")
            #check if numerics variables
            if nt_quanti_var > 0:
                #replace with numerics columns
                tcod.loc[:,t_quanti_var.columns] = t_quanti_var
            #check if categorical variables      
            if nt_quali_var > 0:
                #active categorics
                categorics = [x for x in self.call_.tcod.columns if x not in self.call_.t.columns]
                #replace with dummies
                tcod.loc[:,categorics] = disjunctive(X=t_quali_var,cols=categorics,prefix=True,sep="")
            #insert constant to features
            tcod.insert(0,"const",1)
            # predictions
            tpred = concat((self.call_.model1[k].predict(tcod).to_frame(k) for k in X.columns),axis=1)
            #residuals
            Xcod = X - tpred.to_numpy()
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Zcod = (Xcod - self.call_.center)/self.call_.scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal component analysis with instrumental variables (PCAiv/PCAoiv)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Z = Zcod.copy()
        if self.iv is not None:
            #split z
            split_z = splitmix(z)
            #extract elements
            z_quanti_var, z_quali_var = split_z.quanti, split_z.quali
            nz_quanti_var, nz_quali_var = split_z.k1, split_z.k2
            
            #initialization
            zcod = DataFrame(index=X.index,columns=self.call_.zcod.columns).astype("float")
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

            #predicted values
            Z = concat((self.call_.model2[k].predict(zs).to_frame(k) for k in Zcod.columns),axis=1)
            #residuals for orthogonal instrumental variables
            if self.ortho: 
                Z = Zcod - Z.to_numpy()

        #within class analysis - suppress within effect
        if self.group is not None and self.option == "within":
            Z = Z - self.call_.bary.loc[y.to_numpy(),:].to_numpy()
        
        #coordinates for the new nrows
        coord = (Z * self.call_.col_w).dot(self.svd_.V[:,:self.svd_.ncp])
        coord.columns = self.eig_.index[:self.svd_.ncp]
        return coord