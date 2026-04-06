# -*- coding: utf-8 -*-
from pandas import CategoricalDtype, Series
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..functions.gfa import gFA
from ..functions.gsvd import gSVD
from ..functions.utils import check_is_series
from ..functions.statistics import func_groupby
from ..functions.func_predict import func_predict

class BWCA(BaseEstimator,TransformerMixin):
    """
    Between-Class/within-Class Analysis (BWCA)

    Performns between-class or within-class analysis.

    Parameters
    ----------
    ncp : int, default = 5
        The number of dimensions kept in the results.
    
    option : str, default = "between"
        Which class analysis should be performns.

        - 'between' for between-class analysis.
        - 'within' for within-class analysis.
        
    Returns
    -------
    call_ : call
        An object containing the summary called parameters, with the following attributes:

        X : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        y : Series of shape (n_rows,)
            Classes values.
        bary : None or DataFrame of shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        w : Series of shape (n_rows,) 
            The rows weights.   
        row_w : Series of shape (n_groups,) 
            The groups weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        ncp : int
            The number of components kepted.

    col_ : col
        An object containing all the results for the columns, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the columns.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus of the columns.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the columns.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the columns.
    
    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    group_ : group
        An object containing all the results for the groups, with the following attributes:

        coord : DataFrame of shape (n_groups, ncp)
            The coordinates of the groups.
        cos2 : DataFrame of shape (n_groups, ncp)
            The squared cosinus of the groups.
        contrib : DataFrame of shape (n_groups, ncp), optional
            The relative contributions of the groups.
        dist2 : Series of shape (n_groups,), optional
            The squared distance to origin of the groups.
        infos : DataFrame of shape (n_groups, 4), optional
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the groups.

    ratio_ : float
        The (between-class or within-class) inertia percentage.

    row_ : row
        An object containing all the results for the rows, with the following attributes:

        coord : DataFrame of shape (n_rows, ncp)
            The coordinates of the rows.
        cos2 : DataFrame of shape (n_rows, ncp)
            The squared cosinus of the rows.
        contrib : DataFrame of shape (n_rows, ncp), optional
            The relative contributions of the rows.
        dist2 : Series of shape (n_rows,), optional
            The squared distance to origin of the rows.
        infos : DataFrame of shape (n_rows, 4), optional
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the rows.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_groups, ncp) or (n_rows, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

    References
    ----------
    [1] Bry X., 1996, Analyses factorielles multiples, Economica.

    [2] Dolédec, S. and Chessel, D. (1987) Rythmes saisonniers et composantes stationnelles en milieu aquatique I- Description d'un plan d'observations complet par projection de variables. Acta Oecologica, Oecologia Generalis, 8, 3, 403–426.

    [3] Lebart L., Morineau A. et Warwick K., 1984, Multivariate Descriptive Statistical Analysis, John Wiley and sons, New-York.

    See also
    --------
    :class:`scientisttools.save`
        Print results for general factor analysis model in an Excel sheet
    :class:`scientisttools.sprintf`
        Print the analysis results.
    :class:`scientisttools.summary`
        Printing summaries of general factor analysis model.

    Example
    -------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, BWCA, sprintf, summary, save
    >>> clf1 = CA(ncp=2,row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf1.fit(children.data)
    CA(col_sup=(5,6,7),ncp=2,row_sup=range(14,18),sup_var=8)
    >>> clf = BCWA()
    >>> clf.fit(clf1,children.sup_var["categorie"])
    >>> sprintf(clf)
    >>> summary(clf)
    >>> save(clf)
    """
    def __init__(
            self, ncp = 5, option = "between"
    ):
        self.ncp = ncp
        self.option = option

    def fit(self,obj,y):
        """
        Fit the model to ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`scientisttools.CA`, :class:`scientisttools.FAMD`, :class:`scientisttools.MCA`, :class:`scientisttools.MPCA`, :class:`scientisttools.PCA`, :class:`scientisttools.PCAmix`, :class:`scientisttools.MFA`.

        y : Series of shape (n_rows,)
            Classes values.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(obj)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if obj is object of class CA, FAMD, MCA, MPCA, PCA, PCAmix, MFA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (obj.__class__.__name__ in ("CA","FAMD","MCA","MPCA","PCA","PCAmix","MFA")):
            raise TypeError("'obj' must be an object of class CA, FAMD, MCA, MPCA, PCA, PCAmix, MFA")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if y is an object of class d.Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_series(y)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if option
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not self.option in ("between","within"):
            raise ValueError("'option' should be one of 'between', 'within'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set classes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #unique element in y
        uq_classe = sorted(list(y.unique()))
        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set rows and columns weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X, w, col_w = obj.call_.Z, obj.call_.ind_w if obj.__class__.__name__ != "CA" else obj.call_.row_m, obj.call_.col_w
        row_w, bary = w.copy(), func_groupby(X=X,by=y,func="mean",w=w).loc[uq_classe,:]
        if self.option == "between":
            tab, row_w = bary.copy(), Series([w.loc[y[y==k].index].sum() for k in uq_classe],index=uq_classe,name="weight")
        else:
            tab = X - bary.loc[y.values,:].values

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w)
        #extract elements
        self.svd_, self.eig_, ncp = fit_.svd, fit_.eig, fit_.ncp

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #ratio - percentage of between-class/within-class inertia
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        res_ = gSVD(X=X,ncp=self.ncp,row_w=w,col_w=col_w)
        self.ratio_ = sum(fit_.eig.iloc[:,0])/sum(res_.vs**2)

        #coordinates for the columns (G = MVD)
        if obj.__class__.__name__ == "PCAmix":
            fit_.col["coord"] = (fit_.col["coord"].T * col_w).T
        
        #call informations
        call_ = OrderedDict(X=X,y=y,tab=tab,bary=bary,w=w,row_w=row_w,col_w=col_w,ncp=ncp)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for rows and groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        row_, group_ = func_predict(X=X,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0) if self.option == "between" else fit_.row, func_predict(X=bary,Y=fit_.svd.V[:,:ncp],w=col_w,axis=0) if self.option == "within" else fit_.row

        #convert to namedtuple
        self.group_, self.col_, self.row_ = namedtuple("group",group_.keys())(*group_.values()), namedtuple("col",fit_.col.keys())(*fit_.col.values()), namedtuple("row",row_.keys())(*row_.values())

        return self
    
    def fit_transform(self,obj,y):
        """
        Fit the model with ``obj`` and apply the dimensionality reduction on ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`scientisttools.CA`, :class:`scientisttools.FAMD`, :class:`scientisttools.MCA`, :class:`scientisttools.MPCA`, :class:`scientisttools.PCA`, :class:`scientisttools.PCAmix`, :class:`scientisttools.MFA`.

        y : Series of shape (n_rows,)
            Classes values.
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Transformed values.
        """
        self.fit(obj=obj,y=y)
        return self.row_.coord