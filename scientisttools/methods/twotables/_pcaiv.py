# -*- coding: utf-8 -*-
from pandas import concat, Series
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..functions.gfa import gFA
from ..functions.model_matrix import model_matrix
from ..functions.wlsreg import wlsreg
from ..functions.utils import check_is_bool
from ..functions.statistics import wmean, wstd, wcorr

class PCAiv(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis with (orthogonal) instrumental variables (PCAiv/PCAoiv)

    Performns Principal Component Analysis with (orthogonal) instrumental variables (PCAiv/PCAoiv).

    Parameters
    ----------
    ncp : int, default = 5
        The number of dimensions kept in the results.

    ortho : bool, default = False
        If ``True``, then the principal component analysis with orthogonal instrumental variables (PCAoiv) is performed.

    Returns
    -------
    call_ : call

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

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

    iv_ : iv, optional
        An object containing all the results for the instrumental variables with the following attributes:

        coord : DataFrame of shape (n_iv, ncp)
            The coordinates for the instrumental variables.
        cos2 : DataFrame of shape (n_iv, ncp)
            The squared cosinus for the instrumental variables.

    row_ : rows
        An object containing all the results for the active rows with the following attributes:

        coord : DataFrame of shape (n_rows, ncp)
            The coordinates of the active rows.
        cos2 : DataFrame of shape (n_rows, ncp)
            Thesquared cosinus of the active rows.
        contrib : DataFrame of shape (n_rows, ncp) 
            The relative contributions of the active rows.
        infos : DataFrame of shape (n_rows, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the active rows.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.
    
    Reference
    ---------
    [1] Chessel, D., Lebreton J. D. and Yoccoz N. (1987) Propriétés de l'analyse canonique des correspondances. Une utilisation en hydrobiologie. Revue de Statistique Appliquée, 35, 55-72.

    [2] Lebreton, J. D., Sabatier, R., Banco G. and Bacou A. M. (1991) Principal component and correspondence analyses with respect to instrumental variables : an overview of their role in studies of structure-activity and species- environment relationships. In J. Devillers and W. Karcher, editors. Applied Multivariate Analysis in SAR and Environmental Studies, Kluwer Academic Publishers, 85-114.

    [3] Obadia, J. (1978) L'analyse en composantes explicatives. Revue de Statistique Appliquee, 24, 5-28.
    
    [4] Rao, C. R. (1964) The use and interpretation of principal component analysis in applied research. Sankhya, A 26, 329-359.

    [5] Sabatier, R., Lebreton J. D. and Chessel D. (1989) Principal component analysis with instrumental variables as a tool for modelling composition data. In R. Coppi and S. Bolasco, editors. Multiway data analysis, Elsevier Science Publishers B.V., North-Holland, 341-352
    
    [6] Ter Braak, C. J. F. (1986) Canonical correspondence analysis : a new eigenvector technique for multivariate direct gradient analysis. Ecology, 67, 1167-1179.

    [7] Ter Braak, C. J. F. (1987) The analysis of vegetation-environment relationships by canonical correspondence analysis. Vegetatio, 69, 69-77.

    See also
    --------
    :class:`scientisttools.save`
        Print results for general factor analysis model in an Excel sheet
    :class:`scientisttools.sprintf`
        Print the analysis results.
    :class:`scientisttools.summary`
        Printing summaries of general factor analysis model.

    Examples
    --------
    >>> from scientisttools.datasets import wine, poison, rpjdl
    >>> from scientisttools import PCA, CA, MCA, FAMD, PCAmix, MPCA, MFA, PCAiv
    >>> # example for PCAiv
    >>> clf1 = PCA()
    >>> clf1.fit(wine.data.iloc[:,2:29])
    >>> clf = PCAiv()
    >>> clf.fit(clf1,wine.data.iloc[:,[0,1,29,30]])
    >>> # example for CAiv
    >>> clf1 = CA()
    >>> clf1.fit(rpjdl.iloc[:,:51])
    """
    def __init__(
         self, ncp=5, ortho=False   
    ):
        self.ncp = ncp
        self.ortho = ortho

    def fit(self,obj,y):
        """
        Fit the model to ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`scientisttools.CA`, :class:`scientisttools.FAMD`, :class:`scientisttools.MCA`, :class:`scientisttools.MPCA`, :class:`scientisttools.PCA`, :class:`scientisttools.PCAmix`, :class:`scientisttools.MFA`.

        y : Series of shape (n_rows,) of DataFrame of shape (n_rows, n_ycolumns)
            Instrumental variables.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if ortho is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.ortho)

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
        #convert to DataFrame if Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if isinstance(y,Series):
            y = y.to_frame()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #factor analysis with (orthogonal) instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.__class__.__name__ == "FAMD":
            Z = obj.call_.Z
        else:
            Z = obj.call_.Zcod
        #recode categorical variable into disjunctive and drop first
        ycod = model_matrix(X=y)
        center, scale = wmean(X=ycod,w=obj.call_.row_w), wstd(X=ycod,w=obj.call_.row_w)
        #standardization
        ys = (ycod - center)/scale
        #separate weighted least squared model
        model = wlsreg(X=ys,Y=Z,w=obj.call_.row_w)
        #set variables
        if self.ortho:
            Z = concat((model[k].resid.to_frame(k)  for k in Z.columns),axis=1)
        else:
            Z = concat((model[k].fittedvalues.to_frame(k) for k in Z.columns),axis=1)
        #make a copy
        tab = Z.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=obj.call_.row_w,col_w=obj.call_.col_w)
        #extract elements
        self.svd_, self.eig_, ncp = fit_.svd, fit_.eig, fit_.ncp

        #coordinates for the columns (G = MVD)
        if obj.__class__.__name__ == "PCAmix":
            fit_.col["coord"] = (fit_.col["coord"].T * obj.call_.col_w).T
        
        #call informations
        call_ = OrderedDict(X=Z,y=y,ycod=ycod,ys=ys,center=center,scale=scale,Z=Z,tab=tab,row_w=obj.call_.row_w,col_w=obj.call_.col_w,ncp=ncp,model=model)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for rows and columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to namedtuple
        self.col_, self.row_ = namedtuple("col",fit_.col.keys())(*fit_.col.values()), namedtuple("row",fit_.row.keys())(*fit_.row.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ortho is False:
            nycod = ycod.shape[1]
            #coordinates for the instrumental variables
            iv_coord = wcorr(X=concat((ycod,self.row_.coord),axis=1),w=obj.call_.row_w).iloc[:nycod,nycod:]
            #convert to ordered dictionary
            iv_ = OrderedDict(coord=iv_coord,cos2=iv_coord**2)
            #convert to namedtuple
            self.iv_ = namedtuple("iv",iv_.keys())(*iv_.values())
        
        return self
    
    def fit_transform(self,obj,y):
        """
        Fit the model with ``obj`` and apply the dimensionality reduction on ``obj``

        Parameters
        ----------
        obj : class
            An object of class :class:`scientisttools.CA`, :class:`scientisttools.FAMD`, :class:`scientisttools.MCA`, :class:`scientisttools.MPCA`, :class:`scientisttools.PCA`, :class:`scientisttools.PCAmix`, :class:`scientisttools.MFA`.

        y : Series of shape (n_rows,) of DataFrame of shape (n_rows, n_ycolumns)
            Instrumental variables.
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Transformed values.
        """
        self.fit(obj=obj,y=y)
        return self.row_.coord