# -*- coding: utf-8 -*-
from numpy import ones, ndarray, array, sqrt,diag, sum, linalg, insert, diff, cumsum, nan, c_
from pandas import Series, DataFrame, concat
from collections import namedtuple
from itertools import repeat
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from ..functions.preprocessing import preprocessing
from ..functions.statistics import wmean, wvar
from ..functions.utils import check_is_bool
from ..functions.gsvd import gSVD

class Procrustes(BaseEstimator,TransformerMixin):
    """
    Procrustean Analysis

    Performns simple procrustean rotation between two sets of points.

    Parameters
    ----------
    scale_norm : bool, default = True
        A boolean indicating whether a transformation by the Gower's scaling (1971) should be applied.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : list, tuple
        The number of variables in each group.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columnw weights. The weights are given only for the active columns.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters, with the following attributes:

        Xtot : DataFrame of shape (n_rows, n_columns)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        X1 : DataFrame of shape (n_rows, n_xcolumns)
            First group data.
        X2 : DataFrame of shape (n_rows, n_ycolumns)
            Second group data.
        Z1 : DataFrame of shape (n_rows, n_xcolumns)
            First group standardized data.
        Z2 : DataFrame of shape (n_rows, n_ycolumns)
            Second group standardized data.
        Z : DataFrame of shape (n_xcolumns, n_ycolumns)
            Input data for GSVD.
        scale_norm : bool
            A boolean indicating whether a transformation by the Gower's scaling (1971) should be applied.
        center : Series of shape (n_columns,)
            Weighted average of X.
        norm  : Series of shape (n_columns,)
            The gower's scaling.
        row_w : Series of shape (n_rows,)
            The rows weights.
        col_w : Series of shape (n_columns,)
            The columns weights
        ncp : int
            The number of components kepted.
        group : list
            The number of variables in each group.
        name_group : list
            The name of the group.

    ind_ : ind
        An object containing the results for the individuals, with the following attributes:

        coord : DataFrame of shape (2*n_rows, ncp)
            The partiel coordinates of the individuals.

    quanti_var_ : quanti_var
        An object containing all the results of the continuous variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the continuous variables.
        alpha : DataFrame of shape (1,2)
            Scaling of target to X and Y.

    rotmat_ : rotmat
        An object containing the rotation matrices, with the following attributes:

        X : DataFrame of shape (n_xcolumns, n_ycolumns)
            The rotation matrix for X to Y.
        
        Y : DataFrame of shape (n_ycolumns, n_xcolumns)
            The rotation matrix for Y to X.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (rank,)
            The singular values.
        d : 1d numpy array of shape (rank,)
            The eigen values.
        U : 2d numpy array of shape (n_xcolumns, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_ycolumns, ncp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.
            
    See Also
    --------
    save : Print results for general factor analysis model in an Excel sheet
    sprintf : Print the analysis results
    summary : Printing summaries of general factor analysis model
            
    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import Procrustes, PCA
    >>> # Procrustean Analysis between two DataFrame
    >>> wine2 = wine.data.iloc[:,10:29]
    >>> clf = Procrustes(scale_norm=True,group=(10,9),ncp=9)
    >>> clf.fit(X=wine2)
    Procrustes(group=(10,9),ncp=9,scale_norm=True)
    
    >>> # Procrustean Analysis between two PCA
    >>> from pandas import concat
    >>> clf1 = PCA()
    >>> clf1.fit(wine2.iloc[:,:10])
    PCA()
    >>> clf2 = PCA()
    >>> clf2.fit(wine2.iloc[:,10:])
    PCA()
    >>> X = concat((clf1.call_.Z,clf2.call_.Z),axis=1)
    >>> clf = Procrustes(scale_norm=True,group=(10,9))
    >>> clf.fit(X)
    Procrustes(group=(10,9),scale_norm=True)
    """
    def __init__(
            self, 
            scale_norm = True, 
            ncp = 5, 
            group = None, 
            name_group = None, 
            row_w = None, 
            col_w = None,
            tol = 1e-7
    ):
        self.scale_norm = scale_norm
        self.ncp = ncp
        self.group = group
        self.name_group = name_group
        self.row_w = row_w
        self.col_w = col_w
        self.tol = tol

    def fit(self,X,y=None):
        """Fit the model to X

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if bool
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.scale_norm)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None: 
            raise ValueError("group must be assigned.")
        elif not isinstance(self.group, (list,tuple)): 
            raise ValueError("group must be a list/tuple with the number of variables in each group")
        elif len(self.group) != 2: 
            raise ValueError("group must be a list/tuple with lenght 2.")
        else: 
            group = [int(x) for x in self.group]

        # check if a group has only one columns
        if group[0] == 1 or group[1] == 1: 
            raise ValueError("groups should have at least two columns")

        # number of samples, columns and variables in X
        n_rows, n_vars, n_xcols = X.shape[0], X.shape[1], group[0]

        # check if number of columns is equal to the sum of sum of e
        if n_vars != sum(group): 
            raise ValueError("Not convenient group definition")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assigned name_group
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.name_group is None:
            name_group = ["X","Y"]
        elif not isinstance(self.name_group,(list,tuple,ndarray,Series)):
            raise TypeError("name_group must be a 1d array-like with names of group")
        elif len(self.name_group) != 2:
            raise ValueError("name_group must be a 1d array-like with lenght 2.")
        else:
            name_group = [f"{x}" for x in self.name_group]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set rows weights
        if self.row_w is None: 
            ind_w = Series(ones(n_rows),index=X.index,name="weights")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("row_w must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w),index=X.index,name="weights")

        # set columns weights
        if self.col_w is None: 
            var_w = Series(ones(n_vars),index=X.columns,name="weights")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)):
            raise TypeError("col_w must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_vars: 
            raise ValueError(f"col_w must be a 1d array-like of shape ({n_vars},).")
        else: 
            var_w = Series(array(self.col_w),index=X.columns,name="weights")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #procrustean analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X between X1 and X2
        X1, X2 = X.iloc[:,:n_xcols], X.iloc[:, n_xcols:]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization of X1 and X2
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #weighted average
        x_center, y_center = wmean(X=X1,w=ind_w), wmean(X=X2,w=ind_w)
        #normed
        x_norm = sqrt((wvar(X=X1,w=ind_w) * n_rows).sum()) if self.scale_norm else 1
        y_norm = sqrt((wvar(X=X2,w=ind_w) * n_rows).sum()) if self.scale_norm else 1
        #standardization
        Z1, Z2 = (X1 - x_center)/x_norm, (X2 - y_center)/y_norm

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #reduce to standard procrustes by whitening both geometric
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set columns weights
        row_w, col_w = var_w.iloc[:n_xcols], var_w.iloc[n_xcols:]
        Z1 = DataFrame(linalg.multi_dot([diag(sqrt(ind_w)), Z1, diag(sqrt(row_w))]),index=Z1.index,columns=Z1.columns)
        Z2 = DataFrame(linalg.multi_dot([diag(sqrt(ind_w)), Z2, diag(sqrt(col_w))]),index=Z2.index,columns=Z2.columns)

        #procruste table : Z = Z_x'Z_y
        Z = Z1.T.dot(Z2)
        #concatenate
        center = concat((x_center,y_center),axis=0)
        norm  = concat((Series(repeat(x_norm,n_xcols),index=X1.columns,name="norm"),
                        Series(repeat(y_norm,n_vars - n_xcols),index=X2.columns,name="norm")),axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #generalized singular values decomposition (GSVD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.svd_ = gSVD(X=Z,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
        #reset number of components
        ncp = self.svd_.ncp

        #call informations
        call_ = {"Xtot": X, "X" : X, "X1": X1, "X2": X2, "Z1": Z1, "Z2": Z2, "Z": Z, "center": center, "norm": norm, 
                 "ind_w": ind_w, "var_w": var_w, "row_w": row_w, "col_w": col_w, "ncp": ncp, 
                 "group": group, "name_group": name_group}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eigvals = self.svd_.d
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],
                              index = [f"Dim{x+1}" for x in range(len(eigvals))])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #ortogonal rotation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        xrotation = DataFrame(linalg.multi_dot([diag(1/sqrt(row_w)),self.svd_.U, self.svd_.V.T,diag(sqrt(col_w))]),index=Z1.columns,columns=Z2.columns)
        yrotation = DataFrame(linalg.multi_dot([diag(1/sqrt(col_w)),self.svd_.V, self.svd_.U.T,diag(sqrt(row_w))]),index=Z2.columns,columns=Z1.columns)
        #convert to ordered dictionary
        self.rotmat_ = namedtuple("rotmat",name_group)(xrotation,yrotation)        

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # rows informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coordinates for the rows on X and Y group
        ind_xcoord, ind_ycoord = Z1.dot(self.svd_.U[:,:ncp]), Z2.dot(self.svd_.V[:,:ncp])
        # set index and columns names
        ind_xcoord.columns, ind_ycoord.columns = self.eig_.index[:ncp], self.eig_.index[:ncp]
        ind_xcoord.index, ind_ycoord.index = [f"{x}.{name_group[0]}" for x in Z1.index], [f"{x}.{name_group[1]}" for x in Z2.index]
        # concatenate
        ind_coord_partiel = concat((ind_xcoord,ind_ycoord),axis=0)
        # convert to ordered dictionary
        ind_ = {"coord_partiel" : ind_coord_partiel}
        # convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #columns informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # scaling of target
        xalpha = sum(self.svd_.vs)/(wvar(X=Z2,w=ind_w)*n_rows).sum() if self.scale_norm else 1
        yalpha = sum(self.svd_.vs)/(wvar(X=Z1,w=ind_w)*n_rows).sum() if self.scale_norm else 1
        # loadings for the columns
        quanti_var_xcoord = DataFrame(self.svd_.U[:,:ncp],columns=self.eig_.index[:ncp],index=Z1.columns)
        quanti_var_ycoord = DataFrame(self.svd_.V[:,:ncp],columns=self.eig_.index[:ncp],index=Z2.columns)
        # concatenate
        alpha = DataFrame([[xalpha,yalpha]],columns=name_group,index=["alpha"])
        # convert to ordered dictionary
        quanti_var_ = {"xcoord": quanti_var_xcoord, "ycoord" : quanti_var_ycoord, "alpha": alpha}
        # convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
   
        return self
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` is the number of rows 
            and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (2*n_rows, ncp)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord_partiel