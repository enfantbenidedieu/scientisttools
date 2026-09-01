# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, average
from pandas import Series, DataFrame
from collections import namedtuple

# intern functions
from .cov2corr import cov2corr
from .gfa import gFA
from .statistics import wcov, wmean, wstd

def sPCA(X,
         y,
         scale_unit = False, 
         ncp = 5, 
         row_w = None, 
         col_w = None, 
         tol = 1e-7):
    """
    Separate Principal Component Analysis (sPCA)

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data with only continuous variables.

    y : Series of shape (n_samples,)
        The categorical variable which allows for class analysis. It must have the same number of rows as X.

    scale_unit : bool, default = True
        If True, then the data are scaled to unit variance.

    ncp : int, default = 5
        The number of dimensions kept in the results.
    
    row_w : 1d array-like of shape (n_samples,), default = None
        An optional individuals weights.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights.
    
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.
            
    Returns
    -------
    model : dict
        A dictionary containing separate principal component analysis (sPCA).
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if y is an instance of pd.Series class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(y,Series):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.Series.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.Series.html")

    # check if dimensions are equal in term of number of rows
    if X.shape[0] != y.shape[0]:
        raise TypeError("Dimensions of X and y must match in term of number of rows")

    # number of rows and number of columns
    n_rows, n_cols = X.shape

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set individuals and columns weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set individuals weights
    if row_w is None: 
        row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
    elif not isinstance(row_w,(list,tuple,ndarray,Series)): 
        raise TypeError("row_w must be a 1d array-like of individuals weights.")
    elif len(row_w) != n_rows: 
        raise ValueError(f"row_w must be 1d array-like of shape ({n_rows},).")
    else: 
        row_w = Series(array(row_w)/sum(row_w),index=X.index,name="weight")
    
    # set columns weights
    if col_w is None: 
        col_w = Series(ones(n_cols),index=X.columns,name="weight")
    elif not isinstance(col_w,(list,tuple,ndarray,Series)): 
        raise TypeError("col_w must be a 1d array-like of columns weights.")
    elif len(col_w) != n_cols: 
        raise ValueError(f"col_w must be a 1d array-like of shape ({n_cols},).")
    else: 
        col_w = Series(array(col_w),index=X.columns,name="weight")

    # unique element in y
    uq_classe = sorted(y.unique())

    # group index
    group_dict = {k : y[y==k].index for k in uq_classe}
    
    # separate principal component analysis (sPCA)
    model = {}
    for g, r in group_dict.items():
        X_g, row_g = X.loc[r,:], row_w[r]/sum(row_w[r])
        center_g = wmean(X=X_g,w=row_g)
        if scale_unit:
            scale_g = wstd(X=X_g,w=row_g,ddof=1)
        else:
            scale_g = Series(ones(n_cols),index=X_g.columns,name="scale")
        Z_g = (X_g - center_g)/scale_g
        
        # variance-covariance matrix abd bias variance-covariance matrix
        V, Vb = wcov(X=X_g,w=row_g,ddof=0), (Z_g.T*row_g).dot(Z_g)
        # correlation matrix
        R, Rb = cov2corr(V), cov2corr(Vb)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=Z_g,ncp=ncp,row_w=row_g,col_w=col_w,tol=tol)

        # extract elements
        svd, eig = fit_.svd, fit_.eig
        
        # convert to namedtuple
        quanti_var = namedtuple("quanti_var",fit_.col.keys())(*fit_.col.values())
        ind = namedtuple("ind",fit_.row.keys())(*fit_.row.values())
        
        # convert to ordered dictionary
        call_ = {"X":X_g,"Z":Z_g,"center":center_g,"scale":scale_g,"row_w":row_g,"ncp":svd.ncp,"V":V,"Vb":Vb,"R":R,"Rb":Rb}
        # convert to namedtuple
        call = namedtuple("call",call_.keys())(*call_.values())
        # convert to ordered dictionary
        res_ = {"call_":call,"svd_":svd,"eig_":eig,"ind_":ind,"quanti_var_":quanti_var}
        model[g] = namedtuple("PCA",res_.keys())(*res_.values())
    return model