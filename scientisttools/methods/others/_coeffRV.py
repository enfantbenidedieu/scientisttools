# -*- coding: utf-8 -*-
#intern functions
from ._coeffCOI import coeffCOI
from ._coeffLg import coeffLg
from ..functions.cov2corr import cov2corr

def coeffRV(X, 
            group=None, 
            type_group=None, 
            name_group=None, 
            method="mfa", 
            option="lambda1", 
            row_w=None, 
            col_w=None,
            excl=None, 
            tol=1e-7):
    """
    Calculate the RV coefficient between groups
    
    Calculate the RV coefficients between groups.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data

    group : list, tuple
        The number of variables in each group.

    type_group : list, tuple
        The type of variables in each group. Possible values are: 

        * "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (continuous and categorical variables)
        * "f" for frequency (from contingency tables)

    name_group : list, tuple, default = None
        The name of the groups. If ``None``, the group are named Gr1, Gr2 and so on.

    method : str, default = "mfa'
        Method used to compute the trace \emph{RV} coefficients.

        * 'mfa' for \emph{Lg} coefficients in Multiple Factor Analysis (MFA).
        * 'mcoia' for \emph{coinertia} coefficients in Multiple CO-inertia Analysis (MCOIA).

    option : str, default = "lambda1"
        A string for the weightings of the variables.

        * 'inertia': weighting of group :math:`k` by the inverse of the total inertia of the group :math:`k`.
        * 'lambda1': weighting of group :math:`k` by the inverse of the first eigenvalue of the :math:`k`analysis.
        * 'uniform': uniform weighting of groups.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional individuals weights. The weights are given only for the active individuals.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional variables weights. The weights are given only for the active variables.

    excl : None, list, default = None
        The "junk" categories. It can be a list or a tuple of the names of the categories or a list or a tuple of the indexes in the active disjunctive table.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.

    Returns
    -------
    RV : Dataframe of shape (n_groups, n_groups)
        The \emph{RV} coefficients matrix.

    Examples
    --------
    >>> from scientisttools.datasets import wine, poison, friday87, poison
    >>> from scientisttools import coeffRV
    >>> rv_mfa = coeffLg(X=wine.data,group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name)
    >>> rv_coinertia = coeffLg(X=wine.data,group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name,method="mcoia")
    """
    if method == "mfa":
        X = coeffLg(X=X,group=group,type_group=type_group,name_group=name_group,option=option,row_w=row_w,col_w=col_w,excl=excl,tol=tol)
    elif method == "coia":
        X = coeffCOI(X=X,group=group,type_group=type_group,name_group=name_group,option=option,row_w=row_w,col_w=col_w,excl=excl,tol=tol)
    else:
        raise ValueError("Not convenient method")
    return cov2corr(X=X)