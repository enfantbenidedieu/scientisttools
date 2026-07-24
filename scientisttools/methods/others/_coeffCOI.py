# -*- coding: utf-8 -*-
from numpy import ndarray, array, where, ones, nan,inf
from pandas import DataFrame, Series, concat
from itertools import chain, repeat
from collections import OrderedDict

#interns methods
from ..onetable._pca import PCA
from ..onetable._mca import MCA
from ..onetable._famd import FAMD
from ..functions.statistics import wmean
from ..functions.func_coinertia import func_coinertia

def coeffCOI(X,
             group=None,
             type_group=None,
             name_group=None,
             option="lambda1",
             row_w=None,
             col_w=None,
             excl=None,
             tol=1e-7) -> DataFrame:
    """
    Calculate the CO-inertia coefficients between groups
    
    Performs CO-inertia coefficients between groups. Groups of variables can be continuous, categorical or mixed. Missing values on continuous variables are replaced by the column mean. Missing values on categorical variables are replaced by the most frequent categories in columns.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data

    group : list, tuple, default = None
        The number of variables in each group.

    type_group : list, tuple, default = None
        The type of variables in each group. Possible values are: 

        * "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (continuous and categorical variables)
        * "f" for frequency (from contingency tables)

    name_group : list, tuple, default = None
        The name of the groups. If ``None``, the group are named Gr1, Gr2 and so on.

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
    coinertia : Dataframe of shape (n_groups, n_groups)
        The \emph{coinertia} coefficients matrix.

    References
    ----------
    [1] Dolédec, S. and Chessel, D. (1994) Co-inertia analysis: an alternative method for studying species-environment relationships. Freshwater Biology, 31, 277-294.

    [2] Dray, S., Chessel, D. and J. Thioulouse (2003) Co-inertia analysis and the linking of the ecological data tables. Ecology, 84, 11, 3078-3089.

    Examples
    --------
    >>> from scientisttools.datasets import wine, poison, friday87, poison
    >>> from scientisttools import coeffCOI
    >>> coinertia = coeffCOI(X=wine.data,group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name)
    >>> coinertia = coeffCOI(X=poison.data,group=poison.group,type_group=("s","n","n","n"),name_group=poison.name)
    >>> coinertia = coeffCOI(X=friday87.data,group=friday87.group,type_group=("f","f","f","f","f","f","f","f","f","f"),name_group=friday87.name)
    >>> coinertia = coeffCOI(X=poison.data.iloc[:20,:],group=poison.group,type_group=("s","m","n","s"),name_group=poison.name)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if group is None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if group is None:
        raise ValueError("'group' must be assigned.")
    elif not isinstance(group, (list,tuple,ndarray,Series)):
        raise ValueError("'group' must be a 1d array-like with the number of variables in each group")
    else:
        group = [int(x) for x in group]

    #check if group definition
    if sum(group) != X.shape[1]:
        raise TypeError("Not convenient group definition")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if any group has only one /columns
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if any(x == 1 for x in group):
        raise ValueError("groups should have at least two columns")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if type_group in not None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if type_group is None:
        raise ValueError("'type_group' must be assigned")
    elif not isinstance(type_group, (list,tuple,ndarray,Series)): 
        raise ValueError("'type' must be a 1d array-like with the type of variables in each group")
    else:
        type_group = [str(x) for x in type_group]

    if any(x not in ("c","f","m","n","s") for x in type_group):
        raise ValueError("Not convenient type_group definition")
    
    if len(group) != len(type_group):
        raise TypeError("Not convenient group definition")
    
    #which type_group is f
    num_group_freq = list(where(array(type_group) == "f")[0]) if any(x == "f" for x in type_group) else None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #assigned name_group
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if name_group is None:
        name_group = [f"Gr{x+1}" for x in range(len(group))]
    elif not isinstance(name_group,(list,tuple,ndarray,Series)):
        raise TypeError("'name_group' must be a 1d array-like with name of group")
    else:
        name_group = [x for x in name_group]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if option is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (option in ("lambda1","inertia","uniform")):
        raise ValueError("'option' must be one of 'lambda1', 'inertia', 'uniform'")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #assigned group name to label
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    group_dict, k = OrderedDict(), 0
    for i, g in zip(range(len(group)),name_group):
        group_dict[g] = list(X.columns[k:(k+group[i])])
        k += group[i]

    #number of rows/columns
    n_rows, n_cols = X.shape

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set individuals and variables weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set individuals weights
    if row_w is None:
        ind_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
    elif not isinstance(row_w,(list,tuple,ndarray,Series)):
        raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
    elif len(row_w) != n_rows:
        raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
    else:
        ind_w = Series(array(row_w)/sum(row_w),index=X.index,name="weight")

    #set variables weights
    if col_w is None:
        var_w = Series(ones(n_cols),index=X.columns,name="weight")
    elif not isinstance(col_w,(list,tuple,ndarray,Series)):
        raise TypeError("'col_w' must be a 1d array-like of variables weights.")
    elif len(col_w) != n_cols:
        raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
    else:
        var_w = Series(array(col_w),index=X.columns,name="weight")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Data Preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Xcod, total = X.copy(), 0
    if num_group_freq is not None:
        name_group_freq = [g for i, g in enumerate(name_group) if i in num_group_freq]
        
        if len(name_group_freq) > 0:
            group_freq_dict = OrderedDict({k : group_dict[k] for k in name_group_freq})
            freq_cols = list(chain.from_iterable(group_freq_dict.values()))
            #select frequencies data
            N = X.loc[:,freq_cols]
            #sum of all elements 
            total = N.sum(axis=0).sum()
            #proportional table
            P = N.div(total)
            #set global row margin and columns margin
            row_m, col_m = P.sum(axis=1), P.sum(axis=0)
            #construction of recoded table
            for g, cols in group_freq_dict.items():
                #row margin for group
                row_m_g = P[cols].sum(axis=1)
                #normalize such as sum equal to 1
                row_w_g = row_m_g/sum(row_m_g)
                #recoded columns and fill NA, +/-inf if 1e-15
                Xcod[cols] = P[cols].div(col_m[cols],axis=1).sub(row_w_g,axis=0).div(row_m,axis=0).replace([nan,inf,-inf], 1e-15)
            #update weights for rows and columns
            ind_w, var_w[freq_cols] = row_m, col_m

    #rows weights
    row_w = ind_w.copy()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #separate general factor analysis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #separate general factor analysis model for active group
    model = OrderedDict()
    for g, cols in group_dict.items():
        if type_group[name_group.index(g)] in ("c","f","s"):
            scale_unit = False if type_group[name_group.index(g)] in ("c","f") else True
            fa = PCA(scale_unit=scale_unit,ncp=None,row_w=row_w,col_w=var_w[cols],tol=tol)
        if type_group[name_group.index(g)] == "m":
            fa = FAMD(ncp=None,row_w=row_w,col_w=var_w[cols],tol=tol)
        if type_group[name_group.index(g)] == "n":
            fa = MCA(excl=excl,ncp=None,row_w=ind_w,col_w=var_w[cols],tol=tol)
        model[g]= fa.fit(Xcod[cols])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Data prepartion for active groups
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #standardized data
    Zcod = concat((model[g].call_.Z for g in list(group_dict.keys())),axis=1)
    #active columns dictionary
    columns_dict = {g : list(model[g].call_.Z.columns) for g in list(group_dict.keys())}
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set columns weights for multiple factor analysis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #columns weights in each groups 
    mcol_w = concat((model[g].call_.col_w for g in list(group_dict.keys())),axis=0)

    #set groups weights
    if option == "lambda1":
        alpha = Series([1/model[g].eig_.iloc[0,0] for g in list(model.keys())],index=list(model.keys()))
    elif option == "inertia":
        alpha = Series([1/sum(model[g].eig_.iloc[:,0]) for g in list(model.keys())],index=list(model.keys()))
    else:
        alpha = Series(ones(len(list(model.keys()))),index=list(model.keys()))
    
    #set columns weights for multiple factor analysis
    col_w = Series(array([x*y for x,y in zip(mcol_w,array(list(chain(*[repeat(i,k) for i, k in zip(alpha[list(group_dict.keys())],nb_cols)]))))]),index=Zcod.columns,name="weight")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #non-normed principal component analysis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #center according to non-normed PCA
    z_center = wmean(X=Zcod,w=row_w)
    #center
    Z = Zcod - z_center

    #coinertia coefficients
    coinertia = DataFrame(index=name_group,columns=name_group).astype(float)
    for g1, cols1 in columns_dict.items():
        for g2, cols2 in columns_dict.items():
            coinertia.loc[g1,g2] = func_coinertia(X=Z[cols1],Y=Z[cols2],xcol_w=col_w[cols1],ycol_w=col_w[cols2],row_w=row_w)
    return coinertia
