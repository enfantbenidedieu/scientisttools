# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from typing import Hashable
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
from functools import partial
import itertools
import scipy.stats as st

def get_melt(X,level=- 1, dropna=True):
    """Stack the prescribed level(s) from columns to index.

    Return a reshaped DataFrame or Series having a multi-level index with one or more 
    new inner-most levels compared to the current DataFrame. The new inner-most levels 
    are created by pivoting the columns of the current dataframe:

    Parameters
    ----------
    X       : DataFrame
    level   : int, str, list, default -1
            Level(s) to stack from the column axis onto the index axis, 
            defined as one index or label, or a list of indices or labels.
    dropna  : bool, default True
            Whether to drop rows in the resulting Frame/Series with missing values. 
            Stacking a column level onto the index axis can create combinations of index 
            and column values that are missing from the original dataframe.

    Return
    ------
        Stacked dataframe or series.
    
    """
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
                f"{type(X)} is not supported. Please convert to a DataFrame with "
                "pd.DataFrame. For more information see: "
                "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    return X.stack(level=level, dropna=dropna).rename_axis(('Var1', 'Var2')).reset_index(name='value')

def match_arg(x, lst):
    return [elt for elt in lst if x in elt]


def is_euclidean(D,plot=False,printf=False,tol=1e-07):
    """Confirmation of the Euclidean nature of a distance matrix by the Gower's theorem.

    Parameters
    ----------
    D  : Dissimilarities matrix
    plot : a logical value indicating whether the eigenvalues bar plot of the matrix of the term -(1/2)*d(i,j)^2 centred by rows and columns should be diplayed.
    print : a logical value indicating whether the eigenvalues of the matrix of the term -(1/2)*d(i,j)^2 centred by rows and columns should be printed.
    tol : a tolerance threshold : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.

    Return
    ------
    None

    References
    ----------
    Gower, J.C. and Legendre, P. (1986) Metric and Euclidean properties of dissimilarity coefficients. Journal of Classification, 3, 5--48.
    
    """
    D = np.array(D)
    n = D.shape[0]
    A = -0.5*np.multiply(D,D)
    H = np.identity(n) - (1/n)*np.ones(shape=(n,n))
    B = np.dot(H,np.dot(A,H))
    value, vector = np.linalg.eig(B)
    w0 = value[-1]/value[0]
    if plot:
        plt.bar(np.arange(len(value)),value)
        plt.show()
    if printf:
        print(value)
    return w0 > - tol

def sim_dist(X, method= "standard"):
    """Transforms similarities matrix to dissimilarities matrix

    Parameters
    ----------
    X : array of float, square matrix.
    method : {'standard','oneminus'} 

    Return
    ------
    D : Dissimilarities matrix
    
    """
    if X.shape[0] != X.shape[1]:
        raise ValueError("Error : 'X' must be square matrix")
    if method == "standard":
        D = np.zeros(shape=(X.shape[0],X.shape[0]))
        for i in np.arange(0,X.shape[0]):
            for j in np.arange(0,X.shape[0]):
                D[i,j] = np.sqrt((X[i,i] - X[j,j] +2*X[i,j]))
    elif method == "oneminus":
        D = 1 - X
    else:
        raise ValueError("Error : Allowed method are 'standard' or 'oneminus'.")
    return D


def bicenter_wt(X, row_wt=None,col_wt=None):
    """This function creates a doubly centred matrix.

    Parameters
    ----------
    X : a matrix with n rows and p columns

    Return
    ------
    A doubly centred matrix
    """
    X = np.array(X)
    n, p = X.shape
    if row_wt is None:
        row_wt = np.repeat(a=1,repeats=n)
    if col_wt is None:
        col_wt = np.repeat(a=1,repeats=p)
    row_wt = np.array(row_wt)
    col_wt = np.array(col_wt)
    sr = sum(row_wt)
    row_wt = row_wt/sr
    st = sum(col_wt)
    col_wt = col_wt/st
    row_mean = np.apply_along_axis(func1d=np.sum,axis=0,arr=np.apply_along_axis(arr=X,func1d=lambda x : x*row_wt,axis=0))
    col_mean = np.apply_along_axis(func1d=np.sum,axis=0,arr=np.apply_along_axis(arr=np.transpose(X),func1d=lambda x : x*col_wt,axis=0))
    col_mean = col_mean - np.sum(row_mean * col_wt)
    X = np.apply_along_axis(func1d=lambda x : x - row_mean,axis=1,arr=X)
    X = np.transpose(np.apply_along_axis(func1d=lambda x : x - col_mean,axis=1,arr=np.transpose(X)))
    return X

# Rapport de corrélation
def eta2(categories, value,digits=4):
    K = len(np.unique(categories, return_inverse=True)[0])
    n = value.shape[0]
    
    cat = np.unique(categories, return_inverse=True)[1]
    values = np.array(value)
    
    scintra = 0
    scinter = 0
    for i in np.unique(cat):
        subgroup = values[np.argwhere(cat == i).flatten()]
        scintra += np.sum((subgroup-np.mean(subgroup))**2)
        scinter += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    eta2 = scinter/(scinter+scintra)
    f_stat = (scinter/(K-1))/(scintra/(n-K))
    # calcul de la pvalue
    from scipy.stats import f
    pvalue = np.round(f.sf(f_stat, K-1, n-K),digits)
    return {'Sum. Intra':round(scintra,digits),
            'Sum. Inter':round(scinter,digits),
            'correlation ratio':round(eta2,digits),
            'F-stats': round(f_stat,digits),
            'pvalue': pvalue}

def RGBtoHex(vals, rgbtype=256):
  """Converts RGB values in a variety of formats to Hex values.

     @param  vals     An RGB/RGBA tuple
     @param  rgbtype  Valid valus are:
                          1 - Inputs are in the range 0 to 1
                        256 - Inputs are in the range 0 to 255

     @return A hex string in the form '#RRGGBB' or '#RRGGBBAA'
"""

  if len(vals)!=3 and len(vals)!=4:
    raise Exception("RGB or RGBA inputs to RGBtoHex must have three or four elements!")
  if rgbtype!=1 and rgbtype!=256:
    raise Exception("rgbtype must be 1 or 256!")

  #Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
  if rgbtype==1:
    vals = [255*x for x in vals]

  #Ensure values are rounded integers, convert to hex, and concatenate
  return '#' + ''.join(['{:02X}'.format(int(round(x))) for x in vals])


# https://stackoverflow.com/questions/34693991/repel-annotations-in-matplotlib

def repel_labels(ax,x, y, labels, k=0.01):
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.iteritems():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3", 
                                    color='red'), )
    # expand limits
    all_pos = np.vstack(pos.values())
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos-x_span*0.15, 0)
    maxs = np.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])


def from_dummies(
    data: DataFrame,
    sep: None | str = None,
    default_category: None | Hashable | dict[str, Hashable] = None,
) -> DataFrame:
    """
    Create a categorical ``DataFrame`` from a ``DataFrame`` of dummy variables.
    Inverts the operation performed by :func:`~pandas.get_dummies`.
    .. versionadded:: 1.5.0
    Parameters
    ----------
    data : DataFrame
        Data which contains dummy-coded variables in form of integer columns of
        1's and 0's.
    sep : str, default None
        Separator used in the column names of the dummy categories they are
        character indicating the separation of the categorical names from the prefixes.
        For example, if your column names are 'prefix_A' and 'prefix_B',
        you can strip the underscore by specifying sep='_'.
    default_category : None, Hashable or dict of Hashables, default None
        The default category is the implied category when a value has none of the
        listed categories specified with a one, i.e. if all dummies in a row are
        zero. Can be a single value for all variables or a dict directly mapping
        the default categories to a prefix of a variable.
    Returns
    -------
    DataFrame
        Categorical data decoded from the dummy input-data.
    Raises
    ------
    ValueError
        * When the input ``DataFrame`` ``data`` contains NA values.
        * When the input ``DataFrame`` ``data`` contains column names with separators
          that do not match the separator specified with ``sep``.
        * When a ``dict`` passed to ``default_category`` does not include an implied
          category for each prefix.
        * When a value in ``data`` has more than one category assigned to it.
        * When ``default_category=None`` and a value in ``data`` has no category
          assigned to it.
    TypeError
        * When the input ``data`` is not of type ``DataFrame``.
        * When the input ``DataFrame`` ``data`` contains non-dummy data.
        * When the passed ``sep`` is of a wrong data type.
        * When the passed ``default_category`` is of a wrong data type.
    """
    from pandas.core.reshape.concat import concat

    if not isinstance(data, DataFrame):
        raise TypeError(
            "Expected 'data' to be a 'DataFrame'; "
            f"Received 'data' of type: {type(data).__name__}"
        )

    if data.isna().any().any():
        raise ValueError(
            "Dummy DataFrame contains NA value in column: "
            f"'{data.isna().any().idxmax()}'"
        )

    # index data with a list of all columns that are dummies
    try:
        data_to_decode = data.astype("boolean", copy=False)
    except TypeError:
        raise TypeError("Passed DataFrame contains non-dummy data")

    # collect prefixes and get lists to slice data for each prefix
    variables_slice = defaultdict(list)
    if sep is None:
        variables_slice[""] = list(data.columns)
    elif isinstance(sep, str):
        for col in data_to_decode.columns:
            prefix = col.split(sep)[0]
            if len(prefix) == len(col):
                raise ValueError(f"Separator not specified for column: {col}")
            variables_slice[prefix].append(col)
    else:
        raise TypeError(
            "Expected 'sep' to be of type 'str' or 'None'; "
            f"Received 'sep' of type: {type(sep).__name__}"
        )

    if default_category is not None:
        if isinstance(default_category, dict):
            if not len(default_category) == len(variables_slice):
                len_msg = (
                    f"Length of 'default_category' ({len(default_category)}) "
                    f"did not match the length of the columns being encoded "
                    f"({len(variables_slice)})"
                )
                raise ValueError(len_msg)
        elif isinstance(default_category, Hashable):
            default_category = dict(
                zip(variables_slice, [default_category] * len(variables_slice))
            )
        else:
            raise TypeError(
                "Expected 'default_category' to be of type "
                "'None', 'Hashable', or 'dict'; "
                "Received 'default_category' of type: "
                f"{type(default_category).__name__}"
            )

    cat_data = {}
    for prefix, prefix_slice in variables_slice.items():
        if sep is None:
            cats = prefix_slice.copy()
        else:
            cats = [col[len(prefix + sep) :] for col in prefix_slice]
        assigned = data_to_decode.loc[:, prefix_slice].sum(axis=1)
        if any(assigned > 1):
            raise ValueError(
                "Dummy DataFrame contains multi-assignment(s); "
                f"First instance in row: {assigned.idxmax()}"
            )
        if any(assigned == 0):
            if isinstance(default_category, dict):
                cats.append(default_category[prefix])
            else:
                raise ValueError(
                    "Dummy DataFrame contains unassigned value(s); "
                    f"First instance in row: {assigned.idxmin()}"
                )
            data_slice = concat(
                (data_to_decode.loc[:, prefix_slice], assigned == 0), axis=1
            )
        else:
            data_slice = data_to_decode.loc[:, prefix_slice]
        cats_array = np.array(cats, dtype="object")
        # get indices of True entries along axis=1
        cat_data[prefix] = cats_array[data_slice.to_numpy().nonzero()[1]]

    return DataFrame(cat_data)

def check_array_with_weights(X, weights, **kwargs):
    """Utility to validate data and weights.
    This calls check_array on X and weights, making sure results match.
    """
    if weights is None:
        return check_array(X, **kwargs), weights

    # Always use copy=False for weights
    kwargs_weights = dict(kwargs)
    kwargs_weights.update(copy=False)
    weights = check_array(weights, **kwargs_weights)

    # Always use force_all_finite=False for X
    kwargs_X = dict(kwargs)
    kwargs_X.update(force_all_finite=False)
    X = check_array(X, **kwargs_X)

    # Make sure shapes match and missing data has weights=0
    if X.shape != weights.shape:
        raise ValueError("Shape of `X` and `weights` should match")

    Wzero = (weights == 0)
    X[Wzero] = 0

    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or infinity without "
                         "a corresponding zero in `weights`.")
    return X, weights


def orthonormalize(X, rows=True):
    """Orthonormalize X using QR-decomposition
    Parameters
    ----------
    X : array-like, [N, M]
        matrix to be orthonormalized
    rows : boolean (default=True)
        If True, orthonormalize rows of X. Otherwise orthonormalize columns.
    Returns
    -------
    Y : ndarray, [N, M]
        Orthonormalized version of X
    """
    orient = lambda X: X.T if rows else X
    Q, R = np.linalg.qr(orient(X))
    return orient(Q)


def random_orthonormal(N, M, rows=True, random_state=None):
    """Construct a random orthonormal matrix
    Parameters
    ----------
    N, M : integers
        The size of the matrix to construct.
    rows : boolean, default=True
        If True, return matrix with orthonormal rows.
        Otherwise return matrix with orthonormal columns.
    random_state : int or None
        Specify the random state used in construction of the matrix.
    """
    assert N <= M if rows else N >= M
    rand = np.random.RandomState(random_state)
    return orthonormalize(rand.randn(N, M), rows=rows)


def solve_weighted(A, b, w):
    """solve Ax = b with weights w
    Parameters
    ----------
    A : array-like [N, M]
    b : array-like [N]
    w : array-like [N]
    Returns
    -------
    x : ndarray, [M]
    """
    A, b, w = map(np.asarray, (A, b, w))
    ATw2 = A.T * w ** 2
    return np.linalg.solve(np.dot(ATw2, A),
                           np.dot(ATw2, b))


def weighted_mean(x, w=None, axis=None):
    """Compute the weighted mean along the given axis
    The result is equivalent to (x * w).sum(axis) / w.sum(axis),
    but large temporary arrays are not created.
    Parameters
    ----------
    x : array_like
        data for which mean is computed
    w : array_like (optional)
        weights corresponding to each data point. If supplied, it must be the
        same shape as x
    axis : int or None (optional)
        axis along which mean should be computed
    Returns
    -------
    mean : np.ndarray
        array representing the weighted mean along the given axis
    """
    if w is None:
        return np.mean(x, axis)

    x = np.asarray(x)
    w = np.asarray(w)

    if x.shape != w.shape:
        raise NotImplementedError("Broadcasting is not implemented: "
                                  "x and w must be the same shape.")

    if axis is None:
        wx_sum = np.einsum('i,i', np.ravel(x), np.ravel(w))
    else:
        try:
            axis = tuple(axis)
        except TypeError:
            axis = (axis,)

        if len(axis) != len(set(axis)):
            raise ValueError("duplicate value in 'axis'")

        trans = sorted(set(range(x.ndim)).difference(axis)) + list(axis)
        operand = "...{0},...{0}".format(''.join(chr(ord('i') + i)
                                                 for i in range(len(axis))))
        wx_sum = np.einsum(operand,
                           np.transpose(x, trans),
                           np.transpose(w, trans))

    return wx_sum / np.sum(w, axis)

# Test de sphéricité de Bartlett
def BartlettSphericityTest(x):
  test = calculate_bartlett_sphericity(x.values)
  return pd.DataFrame(test,index=["stats","pvalue"],columns=["Bartlett test"]).T

# Indice KMO Global
def global_kmo_index(x):
  # Matrice des corrélations
  corr = x.corr(method="pearson").values
  # Matrice des corrélations partielles
  pcorr = x.pcorr().values
  # Indice KMO global
  np.fill_diagonal(corr,0)
  np.fill_diagonal(pcorr,0)
  return np.sum(corr**2)/(np.sum(corr**2)+np.sum(pcorr**2))

# Indice KMO par variable
def per_item_kmo_index(x):
  # Matrice des corrélations linéaires
  corr = x.corr(method = "pearson").values
  # Matrice des corrélations partielles
  pcorr = x.pcorr().values
  # Indice KMO global
  np.fill_diagonal(corr,0)
  np.fill_diagonal(pcorr,0)
  A = np.sum(corr**2, axis=0)
  B = np.sum(pcorr**2, axis=0)
  kmo_per_item = A /(A+B)
  return pd.DataFrame(kmo_per_item,index=x.columns,columns = ["KMO"])


def paste(*args, sep = ' ', collapse = None):
    """
    Port of paste from R
    Args:
        *args: lists to be combined
        sep: a string to separate the terms
        collapse: an optional string to separate the results
    Returns:
        A list of combined results or a string of combined results if collapse is not None
    """
    combs = list(itertools.product(*args))
    out = [sep.join(str(j) for j in i) for i in combs]
    if collapse is not None:
        out = collapse.join(out)
    return out
paste0 = partial(paste, sep = '')

def paste(*args, sep = " ", collapse = None):
    l = [list(arg) if isinstance(arg, str) else arg if hasattr(arg, '__len__') else list(str(arg)) for arg in args]
    l = list(itertools.islice((sep.join(parts) for parts in zip(*(itertools.cycle(map(str, e)) for e in l))), (max((len(x) for x in l)))))
    if collapse is not None:
        l = collapse.join(l)
    return l

# Load R datasets in Python
def load_rdatasets(packages=str,data_name=str):
    from rpy2.robjects.packages import importr, data
    import pandas as pd
    import numpy as np
    r_df = data(importr(packages)).fetch(data_name)[data_name]
    py_df = pd.DataFrame.from_dict({ key : np.asarray(r_df.rx2(key)) for key in r_df.names })
    py_df.index = r_df.rownames
    return py_df


def cramer_v(x,y):
    #create 2x2 table
    tab =pd.crosstab(x,y)
    #Chi-squared test statistic, sample size, and minimum of rows and columns
    chi2 = st.chi2_contingency(tab,correction=False)[0]
    n = np.sum(tab)
    minDim = min(tab.shape)-1
    #calculate Cramer's V 
    return  np.sqrt((chi2/n) / minDim)

def tschuprow_t(x,y):
    pass


