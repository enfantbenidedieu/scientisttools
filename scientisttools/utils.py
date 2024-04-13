# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """
    Argument Verification Using Partial Matching
    --------------------------------------------

    Description
    -----------
    match_arg matches a character arg against a table of candidate values as specified by choices

    Parameters
    ----------
    
    """
    match_a = [elt for elt in lst if x in elt]

    return 


def is_euclidean(D,plot=False,printf=False,tol=1e-07):
    """
    Confirmation of the Euclidean nature of a distance matrix by the Gower's theorem
    --------------------------------------------------------------------------------

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


# Test de sphéricité de Bartlett
def BartlettSphericityTest(x):
  test = calculate_bartlett_sphericity(x.values)
  return pd.DataFrame(test,index=["stats","pvalue"],columns=["Bartlett test"]).T




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













    

    
    


