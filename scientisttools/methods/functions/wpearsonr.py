# -*- coding: utf-8 -*-
from numpy import ones, array, sum, sqrt, abs, c_
from scipy import stats
from collections import namedtuple
from typing import NamedTuple
from pandas import DataFrame

#intern function
from .wcorrcoef import wcorrcoef

def wpearsonr(x,y,weights=None) -> NamedTuple:
    """
    Weighted pearson correlation coefficient and p-value for testing non-correlation.
    ---------------------------------------------------------------------------------

    Description
    -----------
    Test for weighted pearson correlation coefficient.

    Usage
    -----
    ```python
    >>> wpearsonr(x,y,weights=None)
    ```

    Parameters
    ----------
    `x`: numpy array or pandas series

    `y`: numpy array or pandas series

    `weights`: an optional individuals weights 

    Return(s)
    ---------
    nametuple containing estimates of the weighted correlation, the degree of freedom and the pvalue associated:
    
    `statistic`: a numeric (=float) value indicating the weighted Pearson product-moment correlation coefficient
    
    `dof`: an integer indicating the degre of freedom   
    
    `pvalue`: a numeric (=float) value indicating the p-value associated
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> import numpy as np
    >>> from scientisttools import wpearsonr
    >>> x = np.arange(1,11)
    >>> y = np.array([1,2,3,8,7,6,5,8,9,10])
    >>> wt = np.array([0,0,0,1,1,1,1,1,0,0])
    >>> res = wpearsonr(x=x,y=y,weights=wt)
    >>> res
    ...WPearsonRResult(statistic=-0.24253562503633294, dof=8, pvalue=0.49957589436325933)
    ```
    """
    #set weights
    if weights is None:
        weights = ones(x.shape[0])/x.shape[0]
    else:
        weights = array([x/sum(weights) for x in weights])

    statistic = wcorrcoef(DataFrame(c_[x,y]),weights=weights).iloc[0,1]
    t_stat, dof = statistic*sqrt(((len(x)-2)/(1- statistic**2))), len(x) - 2
    pvalue = 2*(1 - stats.t.cdf(abs(t_stat),dof))
    return namedtuple("WPearsonRResult",["statistic","dof","pvalue"])(statistic,dof,pvalue)