# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from collections import namedtuple

def wpearsonr(x,y,weights=None):
    """
    Weighted Pearson correlation coefficient and p-value for testing non-correlation.
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
    `x` : numpy array or pandas series

    `y` : numpy array or pandas series

    `weights` : an optional individuals weights 

    Return
    ------
    nametuple containing estimates of the weighted correlation, the degree of freedom and the pvalue associated :
        * statistic : weighted Pearson product-moment correlation coefficient
        * dof : degre of freedom   
        * pvalue : the p-value associated
    
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
     # Set weights
    if weights is None:
        weights = np.ones(x.shape[0])/x.shape[0]
    else:
        weights = np.array([x/np.sum(weights) for x in weights])
    
    statistic = DescrStatsW(np.c_[x,y],weights=weights).corrcoef[0,1]
    t_stat, dof = statistic*np.sqrt(((len(x)-2)/(1- statistic**2))), len(x) - 2
    pvalue = 2*(1 - stats.t.cdf(np.abs(t_stat),dof))
    return namedtuple("WPearsonRResult",["statistic","dof","pvalue"])(statistic,dof,pvalue)