# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from .weightedcorrcoef import weightedcorrcoef

def weightedcorrtest(x,y,weights=None):
    """
    Weighted Pearson correlation coefficient and p-value for testing non-correlation.
    ---------------------------------------------------------------------------------

    Description
    -----------
    Test for weighted pearson correlation coefficient.

    Usage
    -----
    ```python
    >>> weightedcorrtest(x,y,weights=None)
    ```

    Parameters
    ----------
    `x` : numpy array or pandas series

    `y` : numpy array or pandas series

    `weights` : an optional individuals weights 

    Return
    ------
    a dictionary containing estimates of the weighted correlation, the degree of freedom and the pvalue associated :
        * statistic : weighted Pearson product-moment correlation coefficient
        * dof : degre of freedom   
        * pvalue : the p-value associated
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```
    >>> import numpy as np
    >>> from scientisttools import weightedcorrtest
    >>> x = np.arange(1,11)
    >>> y = np.array([1,2,3,8,7,6,5,8,9,10])
    >>> wt = np.array([0,0,0,1,1,1,1,1,0,0])
    >>> res = weightedcorrtest(x=x,y=y,weights=wt)
    ```
    """
     # Set weights
    if weights is None:
        weights = np.ones(x.shape[0])/x.shape[0]
    else:
        weights = np.array([x/np.sum(weights) for x in weights])
    
    statistic = weightedcorrcoef(x=x,y=y,w=weights)[0,1]
    t_stat = statistic*np.sqrt(((len(x)-2)/(1- statistic**2)))
    dof = len(x) - 2
    pvalue = 2*(1 - sp.stats.t.cdf(np.abs(t_stat),dof))
    return {"statistic" : statistic,"dof" : dof ,"pvalue" : pvalue}