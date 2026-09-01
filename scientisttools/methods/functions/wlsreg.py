# -*- coding: utf-8 -*-
from statsmodels.api import WLS, add_constant

def wlsreg(X,Y,w=None):
    """
    Weighted Least Squares Regression

    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_xcolumns)
        Explanatory variables

    Y : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_ycolumns)
        Target variables

    w : 
        Weighe
    
    Returns
    -------
    


    """
    #set weights
    if w is None: 
        w = 1.0/Y.shape[0]
    return {y : WLS(endog=Y[y],exog=add_constant(X),weights=w).fit(disp=False) for y in Y.columns}
