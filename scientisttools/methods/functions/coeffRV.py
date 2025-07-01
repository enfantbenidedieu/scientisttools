# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .coeffLg import coeffLg

def coeffRV(X=None,Y=None,first_eigen_X=None,first_eigen_Y=None):
    """
    Calculate the RV coefficient between groups
    ---------------------------------------------

    Description
    -----------
    Calculate the RV coefficients between two groups X and Y

    Usage
    -----
    ```python
    >>> coeffRV(X=None,Y=None,first_eigen_X=None,first_eigen_Y=None)
    ```

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_samples, n_columns) or (n_columns, n_columns)

    `Y` : pandas dataframe of shape (n_samples, n_columns)

    'first_eigen_X' : a float specifying the first eigenvalue of X (by default = None)

    `first_eigen_Y` : a float specifying the first eigenvalue of Y (by default = None)

    Returns
    -------
    pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com    
    """

    if Y is None:
        Lg = X
    else:
        Lg = pd.DataFrame(coeffLg(X=X,Y=Y,first_eigen_X=first_eigen_X,first_eigen_Y=first_eigen_Y))
        Lg.index = ["Gr"+str(x+1) for x in range(Lg.shape[0])]
        Lg.columns = ["Gr"+str(x+1) for x in range(Lg.shape[1])]
        
    # Initialiaze
    RV = pd.DataFrame().astype("float")
    # Iteration
    for i in Lg.index:
        for j in Lg.columns:
            RV.loc[i,j] = Lg.loc[i,j]/(np.sqrt(Lg.loc[i,i])*np.sqrt(Lg.loc[j,j]))
    return RV