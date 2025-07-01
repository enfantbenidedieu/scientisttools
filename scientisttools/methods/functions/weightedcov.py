
import numpy as np

def weightedcov(x,y,weights=None):
    """
    Weighted Covariance Matrices
    ----------------------------

    Description
    -----------

    Usage
    -----
    ```
    >>> weightedcov(x,y,weights=None)
    ```

    Parameters
    ----------
    `x` : numpy array or pandas series/dataframe

    `y` : numpy array or pandas series/dataframe

    `weights` :  weights for each observation, with same length as zero axis of data.

    Returns
    -------
    a 2-D array of weighted covariance matrix

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE
    """

    def wcov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - np.average(a=x,weights=w)) * (y - np.average(a=x,weights=w))) / np.sum(w)

    # Set number of rows
    if x.ndim == 1:
        nrows = len(x)
    else:
        nrows = x.shape[0]
    
    # Set weights
    if weights is None:
        weights = np.ones(nrows)/nrows
    else:
        weights = np.array([x/np.sum(weights) for x in weights])
    
    if y is None:
        x = np.asarray(x)
    else:
        x = np.array(y)
        y = np.array(y)
        # Resize
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        # Append
        x = np.append(x,y,axis=1)
    # 
    mat = np.zeros(shape=(x.shape[1],x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            mat[i,j] = wcov(x[:,i],x[:,j],w=weights)
    return mat