# -*- coding: utf-8 -*-
import numpy as np

def weightedcorrcoef(x,y=None,w=None):
    """
    Weighted pearson correlation
    ----------------------------

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations. Each row of x 
        represents a variable, and each column a single observation of all those variables.
    
    y : array_like, optional
        An additional set of variables and observations. y has the same shape as x.
    
    w : None or 1-D ndarray
        weights for each observation, with same length as zero axis of data.
    
    Return
    ------
    corrcoef : 
        weighted correlation with default ddof=0
    """
    # Weighted covariance
    def wcov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - np.average(a=x,weights=w)) * (y - np.average(a=x,weights=w))) / np.sum(w)
    # Weighted correlation
    def wcorr(x, y, w):
        """Weighted Correlation"""
        return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))

    if w is None:
        corrcoef = np.corrcoef(x=x,y=y,rowvar=False,ddof=0)
    else:
        w = np.array(w)
        if y is None:
            if x.ndim !=1 :
                x = np.asarray(x)
                corrcoef = np.zeros(shape=(x.shape[1],x.shape[1]))
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        corrcoef[i,j] = wcorr(x[:,i],x[:,j],w=w)
            else:
                raise ValueError("x is a 1-D array, y must not be None")
        else:
            x = np.array(x)
            y = np.array(y)
            # Resize
            if x.ndim == 1:
                x = x.reshape(-1,1)
            if y.ndim == 1:
                y = y.reshape(-1,1)
            # Append
            x = np.append(x,y,axis=1)
            corrcoef = np.zeros(shape=(x.shape[1],x.shape[1]))
            for i in range(x.shape[1]):
                for j in range(x.shape[1]):
                    corrcoef[i,j] = wcorr(x[:,i],x[:,j],w=w)
    return corrcoef