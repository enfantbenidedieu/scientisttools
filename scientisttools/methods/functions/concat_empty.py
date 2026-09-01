# -*- coding: utf-8 -*-
from pandas import concat

def concat_empty(initial, actual, axis=0, **kwargs):
    """
    Concatenate DataFrame or Series

    Concatenate pandas objects along a particular axis.

    Parameters
    ----------
    initial : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns) or None
        Initial objects.

    actual : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        actual objects.
    
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.

    Returns
    -------
    obj : DataFrame or Series
        Concatenate object.
    """
    if initial is None:
        obj = actual 
    else:
        obj = concat((initial,actual),axis=axis,**kwargs)
    return obj                                      