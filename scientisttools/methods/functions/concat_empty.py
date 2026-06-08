# -*- coding: utf-8 -*-
from pandas import concat

def concat_empty(
        initial,actual,axis=0, **kwargs
):
    """
    Concatenate DataFrame or Series

    Concatenate pandas objects along a particular axis.

    Parameters
    ----------
    initial : Series or DataFrame
        Initial objects.

    actual : Series or DataFrame
        actual objects.
    
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.

    Returns
    -------
    obj : DataFrame or Series
        Concatenate object.
    """
    obj = actual if initial is None else concat((initial,actual),axis=axis,**kwargs)
    return obj                                      