# -*- coding: utf-8 -*-
from numpy import ndarray
from pandas import Series

def get_indices(x,value):
    """
    Fill all occurrences of an element

    Parameters
    ----------
    x : 1d array-like of shape (n_samples,)
        Input data.

    value : int
        value for which occurence should be find

    Return
    ------
    indices : list
        Occurrences of an element.
    """
    #check if x is a list, a tuple, an array or a pandas series
    if not isinstance(x, (list,tuple,ndarray,Series)):
        raise TypeError("x must be a list/tuple/array/Series")
    
    #convert to list
    if isinstance(x,ndarray):
        x = x.tolist()
    elif isinstance(x,tuple):
        x = list(x)
    elif isinstance(x,Series):
        x = x.to_numpy().tolist()

    indices = []
    i = 0
    while True:
        try:
            i = x.index(value,i) # find an occurrence of value and update i to that index
            indices.append(i) # add i to the list
            i += 1 # advance i by 1
        except ValueError as e:
            break
    return indices