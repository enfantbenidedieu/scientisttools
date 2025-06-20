# -*- coding: utf-8 -*-
from numpy import ndarray
from pandas import Series

def get_indices(x,value) -> list:
    """
    Fill all occurrences of an element

    Description
    ------------
    Fill all occurrences of an element

    Usage
    -----
    ```python
    >>> get_indices(x,value)
    ```

    Parameters
    ----------
    `x` : a list/tuple/ndarray/Series of element

    `value` : value for which occurence should be find

    Return
    ------
    list with occurrences

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #check if x is a list, a tuple, an array or a pandas series
    if not isinstance(x, (list,tuple,ndarray,Series)):
        raise TypeError("'x' must be a list/tuple/array/Series")
    
    #convert to list
    if isinstance(x,ndarray):
        x = x.tolist()
    elif isinstance(x,tuple):
        x = list(x)
    elif isinstance(x,Series):
        x = x.values.tolist()

    indices = list()
    i = 0
    while True:
        try:
            i = x.index(value,i) # find an occurrence of value and update i to that index
            indices.append(i) # add i to the list
            i += 1 # advance i by 1
        except ValueError as e:
            break
    return indices