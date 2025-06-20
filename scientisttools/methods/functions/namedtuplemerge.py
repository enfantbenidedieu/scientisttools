# -*- coding: utf-8 -*-
from collections import namedtuple
from functools import reduce
from itertools import chain
from operator import add

def namedtuplemerge(typename=None,*args):
    """"
    Merge namedtuple
    
    """

    if typename is None:
        typename = '_'.join(arg.__class__.__name__ for arg in args)
    elif not isinstance(typename,str):
        raise TypeError("'typename' must be a string")
    
    cls = namedtuple(typename, reduce(add, (arg._fields for arg in args)))
    return cls(*chain(*args))