# -*- coding: utf-8 -*-
import numpy as np

def bicenter_wt(X, row_wt=None,col_wt=None):
    """
    This function creates a doubly centred matrix
    ---------------------------------------------

    Parameters
    ----------
    X : a matrix of shape (n_rows, n_columns)

    row_wt : 

    col_wt : 

    Return
    ------
    A doubly centred matrix
    """
    X = np.array(X)
    n, p = X.shape
    if row_wt is None:
        row_wt = np.ones(n)
    if col_wt is None:
        col_wt = np.ones(p)
    row_wt = np.array(row_wt)
    col_wt = np.array(col_wt)
    sr = sum(row_wt)
    row_wt = row_wt/sr
    st = sum(col_wt)
    col_wt = col_wt/st
    row_mean = np.apply_along_axis(func1d=np.sum,axis=0,arr=np.apply_along_axis(arr=X,func1d=lambda x : x*row_wt,axis=0))
    col_mean = np.apply_along_axis(func1d=np.sum,axis=0,arr=np.apply_along_axis(arr=np.transpose(X),func1d=lambda x : x*col_wt,axis=0))
    col_mean = col_mean - np.sum(row_mean * col_wt)
    X = np.apply_along_axis(func1d=lambda x : x - row_mean,axis=1,arr=X)
    X = np.transpose(np.apply_along_axis(func1d=lambda x : x - col_mean,axis=1,arr=np.transpose(X)))
    return X