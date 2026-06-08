# -*- coding: utf-8 -*-
from warnings import warn
from numpy import max, abs, dot, eye
import numpy.linalg as linalg
from collections import namedtuple

def gensvd(
          Rxy,Rxx=None,Ryy=None, nu=None, nv=None, tol=1e-10
):
    """
    Generalized Singular Value Decomposition of Matrices
    
    Parameters
    ----------
    A : 2d array-like of shape (p,q)
        Matrix

    B : 2d array-like of shape (p,q)
        Rxx is order p symmetric positive definite.

    C : 2d array-like of shape (q,q)
        Ryy is order q symmetric positive definite.

    Returns
    -------
    xcoef : 2d array-like of shape (p,s)
        The coefficients of ``X``.
    ycoef : 2d array-like of shape (q,s)
        The coefficients of ``Y``.
    cor : 1d array-like of shape(s,) with s = min(p,q)
        The squared correlations values
    """
    #set nu and nv
    if nu is None: nu = Rxy.shape[0]
    if nv is None: nv = Rxy.shape[1]

    #set Rxx and Ryy
    if Rxx is None: Rxx = eye(nu)
    if Ryy is None: Ryy = eye(nv)

    #check if Rxx is square
    if Rxx.shape[0] != Rxx.shape[1]:
         raise TypeError("Rxx is not square")
    
    #check if Ryy is square
    if Ryy.shape[0] != Ryy.shape[1]:
        raise TypeError("Rxx is not square")
    
    s = min(nu,nv)
    if (max(abs(Rxx - Rxx.T))/max(abs(Rxx))) > tol:
        warn("Rxx not symmetric.")
        Rxx = 0.5*(Rxx + Rxx.T)
    if (max(abs(Ryy - Ryy.T))/max(abs(Ryy))) > tol:
        warn("Ryy not symmetric.")
        Ryy = 0.5*(Ryy + Ryy.T)
    
    Rxxinv, Ryyinv  = linalg.inv(linalg.cholesky(Rxx,upper=True)), linalg.inv(linalg.cholesky(Ryy,upper=True))
    D = Rxxinv.T.dot(Rxy).dot(Ryyinv)
    if nu >= nv:
        U, vs, V = linalg.svd(D)
        U, V = dot(Rxxinv,U), dot(Ryyinv, V.T)
    else:
        V, vs, U = linalg.svd(D.T)
        U, V = dot(Rxxinv,U.T), dot(Ryyinv,V)
    
    return namedtuple("gensvdResult",["xcoef","ycoef","cor"])(U[:,:s],V[:,:s],vs[:s])