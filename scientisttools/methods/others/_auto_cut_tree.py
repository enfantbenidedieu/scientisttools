# -*- coding: utf-8 -*-
from numpy import ones, outer,divide,add,triu_indices,cumsum,argmax
from pandas import concat
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from collections import namedtuple

def auto_cut_tree(obj,
                  mincl,
                  maxcl,
                  metric,
                  method,
                  order,
                  w=None):
    """
    Automatic tree cut
    
    Automatic tree cut to determine optimal number of clusters.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.

    mincl : int
        The least possible number of clusters suggested.

    maxcl : int
        The higher possible number of clusters suggested.

    metric : str
        The metric used to build the tree. For more, see `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`.

    method : str
        The method used to build the tree. For more, see `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`.

    order : bool
        If True, clusters are ordered following their center coordinate on the first axis.

    w : 1d array-like of shape (n_samples,), default=None
        Weights for each observation, with same length as zero axis of data.

    Returns
    -------
    res : auto_cut_tree
        An object containing attributes:

        obj : class
            An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`.
        
        Z : 2D array-like of shape (n_samples - 1, 4)
            The hierarchical clustering encoded as a linkage matrix.

        ncl : int
            The number of clusters.
        
        within : float
            The within inertia

        inertia : float
            Inertia

        quot : 1D array-like

    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class PCA, MCA, FAMD, PCAmix, MPCA, MFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("PCA","MCA","FAMD","PCAmix","MPCA","MFA")):
        raise TypeError("'obj' must be an object of class PCA, MCA, FAMD, PCAmix, MPCA, MFA")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # order dataset
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if order:
        data = concat((obj.ind_.coord,obj.call_.X,obj.call_.row_w),axis=1)
        if w is not None:
            w = w[::-1]
        data = data.sort_values(by=data.columns.tolist()[0],ascending=True)
        obj.ind_ = obj.ind_._replace(coord = data.iloc[:,:obj.ind_.coord.shape[1]])
        obj.call_ = obj.call_._replace(X=data.iloc[:,(obj.ind_.coord.shape[1]+1):(data.shape[1]-1)],row_w=data.iloc[:,-1])
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # automatic tree cut
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # extract individuals coordinates
    X = obj.ind_.coord
    # dissimilarity matrix
    do = pdist(X,metric=metric)**2
    # set weights
    if w is None:
        w = ones(X.shape[0])

    # effectifs
    eff = divide((outer(w,w)/sum(w)),add.outer(w,w))
    dissi = do*eff[triu_indices(eff.shape[0], k = 1)]
    # agglometrive clustering
    Z = linkage(dissi,metric=metric,method=method)
    inertia = Z[:,2][::-1]
    intra = cumsum(inertia[::-1])[::-1]
    quot = inertia[(mincl-1):maxcl]/inertia[(mincl):(maxcl+1)]
    ncl = (argmax(quot)+1) + mincl - 1

    # convert to ordered dictionary
    res_ = {"obj":obj,"Z":Z,"ncl":ncl,"within":intra,"inertia":inertia,"quot":quot}
    # convert to namedtuple 
    return namedtuple("auto_cut_tree",res_.keys())(*res_.values())