# -*- coding: utf-8 -*-

def get_eig(
        obj
):
    """
    Extract the eigenvalues/variances of dimensions
    
    Eigenvalues correspond to the amount of the variation explained by each principal component.

    Parameters
    ----------
    obj : class 
        An object which has attribute ``eig_``.

    Returns
    -------
    eig : DataFrame of shape (n_components, 4)
        The eigenvalue, difference, variance percent and cumulative variance of percent.

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, get_eig
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children.data)
    CA(col_sup=(5,6,7),row_sup=range(14,18),sup_var=8)
    >>> eig = get_eig(clf)
    >>> print(eig)
    """
    if not hasattr(obj,"eig_"):
        raise ValueError("{} does not have an attribute named `eig_`.".format(obj.__class__.__name__))
    return obj.eig_

def get_eigenvalue(
        obj
):
    """
    Extract the eigenvalues/variances of dimensions

    see :class:`~scientisttools.get_eig`.
    """
    return get_eig(obj)