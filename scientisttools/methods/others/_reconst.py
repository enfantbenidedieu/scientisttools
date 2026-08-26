# -*- coding: utf-8 -*-

# intern function
from ..functions.concat_empty import concat_empty

def reconst(obj,
            ncp=None):
    """
    Reconstruction of the data from the PCA, CA, MCA or MFA results

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.CA`:class:`~scientisttools.MCA` or :class:`~scientisttools.MFA`.

    ncp: int, default = None
        The number of dimensions to use to reconstitute data. If None, the number of dimensions calculated for the PCA, CA, MCA or MFA is used.

    Returns
    -------
    X : DataFrame of shape (n_samples, n_columns)
        Output data with the number of individuals and the number of variables used for the :class:`~scientisttools.PCA`, :class:`~scientisttools.CA`, :class:`~scientisttools.MCA` or :class:`~scientisttools.MFA`.

    References
    ----------
    [1] Escofier B., Pagès J. (2023), `Analyses Factorielles Simples et Multiples <https://cdn-cms.f-static.com/uploads/1460418/normal_5b9ba5dc15394.pdf>`_. 4ed, Dunod.

    [2] Rakotomalala, R. (2020). `Pratique des méthodes factorielles avec Python <https://hal.science/hal-04868625>`_. Université Lumière Lyon 2. Version 1.0

    Examples
    --------
    >>> from scientisttools;datasets import decathlon
    >>> from scientisttools import PCA, reconst
    >>> pca = PCA(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> pca.fit(decathlon.data)
    PCA(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> rec = reconst(res_pca, ncp=2)
    """
    #check if obj is an object of class PCA, CA or MFA
    if obj.__class__.__name__ not in ["PCA","CA","MCA","MFA"]:
        raise ValueError("obj must be an object of class PCA, CA, MCA or MFA")
    
    if ncp is not None:
        if ncp < 1:
            raise ValueError("'ncp' must be greater than or equal to 1")
        if ncp > obj.call_.ncp:
            raise ValueError("Not convenient ncp" )
    else:
        raise ValueError("'ncp' must be pass.")
    
    if obj.__class__.__name__ in ("PCA","MCA","MFA"):
        F = obj.ind_.coord
        if obj.__class__.__name__ == "PCA":
            G = obj.quanti_var_.coord
        elif obj.__class__.__name__ == "MCA":
            G = obj.levels_.coord
        else:
            G = None
            if hasattr(obj,"quanti_var_"):
                G = concat_empty(G,obj.quanti_var_.coord,axis=0)
            if hasattr(obj,"levels_"):
                G = concat_empty(G,obj.levels_.coord,axis=0)
    elif obj.__class__.__name__ == "CA":
        F, G = obj.row_.coord, obj.col_.coord

    # initial step : z_ik
    hatX = F.dot((G/obj.svd_.vs[:ncp]).T)
    if obj.__class__.__name__ == "PCA":
        return ((hatX * obj.call_.scale)/obj.call_.col_w) + obj.call_.center
    elif obj.__class__.__name__ == "CA":
        return (((hatX + 1).T * obj.call_.row_m).T * obj.call_.col_m) * obj.call_.total
    elif obj.__class__.__name__ == "MCA":
        hatX = (hatX + 1) * obj.call_.dummies.mean(axis=0)
        return (hatX > (obj.call_.X.shape[1]/obj.call_.dummies.shape[1])).astype(int)
    else:
        return (((hatX + obj.call_.z_center)*obj.call_.scale)/obj.call_.col_w) + obj.call_.center