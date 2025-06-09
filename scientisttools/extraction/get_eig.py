# -*- coding: utf-8 -*-
from pandas import DataFrame

def get_eig(self) -> DataFrame:
    """
    Extract the eigenvalues/variances of dimensions
    -----------------------------------------------

    Description
    -----------
    Eigenvalues correspond to the amount of the variation explained by each principal component.

    Usage
    -----
    ```python
    >>> get_eig(self)
    ```

    Parameters:
    -----------
    `self` : an object of class PCA, PartialPCA, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT, DMFA, CMDSCALE

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load children dataset
    >>> from scientisttools import load_children
    >>> children  = load_children()
    >>> from scientisttools import CA, get_eig
    >>> res_ca = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)),quali_sup=8)
    >>> res_ca.fit(children)
    >>> eig = get_eig(res_ca)
    >>> print(eig)
    ```
    """
    if self.model_ not in ["pca","partialpca","ca","mca","specificmca","famd","mpca","pcamix","efa","mfa","mfaqual","mfamix","mfact","dmfa","cmdscale"]:
        raise TypeError("'self' must be an object of class PCA, PartialPCA, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, EFA, MFA, MFAQUAL, MFAMIX, MFACT, DMFA, CMDSCALE")
    return self.eig_

def get_eigenvalue(self) -> DataFrame:
    """
    Extract the eigenvalues/variances of dimensions
    -----------------------------------------------

    see get_eig(self)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    return get_eig(self)