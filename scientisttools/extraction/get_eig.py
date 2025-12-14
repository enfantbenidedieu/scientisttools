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
    `self`: an object with attribute named `eig_`

    Returns
    -------
    a apndas DaatFrame conatining the eigenvalue, difference, variance percent and cumulative variance of percent

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, CA, get_eig
    >>> children = load_dataset("children")
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> res_ca.fit(children)
    >>> eig = get_eig(res_ca)
    >>> print(eig)
    ```
    """
    if not hasattr(self,"eig_"):
        raise ValueError("{} does not have an attribute named `eig_`.".format(self.model_))
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