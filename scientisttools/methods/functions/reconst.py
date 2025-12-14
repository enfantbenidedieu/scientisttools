# -*- coding: utf-8 -*-
from numpy import outer, sqrt
from pandas import DataFrame, Series

def reconst(self,n_components=None) -> DataFrame:
    """
    Reconstruction of the data from the PCA, CA, MCA or MFA results
    ---------------------------------------------------------------

    Description
    -----------
    Reconstruct the data from the PCA, CA, MCA or MFA results.

    Usage
    -----
    ```
    >>> reconst(self,n_components=None)
    ```

    Parameters
    ----------
    `self`: an object of class PCA, CA, MCA or MFA

    `n_components`: int, the number of dimensions to use to reconstitute data (by default None and the number of dimensions calculated for the PCA, CA or MFA is used)

    Returns
    -------
    `X`: pandas DataFrame with the number of individuals and the number of variables used for the PCA, CA or MFA

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Escofier B., Pagès J. (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod p131

    * Rakotomalala, R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    Examples
    --------
    ```
    >>> from scientisttools import decathlon, PCA, reconst
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12),rotate=None)
    >>> res_pca.fit(decathlon)
    >>> rec = reconst(res_pca, n_components=2)
    ```
    """
    if self.model_ not in ["pca","ca","mca","mfa"]: #check if self is an object of class PCA, CA or MFA
        raise ValueError("'self' must be an object of class PCA, CA, MCA or MFA")
    
    if n_components is not None:
        if n_components < 1:
            raise ValueError("'n_components' must be greater than or equal to 1")
        if n_components > self.call_.n_components:
            raise ValueError("Enter good number of n_components" )
    else:
        raise ValueError("'n_components' must be pass.")
    
    if self.model_ in ("pca","mca"):
        F, G = self.ind_.coord.iloc[:,:n_components], self.var_.coord.iloc[:,:n_components]
        if self.model_ == "pca":
            m = self.call_.var_weights
        else:
            m = self.call_.levels_weights
    elif self.model_ == "ca":
        F, G, m = self.row_.coord.iloc[:,:n_components], self.col_.coord.iloc[:,:n_components], self.call_.col_marge

    #initial step : z_ik
    hatX = F.dot(G.div(sqrt(G.pow(2).T.dot(m)),axis=1).T)
    if self.model_ == "pca":
        return hatX.mul(self.call_.scale,axis=1).add(self.call_.center,axis=1)
    elif self.model_ == "ca":
        return hatX.add(1).mul(self.call_.row_marge,axis=0).mul(self.call_.col_marge,axis=1).mul(self.call_.total)
    elif self.model_ == "mca":
        hatX = hatX.add(1).mul(self.call_.dummies.mean(axis=0),axis=1)
        return (hatX > (self.call_.X.shape[1]/self.call_.dummies.shape[1])).astype(int)