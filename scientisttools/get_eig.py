# -*- coding: utf-8 -*-

import pandas as pd

def get_eig(self) -> pd.DataFrame:

    """
    self : an instance of class PCA, PartialPCA, CA, MCA, FAMD, MFA,CMDS

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent
    """
    if self.model_ in ["pca","ppca","ca","mca","famd","mfa","cmds"]:
        eig = pd.DataFrame(self.eig_.T,columns=["eigenvalue","difference","proportion","cumulative"],index = self.dim_index_)
        return eig
    else:
        raise ValueError("Eroor : 'self' must be an instance of class PCA, PPCA, CA, MCA, FAMD, MFA, CMDS")

def get_eigenvalue(self) -> pd.DataFrame:

    """
    self : an instance of class PCA, PartialPCA, CA, MCA, FAMD, MFA, MDS

    Returns
    -------
    eigenvalue, variance percent and cumulative variance of percent
    """
    return get_eig(self)