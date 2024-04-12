# -*- coding: utf-8 -*-
import numpy as np

def reconst(self,n_components=None):
    """
    Reconstitution of data
    ----------------------

    This function reconstructs a data set from the result of a PCA 

    Parameters:
    -----------
    self : an object of class PCA

    n_components : int, the number of dimensions to use to reconstitute data.

    Return
    ------
    X : Reconstitution data.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "pca":
        raise ValueError("'self' must be an object of class PCA.")
    
    if n_components is not None:
        if n_components > self.call_["n_components"]:
            raise ValueError("Enter good number of n_components" )
    else:
        raise ValueError("'n_components' must be pass.")
    
    # Valeurs centrées
    Z = np.dot(self.ind_["coord"].iloc[:,:n_components],self.svd_["V"][:,:n_components].T)
    # Déstandardisation et décentrage
    X = self.call_["X"].copy()
    for k in np.arange(self.var_["coord"].shape[0]):
        X.iloc[:,k] = Z[:,k]*self.call_["std"].values[k] + self.call_["means"].values[k]
    
    return X