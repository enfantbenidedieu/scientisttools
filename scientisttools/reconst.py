import numpy as np

def reconst(self,n_components=None):
    """
    Reconstruction of the data from the PCA, CA or MFA results
    ----------------------------------------------------------

    Description
    -----------
    Reconstruct the data from the PCA, CA or MFA results.

    Usage
    -----
    ```python
    >>> reconst(self,n_components=None)
    ```

    Parameters
    ----------
    `self` : an object of class PCA, CA or MFA

    `n_components` : int, the number of dimensions to use to reconstitute data (by default None and the number of dimensions calculated for the PCA, CA or MFA is used)

    Returns
    -------
    `X` : pandas data frame with the number of individuals and the number of variables used for the PCA, CA or MFA

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load decathlon dataset
    >>> from scientisttools import load_decathlon
    >>> decathlon = load_decathlon()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(quanti_sup = [10,11], quali_sup=12)
    >>> res_pca.fit(decathlon)
    >>> from scientisttools import reconst
    >>> rec = reconst(res_pca, n_components=2)
    ```
    """
    # Check if self is an object of class PCA, CA or MFA
    if self.model_ not in ["pca","ca","mfa"]:
        raise ValueError("'self' must be an object of class PCA, CA or MFA")
    
    if n_components is not None:
        if n_components > self.call_["n_components"]:
            raise ValueError("Enter good number of n_components" )
    else:
        raise ValueError("'n_components' must be pass.")
    
    if self.model_ == "pca":
        # Valeurs centrées
        Z = np.dot(self.ind_["coord"].iloc[:,:n_components],self.svd_["V"][:,:n_components].T)
        # Déstandardisation et décentrage
        X = self.call_["X"].copy()
        for k in np.arange(self.var_["coord"].shape[0]):
            X.iloc[:,k] = Z[:,k]*self.call_["std"].values[k] + self.call_["means"].values[k]
    
    return X