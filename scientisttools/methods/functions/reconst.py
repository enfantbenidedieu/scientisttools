# -*- coding: utf-8 -*-
from mapply.mapply import mapply
from pandas import DataFrame

def reconst(self,n_components=None) -> DataFrame:
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
        if n_components > self.call_.n_components:
            raise ValueError("Enter good number of n_components" )
    else:
        raise ValueError("'n_components' must be pass.")
    
    if self.model_ == "ca":
        X = self.call_.X
        F = X

    else:
        #variables factor coordinates
        if self.model_ == "pca":
            var_coord = self.var_.coord.iloc[:,:n_components]
        
        #individuals factor coordinates
        ind_coord = self.ind_.coord.iloc[:,:n_components]

        hatX = ind_coord.dot(mapply(var_coord,lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
        
        #Principal Component Analysis (PCA)
        if self.model_ == "pca":
            hatX = mapply(hatX,lambda x : (x*self.call_.scale)+self.call_.center,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #Multiple Factor Analysis (MFA)
        if self.model_ == "mfa":
            pass

    return hatX