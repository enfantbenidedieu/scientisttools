# -*- coding: utf-8 -*-
from numpy import outer, sqrt
from mapply.mapply import mapply
from pandas import DataFrame, Series, from_dummies

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
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> rec = reconst(res_pca, n_components=2)
    ```
    """
    # Check if self is an object of class PCA, CA or MFA
    if self.model_ not in ["pca","ca","mca","mfa"]:
        raise ValueError("'self' must be an object of class PCA, CA, MCA or MFA")
    
    if n_components is not None:
        if n_components > self.call_.n_components:
            raise ValueError("Enter good number of n_components" )
    else:
        raise ValueError("'n_components' must be pass.")
    
    if self.model_ in ("pca","mca"):
        F, G = self.ind_.coord.iloc[:,:n_components], self.var_.coord.iloc[:,:n_components]
        if self.model_ == "pca":
            m = self.call_.var_weights
        else:
            m = self.call_.mod_weights
    elif self.model_ == "ca":
        F, G, m = self.row_.coord.iloc[:,:n_components], self.col_.coord.iloc[:,:n_components], self.call_.col_marge

    #initial step
    hatX = F.dot(mapply(G,lambda x : x/sqrt(G.pow(2).T.dot(m)),axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
    if self.model_ == "pca":
        return mapply(hatX,lambda x : (x*self.call_.scale)+self.call_.center,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    elif self.model_ == "ca":
        return mapply(mapply(hatX,lambda x : x*self.call_.row_marge,axis=0,progressbar=False,n_workers=self.call_.n_workers),lambda x : (x*self.call_.col_marge)-1,axis=1,progressbar=False,n_workers=self.call_.n_workers).mul(self.call_.total)
    elif self.model_ == "mca":
        hatX = mapply(hatX.add(1),lambda x : x*self.call_.dummies.mean(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
        return (hatX > (self.call_.X.shape[1]/self.call_.dummies.shape[1])).astype(int)
    

    
    #if self.model_ == "ca":
    #    X_original = self.call_.X
    #    freq = X_original.div(X_original.sum().sum())
    #    col_marge, row_marge = freq.sum(axis=0), freq.sum(axis=1)
    #    if n_components > 0:
    #        row_coord, col_coord = self.row_.coord.iloc[:,:n_components], self.col_.coord.iloc[:,:n_components]
    #        if isinstance(row_coord,Series):
    #            row_coord, col_coord = row_coord.to_frame(), col_coord.to_frame()
    #        U = mapply(mapply(row_coord,lambda x : x*sqrt(row_marge),axis=0,progressbar=False,n_workers=self.call_.n_workers),
    #                   lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #        V = mapply(mapply(col_coord,lambda x : x*sqrt(col_marge),axis=0,progressbar=False,n_workers=self.call_.n_workers),
    #                   lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #        S = mapply(U,lambda x : x*self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(V.T)
    #        hatX = mapply(mapply(S,lambda x : x*sqrt(row_marge),axis=0,progressbar=False,n_workers=self.call_.n_workers),
    #                      lambda x : x*sqrt(col_marge),axis=1,progressbar=False,n_workers=self.call_.n_workers).add(outer(row_marge,col_marge)).mul(self.call_.total)
    #    else:
    #        hatX = DataFrame(outer(row_marge,col_marge),index=X_original.index,columns=X_original.columns)
    #else:
        #variables factor coordinates
    #    if self.model_ in ("pca","mca"):
    #        var_coord = self.var_.coord.iloc[:,:n_components]
        
        #individuals factor coordinates
    #    ind_coord = self.ind_.coord.iloc[:,:n_components]
    #    hatX = ind_coord.dot(mapply(var_coord,lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
    #    #principal component analysis (PCA)
    #    if self.model_ == "pca":
    #        #de-standardisation
    #        hatX = mapply(hatX,lambda x : (x*self.call_.scale)+self.call_.center,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #multiple correspondence analysis (MCA)
    #    if self.model_ == "mca":
    #        #estimation of standardize data
    #        hatX = mapply(hatX.add(1),lambda x : x*self.call_.dummies.mean(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #        #disjunctive table
    #        hatX = (hatX > (self.call_.X.shape[1]/self.call_.dummies.shape[1])).astype(int)
    #    #Multiple Factor Analysis (MFA)
    #    if self.model_ == "mfa":
    #       pass

    #return hatX