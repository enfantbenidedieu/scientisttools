# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

def auto_cut_tree(self,min_clust,max_clust,metric,method,order,weights=None):
    """
    Automatic tree cut
    ------------------

    Description
    -----------
    Automatic tree cut to determine optimal number of clusters.

    Usage
    -----
    ```python
    >>> auto_cut_tree(self,min_clust,max_clust,metric,method,order,weights=None)
    ```

    Parameters
    ----------
    `self` : an object of class PCA, MCA, SpecificMCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX, MFACT

    `min_clust` : an integer specifying the least possible number of clusters suggested

    `max_clust` : an integer specifying the higher possible number of clusters suggested

    `metric` : a string specifying the metric used to build the tree. See https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

    `method` : a string specifying the method used to build the tree. See https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

    `order` : a boolean. If True, clusters are ordered following their center coordinate on the first axis.

    `weights` : weights for each observation, with same length as zero axis of data.

    Return
    ------
    `nb_clust` : number of cluster

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if self is an object of class
    if self.model_ not in ["pca","mca","specificmca","famd","pcamix","mpca","mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an object of class PCA, MCA, SpecificMCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX, MFACT")

    if order:
        data = pd.concat((self.ind_["coord"],self.call_["X"],self.call_["ind_weights"]),axis=1)
        if weights is not None:
            weights = weights[::-1]
        data = data.sort_values(by=data.columns.tolist()[0],ascending=True)
        self.ind_["coord"] = data.iloc[:,:self.ind_["coord"].shape[1]]
        self.call_["X"] = data.iloc[:,(self.ind_["coord"].shape[1]+1):(data.shape[1]-1)]
        self.call_["ind_weights"] = data.iloc[:,-1]
    
    # Extract
    X = self.ind_["coord"]
    # Dissimilarity matrix
    do = pdist(X,metric=metric)**2
    # Set weights
    if weights is None:
        weights = np.ones(X.shape[0])

    # Effec
    eff = np.zeros(shape=(len(weights),len(weights)))
    for i in range(len(weights)):
        for j in range(len(weights)):
            eff[i,j] = (weights[i]*weights[j])/(sum(weights))/(weights[i]+weights[j])
    dissi = do*eff[np.triu_indices(eff.shape[0], k = 1)]
    # Agglometrive clustering
    link_matrix = hierarchy.linkage(dissi,metric=metric,method=method)
    inertia_gain = link_matrix[:,2][::-1]
    intra = np.cumsum(inertia_gain[::-1])[::-1]
    quot = intra[(min_clust-1):max_clust]/intra[(min_clust-2):(max_clust-1)]
    nb_clust = (np.argmin(quot)+1) + min_clust - 1
    return nb_clust