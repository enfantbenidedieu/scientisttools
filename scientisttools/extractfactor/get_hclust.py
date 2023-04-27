
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.cluster import hierarchy

def get_hclust(X, method='single', metric='euclidean', optimal_ordering=False):
    Z = hierarchy.linkage(X,method=method, metric=metric)
    if optimal_ordering:
        order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z,X))
    else:
        order = hierarchy.leaves_list(Z)
    return dict({"order":order,"height":Z[:,2],"method":method,
                "merge":Z[:,:2],"n_obs":Z[:,3],"data":X})
    



