# -*- coding: utf-8 -*-

import pandas as pd

def get_mds(self) -> dict:

    """
    self : an object of class MDS

    Returns
    -------
    Multidimensional Scaling - Results 
    ===============================================================
        Names       Description
    1   "coord"     "coordinates"
    2   "res.dist"  "Restitues distances"
    """
    if self.model_ not in ["mds","cmds"]:
        raise ValueError("Error : 'res' must be an object of class MDS or CMDS.")

    # Store informations
    df = dict({
        "coord"     : pd.DataFrame(self.coord_,index=self.labels_,columns=self.dim_index_),
        "res.dist"  : pd.DataFrame(self.res_dist_,index=self.labels_,columns=self.labels_)
    })
    return df



