# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.cluster.hierarchy import ward, fcluster

"""
def switch(subject,X,element):
    match subject:
        case "pca":
            return get_pca(X,element)
        case "ca":
            return get_ca(X,element)
        case "mca":
            return get_mca(X,element)
"""

def factor_summary(self,X,element,group_names,node_level=1,
                    result=list(["coord","cos2","contrib"]),axes=range(2),select=None):
    allowed_elements = list(["row", "col", "var", "ind", "quanti_var", "quali_var",
                     "mca_cor", "quanti_sup",  "group", "partial_axes", "partial_node"])

    if element not in allowed_elements:
        raise ValueError(f"Can't handle element = '{element}'")
    if element in ["mca_cor","quanti_sup"]:
        if self.model_ != "mca":
            raise ValueError("element = 'mca_cor' is supported only for FactoMineR::MCA().")
        result = None
    element = element[0]
    # elmt = switch(self.model_,X,element)
    raise NotImplementedError("Error : This method is not implemented yet.")
    
