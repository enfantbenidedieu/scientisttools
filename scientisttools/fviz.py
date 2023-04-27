# -*- coding: utf-8 -*-

import plotnine as pn

def fviz(res, axes =  [0,1], geom = ["point", "text"],geom_ind = ["point", "text"], 
                repel = False,habillage="none", palette = None, addEllipses=False, 
                col_ind = "black", fill_ind = "white", col_ind_sup = "blue", alpha_ind =1,
                select_ind = dict({"name" :None, "cos2" :None, "contrib": None}),**kwargs):
    if len(axes) != 2:
        raise ValueError("")
    
    if res.model_ not in ["pca","ca","mca","famd","mfa"]:
        raise ValueError("'res' must be an object of class PCA, CA, MCA, FAMD and MFA")
    
    raise NotImplementedError("Error : This method is not implemented yet.")

    