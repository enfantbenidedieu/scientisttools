# -*- coding: utf-8 -*-

import plotnine as pn
import pandas as pd
from scientisttools.extractfactor import get_hclust,get_melt,get_dist
from scipy.spatial.distance import squareform


def fviz_dist(X,metric="euclidean",order = True, show_labels = True, lab_size = None,        
              gradient = dict({"low" : "red", "mid" : "white", "high" : "blue"}),**kwargs):

    dist = get_dist(X,method=metric,**kwargs)
    res_dist = pd.DataFrame(squareform(dist["dist"]),index=dist["labels"],columns=dist["labels"])
        
    if order:
        res_hclust = get_hclust(dist["dist"],method='ward')
        res_mat = res_dist.iloc[res_hclust["order"],res_hclust["order"]]
    else:
        res_mat = res_dist
    
    d = get_melt(res_mat)
    p = pn.ggplot(d,pn.aes(x= "Var1",y="Var2"))+pn.geom_tile(pn.aes(fill="value"))
    if gradient["mid"] is None:
        p = p + pn.scale_fill_gradient(
            low = gradient["low"],
            high = gradient["high"]
        )
    else:
        p = p + pn.scale_fill_gradient2(
            midpoint = res_mat.mean().mean(),
            low = gradient["low"],
            mid = gradient["mid"],
            high = gradient["high"]
        )
    if show_labels:
        p = p + pn.theme(
            axis_title_x=pn.element_blank(),
            axis_title_y=pn.element_blank(),
            axis_text_x=pn.element_text(
                angle = 45,
                ha="center",
                size=lab_size
            )
        )
    else:
        p = p + pn.theme(
            axis_text=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_title_x=pn.element_blank(),
            axis_title_y=pn.element_blank()
        )
    return p
        

        


