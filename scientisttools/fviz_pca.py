# -*- coding: utf-8 -*-

import plotnine as pn
import numpy as np
import pandas as pd
from scientisttools.extractfactor import get_pca_ind,get_pca_var


def fviz_pca_ind(res, axes =  [0,1], geom = ["point", "text"],
xlim =None, ylim=None,
geom_ind = ["point", "text"],label = ["none","ind", "ind.sup"],
invisible = ["none","ind", "ind.sup"],
title = None, auto_lab = ["auto","yes","non"],new_plot=False,
select = None, unselect = 0.7,show_text=False,
legend = dict({"bty":"y","x":"topleft"}),
col_hab = None,repel = False,habillage="none", palette = None, addEllipses=False, 
                col_ind = "black", fill_ind = "white", col_ind_sup = "blue", alpha_ind =1,
                select_ind = dict({"name" :None, "cos2" :None, "contrib": None}),
                ggtheme = pn.theme_minimal(),**kwargs):
    
    ind_infos = get_pca_ind(res)
    var_infos = get_pca_var(res)

    ggoptions_default =  dict({"size" : 4, 
    "point_shape" : 19, "line_lty" : 2, "line_lwd" : 0.5, 
    "line_color" : "black", "segment_lty" : 1, 
    "segment_lwd" : 0.5, "circle_lty" : 1, 
    "circle_lwd" : 0.5, "circle_color" : "black", 
    "low.col_quanti" :"blue", "high_col_quanti" : "red3"})

    if isinstance(unselect,float):
        if ((unselect >1) or (unselect < 0)):
            raise ValueError("unselect should be betwwen 0 and 1")

    if auto_lab == "yes":
        auto_lab = True
    elif auto_lab == "non":
        auto_lab = False
    #else:
    #    raise ValueError("Allowed values")
    
    if palette is None:
        palette = ["black", "red", "green3", "blue", "magenta", "darkgoldenrod","darkgray", 
                    "orange", "cyan", "violet", "lightpink", "lavender", "yellow", "darkgreen",
                    "turquoise", "lightgrey", "lightblue", "darkkhaki","darkmagenta","lightgreen", 
                    "darkolivegreen", "lightcyan", "darkorange","darkorchid", "darkred", "darksalmon", 
                    "darkseagreen","darkslateblue", "darkslategray", "darkslategrey","darkturquoise", 
                    "darkviolet", "lightgray", "lightsalmon","lightyellow", "maroon"]

    if "none" in invisible:
        invisible = None

    lab_ind = lab_ind_sup = False
    if ((len(label) == 1) and label == "all"):
        lab_ind = lab_ind_sup = True
    if "ind" in label:
        lab_ind = True
    if "ind_sup" in label :
        lab_ind_sup = True
    
    if title is None:
        title = "PCA graph of individuals"

    lab_x = list(["Dim."+str(axes[0]+1)+" ("+str(round(res.eig_[2][axes[0]],2))+"%)"])
    lab_x = list(["Dim."+str(axes[1]+1)+" ("+str(round(res.eig_[2][axes[1]],2))+"%)"])
    
    if col_hab is not None:
        palette = col_hab
        theme = pn.theme(
            axis_title=pn.element_text(ha="center",size=11),
            plot_title=pn.element_text(ha="center",size=11),
            legend_position = legend["x"] 
        )
    
    row_coord = ind_infos["coord"]
    row_sup_coord = None
    quali_sup_coord = None
    ellipse_coord = None

    if res.row_sup_labels_ is not None:
        row_sup_coord = ind_infos["ind_sup"]["coord"]
    if res.quali_sup_labels_ is not None:
        quali_sup_coord = var_infos["quali_sup"]["coord"]
    
    text_invisible = list([False,False])
    if invisible is not None:
        text_invisible[0] =invisible.index('ind')
        text_invisible[1] =invisible.index('ind_sup')
        text_invisible[2] =invisible.index('quali')
    else:
        text_invisible = np.repeat(np.nan,3)
    
    if xlim is None:
        xmin = xmax = 0
        if text_invisible[0] == np.nan:
            xmin = min(xmin, row_coord[row_coord.columns[0]].min())
            xmax = max(xmax, row_coord[row_coord.columns[0]].max())
        if ((row_sup_coord is not None) and (text_invisible[1]== np.nan)):
            xmin = min(xmin, row_sup_coord[row_sup_coord.columns[0]].min())
            xmax = max(xmax, row_sup_coord[row_sup_coord.columns[0]].max())
        if ((quali_sup_coord is not None) and (text_invisible[2]== np.nan)):
            xmin = min(xmin, quali_sup_coord[quali_sup_coord.columns[0]].min())
            xmax = max(xmax, quali_sup_coord[quali_sup_coord.columns[0]].max())
        xlim = list([xmin,xmax])
        xlim = (xlim - np.mean(xlim))*1.2 + np.mean(xlim)

    p = pn.ggplot(row_coord,pn.aes(x="Dim."+str(axes[0]+1),y="Dim."+str(axes[1]+1)))+\
        pn.geom_point(shape="o",color=col_ind) 
    
    p = p + ggtheme
    return p

def fviz_pca_var(res):
    pass

def fviz_pca(res,choice="ind",**kwargs):
    pass
