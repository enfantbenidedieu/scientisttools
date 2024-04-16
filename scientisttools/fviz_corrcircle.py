# -*- coding: utf-8 -*-
import plotnine as pn
import numpy as np

from .text_label import text_label
from .gg_circle import gg_circle

def fviz_corrcircle(self,
                    axis=[0,1],
                    x_label=None,
                    y_label=None,
                    title=None,
                    geom = ["arrow","text"],
                    color = "black",
                    color_sup = "blue",
                    text_type = "text",
                    arrow_length=0.1,
                    text_size=8,
                    arrow_angle=10,
                    add_circle=True,
                    color_circle = "gray",
                    add_hline=True,
                    add_vline=True,
                    add_grid=True,
                    ggtheme=pn.theme_minimal()) -> pn:
    """
    Draw correlation circle
    -----------------------

    Description
    -----------


    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if self.model_ not in ["pca","ca","mca","famd","mfa","mfaqual","mfamix","partialpca","efa"]:
        raise ValueError("Factor method not allowed.")
    
    if self.model_ in ["pca","partialpca","efa"]:
        coord = self.var_["coord"]
    elif self.model_ in ["famd","mfa","mfamix"]:
        coord = self.quanti_var_["coord"]
    else:
        if hasattr(self, "quanti_sup_"):
            coord = self.quanti_sup_["coord"]
        if hasattr(self, "quanti_var_sup_"):
            coord = self.quanti_var_sup_["coord"]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "arrow" in geom:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
    if "text" in geom:
            p = p + text_label(text_type,color=color,size=text_size,va="center",ha="center")
        
    if self.model_ in ["pca","famd","mfa","mfamix"]:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_["coord"]
            if "arrow" in geom:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                     arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype="--")
            if "text" in geom:
                p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va="center",ha="center")
        elif hasattr(self, "quanti_var_sup_"):
            sup_coord = self.quanti_var_sup_["coord"]
            if "arrow" in geom:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                     arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype="--")
            if "text" in geom:
                p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va="center",ha="center")
    
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=color_circle, fill=None)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Correlation circle"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour="black", linetype ="dashed")
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour="black", linetype ="dashed")
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p