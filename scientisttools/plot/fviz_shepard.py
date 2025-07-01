# -*- coding: utf-8 -*-
import numpy as np
import plotnine as pn
import pandas as pd

# Shepard Diagram
# see https://github.com/Mthrun/DataVisualizations/blob/master/R/Sheparddiagram.R
def fviz_shepard(self,
                 x_lim=None,
                 y_lim=None,
                 x_label=None,
                 y_label=None,
                 color="black",
                 title=None,
                 add_grid=True,
                 ggtheme=pn.theme_minimal())-> pn:
    """
    Draws a Shepard Diagram
    -----------------------

    Description
    -----------

    This function plots a Shepard diagram which is a scatter plot of InputDist and OutputDist

    Parameters
    ----------
    self : an object of class MDS, CMDSCALE


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ not in ["cmdscale","mds"]:
        raise TypeError("'Method' is allowed only for multidimensional scaling.")
    
    dist, res_dist = self.result_["dist"].values, self.result_["res_dist"].values
    #
    coord = pd.DataFrame({"InDist": dist[np.triu_indices(dist.shape[0], k = 1)],
                          "OutDist": res_dist[np.triu_indices(dist.shape[0], k = 1)]})
    
    p = pn.ggplot(coord,pn.aes(x = "InDist",y = "OutDist"))+pn.geom_point(color=color)+pn.geom_line(linetype = "dashed")

    if x_label is None:
        x_label = "Input Distances"
    if y_label is None:
        y_label = "Output Distances"
    if title is None:
        title = "Shepard Diagram"
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)

    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    return p+ ggtheme