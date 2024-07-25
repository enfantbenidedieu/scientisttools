# -*- coding: utf-8 -*-
import plotnine as pn
from .text_label import text_label

def fviz_mds(self,
            axis=[0,1],
            x_label=None,
            y_label=None,
            x_lim=None,
            y_lim=None,
            geom = ["point","text"],
            text_type = "text",
            point_size = 1.5,
            text_size = 8,
            title =None,
            color="black",
            color_sup ="blue",
            marker = "o",
            marker_sup = "^",
            ind_sup = True,
            add_grid =True,
            add_hline = True,
            add_vline=True,
            ha="center",
            va="center",
            hline_color="black",
            hline_style="dashed",
            vline_color="black",
            vline_style ="dashed",
            repel=False,
            ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize the results for Multidimension Scaling (MDS)
    ------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_mds(self,
                axis=[0,1],
                x_label=None,
                y_label=None,
                x_lim=None,
                y_lim=None,
                geom = ["point","text"],
                text_type = "text",
                point_size = 1.5,
                text_size = 8,
                title =None,
                color="black",
                color_sup ="blue",
                marker = "o",
                marker_sup = "^",
                ind_sup = True,
                add_grid =True,
                add_hline = True,
                add_vline=True,
                ha="center",
                va="center",
                hline_color="black",
                hline_style="dashed",
                vline_color="black",
                vline_style ="dashed",
                repel=False,
                ggtheme=pn.theme_minimal()) 
    ```

    Parameters
    ----------
    see fviz_pca_ind

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if self is an object of class MDS
    if self.model_ != "mds":
        raise TypeError("'self' must be an instance of class MDS")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.call_["n_components"]-1)  or
            (axis[0] > axis[1])) :
            raise ValueError("You must pass a valid axis")
    
    coord = self.result_["coord"]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    # Add point
    if "point" in geom:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                        adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if ind_sup:
        if hasattr(self, "sup_coord_"):
            sup_coord = self.sup_coord_
            if "point" in geom:
                p = p + pn.geom_point(data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=point_size,shape=marker_sup)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    # Set title
    if title is None:
        title = self.call_["title"]
    p = p + pn.ggtitle(title)
    # Set x label
    if x_label is not None:
        p = p + pn.xlab(xlab=x_label)
    # Set y label
    if y_label is not None:
        p = p + pn.ylab(ylab=y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim:
        p = p + pn.ylim(y_lim)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p + pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    
    p = p + ggtheme
    return p