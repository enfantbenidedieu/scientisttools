# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from .text_label import text_label
from .gg_circle import gg_circle

def plot_dendrogram(self,**kwargs) -> plt:
    """
    Visualization of Dendrogram
    ---------------------------

    Description
    -----------
    Draws easily beautiful dendrograms using matplotlib

    Parameters
    ----------
    self : an object of class HCPC, VARHCA, CATVARHCA

    **kwargs : additional element from scipy.cluster.hierarchy.dendrogram.
                see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

    Return
    ------
    figure

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ not in ["hcpc","varhca","catvarhca","varhcpc"]:
        raise TypeError("'self' must be an object of class HCPC, VARHCA, CATVARHCA, VARHCPC")

    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    if self.model_ == "hcpc":
        labels = self.call_["model"].ind_["coord"].index
    elif self.model_ == "varhca":
        labels = self.call_["tree"]["data"].index
    elif self.model_ == "catvarhca":
        labels = self.cluster_["data_clust"].index
    elif self.model_ == "varhcpc":
        labels = self.call_["X"].index

    ddata = dendrogram(self.call_["tree"]["linkage"],labels=labels,**kwargs)
    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'],ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y,'o',c=c);
                plt.annotate("%.3g" % y,(x,y),xytext=(0, -5),textcoords='offset points',va='top', ha='center');
        if max_d:
          plt.axhline(y=max_d, c = "k");

def fviz_hcpc_cluster(self,
                      axis=(0,1),
                      x_lim=None,
                      y_lim=None,
                      x_label=None,
                      y_label=None,
                      title=None,
                      legend_title = None,
                      geom = ["point","text"],
                      show_clust_cent = False, 
                      cluster = None,
                      center_marker_size=5,
                      repel=True,
                      ha = "center",
                      va = "center",
                      point_size = 1.5,
                      text_size = 8,
                      text_type = "text",
                      add_grid=True,
                      add_hline=True,
                      add_vline=True,
                      hline_color="black",
                      vline_color="black",
                      hline_style = "dashed",
                      vline_style = "dashed",
                      add_ellipse=False,
                      ellipse_type = "t",
                      confint_level = 0.95,
                      geom_ellipse = "polygon",
                      ggtheme = pn.theme_minimal()) -> pn:
    """
    Visualize Hierarchical Clustering on Principal Components Results
    -----------------------------------------------------------------

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "hcpc":
        raise TypeError("'self' must be an object of class HCPC")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["model"].call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    if legend_title is None:
        legend_title = "Cluster"

    #### Extract data
    coord = self.call_["tree"]["data"]

    if cluster is None:
        coord = pd.concat([coord,self.cluster_["cluster"]],axis=1)
    else:
        coord = pd.concat([coord,cluster],axis=1)
        
    # Rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]
    
    # Transform to str
    coord[coord.columns[-1]] = coord[coord.columns[-1]].astype("str")

     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist(),color=legend_title))
    
    if "point" in geom:
        p  = p + pn.geom_point(size=point_size) + pn.guides(color=pn.guide_legend(title=legend_title))
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=legend_title),size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=legend_title),size=text_size,va=va,ha=ha)
    if add_ellipse:
        p = p + pn.stat_ellipse(geom=geom_ellipse,type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if show_clust_cent:
        cluster_center = self.cluster_["coord"]
        if "point" in geom:
            p = p + pn.geom_point(cluster_center,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=cluster_center.index,color=cluster_center.index),size=center_marker_size)
    
    # Add additionnal        
    proportion = self.call_["model"].eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set title
    if title is None:
        title = "Individuals factor map - HCPC"
    p = p + pn.labs(title=title,x=x_label,y = y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    
    p = p + ggtheme

    return p


def fviz_varhcpc_cluster(self,
                      axis=(0,1),
                      x_lim=None,
                      y_lim=None,
                      x_label=None,
                      y_label=None,
                      title=None,
                      legend_title = None,
                      geom = ["point","text"],
                      show_clust_cent = False, 
                      cluster = None,
                      center_marker_size=5,
                      repel=True,
                      ha = "center",
                      va = "center",
                      point_size = 1.5,
                      text_size = 8,
                      text_type = "text",
                      add_grid=True,
                      add_hline=True,
                      add_vline=True,
                      hline_color="black",
                      vline_color="black",
                      hline_style = "dashed",
                      vline_style = "dashed",
                      color_circle="gray",
                      arrow_angle=10,
                      arrow_length =0.1,
                      add_ellipse=False,
                      ellipse_type = "t",
                      confint_level = 0.95,
                      geom_ellipse = "polygon",
                      ggtheme = pn.theme_minimal()) -> pn:
    """
    Visualize Variables Hierarchical Clustering on Principal Components Results
    ---------------------------------------------------------------------------

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "varhcpc":
        raise TypeError("'self' must be an object of class VARHCPC")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["model"].call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    ###### Off somme components
    if self.call_["model"].model_ == "pca":
        if isinstance(geom,list):
            geom = [x for x in geom if x != "point"]
        elif isinstance(geom,str):
            if geom == "point":
                geom = "arrow"
    if self.call_["model"].model_ == "mca":
        add_ellipse = False
        if isinstance(geom,list):
            geom = [x for x in geom if x != "arrow"]
        elif isinstance(geom,str):
            if geom == "arrow":
                geom = "point"
    
    if legend_title is None:
        legend_title = "Cluster"

    #### Extract data
    coord = self.call_["tree"]["data"]

    if cluster is None:
        coord = pd.concat([coord,self.cluster_["cluster"]],axis=1)
    else:
        coord = pd.concat([coord,cluster],axis=1)
        
    # Rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]
    
    # Transform to str
    coord[coord.columns[-1]] = coord[coord.columns[-1]].astype("str")

     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist(),color=legend_title))
    
    ##### Set for MCA
    if self.call_["model"].model_ == "mca":
        if "point" in geom:
            p  = p + pn.geom_point(size=point_size) + pn.guides(color=pn.guide_legend(title=legend_title))
    elif self.call_["model"].model_ == "pca":
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                     arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                     pn.guides(color=pn.guide_legend(title=legend_title)))
    
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=legend_title),size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=legend_title),size=text_size,va=va,ha=ha)
    if add_ellipse:
        p = p + pn.stat_ellipse(geom=geom_ellipse,type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if show_clust_cent:
        cluster_center = self.cluster_["coord"]
        if self.call_["model"].model_ == "mca":
            if "point" in geom:
                p = p + pn.geom_point(cluster_center,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=cluster_center.index,color=cluster_center.index),size=center_marker_size)
        if self.call_["model"].model_ == "pca":
            if "arrow" in geom:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(cluster_center.iloc[:,axis[0]]),yend=np.asarray(cluster_center.iloc[:,axis[1]]),
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))

    # Create circle
    if self.call_["model"].model_ == "pca":
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=color_circle, fill=None)

    # Add additionnal        
    proportion = self.call_["model"].eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set title
    if title is None:
        title = "Variables factor map - VARHCPC"
    p = p + pn.labs(title=title,x=x_label,y = y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    
    p = p + ggtheme

    return p
