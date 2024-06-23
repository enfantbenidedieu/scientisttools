# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from .text_label import text_label
from .gg_circle import gg_circle

def fviz_efa_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label=None,
                 title =None,
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 color ="black",
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title = None,
                 ind_sup = True,
                 color_sup = "blue",
                 marker_sup = "^",
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
    Visualize Exploratory Factor Analysis (EFA) - Graph of individuals
    ------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_efa_ind(self,
                    axis=[0,1],
                    x_lim=None,
                    y_lim=None,
                    x_label = None,
                    y_label=None,
                    title =None,
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    color ="black",
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
                    legend_title = None,
                    ind_sup = True,
                    color_sup = "blue",
                    marker_sup = "^",
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
    `self` : an object of class EFA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).
    
    `color` : a color for the active individuals (by default = "black").

    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `ind_sup` : a boolean to either add or not supplementary individuals (by default = True).

    `color_sup` : a color for the supplementary individuals points (by default = "blue").

    `marker_sup` :  a marker style for the supplementary individuals points (by default = "^").

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `repel` : a boolean, whether to avoid overplotting text labels or not (by default == False)

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    coord = self.ind_["coord"]

    ##### Add initial data
    coord = pd.concat((coord,self.call_["Xtot"]),axis=1)

    if isinstance(color,str):
        if color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise TypeError("'color' must me a numeric variable")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    # Using cosine and contributions
    if (isinstance(color,str) and color in coord.columns.tolist()) or (isinstance(color,np.ndarray)):
            # Add gradients colors
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],
                                                name = legend_title)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '->',"lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),size=point_size)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    ############################## Add supplementary individuals informations
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color = color_sup,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Individuals factor map - EFA"
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    p = p + pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_efa_var(self,
                 axis=[0,1],
                 title =None,
                 x_label = None,
                 y_label = None,
                 color ="black",
                 geom = ["arrow", "text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 scale = 1,
                 text_type = "text",
                 text_size = 8,
                 legend_title = None,
                 add_grid =True,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 lim_contrib = None,
                 add_circle = True,
                 color_circle = "gray",
                 arrow_angle=10,
                 arrow_length =0.1,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Exploratory Factor Analysis (EFA) - Graph of variables
    ----------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_efa_var(self,
                    axis = [0,1],
                    title = None,
                    x_label = None,
                    y_label = None,
                    color = "black",
                    geom = ["arrow", "text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    scale = 1,
                    text_type = "text",
                    text_size = 8,
                    legend_title = None,
                    add_grid = True,
                    add_hline = True,
                    add_vline = True,
                    ha = "center",
                    va = "center",
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style ="dashed",
                    lim_contrib = None,
                    add_circle = True,
                    color_circle = "gray",
                    arrow_angle = 10,
                    arrow_length = 0.1,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    ``self` : an object of class EFA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `color` : a color for the active variables (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).
    
    `scale` : a numeric specifying scale the variables coordinates (by default 1)

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `text_size` : a numeric value specifying the label size (by default = 8).

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

    `add_circle` : a boolean, whether to add or not a circle to plot.

    `color_circle` : a string specifying the color for the correlation circle (by default = "gray")

    `arrow_angle` : a numeric specifying the angle in degrees between the tail a single edge (by default = 10)

    `arrow_length` : a numeric specifying the length of the edge in "inches" (by default = 0.1)

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    coord = self.var_["coord"]*scale

    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    if isinstance(color,str):
        if color == "contrib":
            c = self.var_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(color,str) and color in ["contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                     pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if "text" in geom:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
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
        title = "Variables factor map - EFA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_efa_biplot(self,
                    axis = [0,1],
                    x_label = None,
                    y_label = None,
                    x_lim = None,
                    y_lim = None,
                    title = "EFA - Biplot",
                    marker = "o",
                    ind_text_size = 8,
                    var_text_size = 8,
                    ind_text_type = "text",
                    var_text_type = "text",
                    ind_point_size = 1.5,
                    ind_geom = ["point","text"],
                    var_geom = ["arrow","text"],
                    ind_color = "black",
                    var_color = "steelblue",
                    add_circle = False,
                    var_color_circle="gray",
                    ind_sup = True,
                    ind_color_sup = "blue",
                    ind_marker_sup = "^",
                    repel = True,
                    arrow_angle=10,
                    arrow_length =0.1,
                    add_hline = True,
                    add_vline=True,
                    add_grid = True,
                    ha = "center",
                    va = "center",
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style = "dashed",
                    ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Exploratory Factor Analysis (EFA) - Biplot of individuals and variables
    ---------------------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_efa_biplot(self,
                        axis = [0,1],
                        x_label = None,
                        y_label = None,
                        x_lim = None,
                        y_lim = None,
                        title = "EFA - Biplot",
                        marker = "o",
                        ind_text_size = 8,
                        var_text_size = 8,
                        ind_text_type = "text",
                        var_text_type = "text",
                        ind_point_size = 1.5,
                        ind_geom = ["point","text"],
                        var_geom = ["arrow","text"],
                        ind_color = "black",
                        var_color = "steelblue",
                        add_circle = False,
                        var_color_circle="gray",
                        ind_sup = True,
                        ind_color_sup = "blue",
                        ind_marker_sup = "^",
                        repel = True,
                        arrow_angle=10,
                        arrow_length =0.1,
                        add_hline = True,
                        add_vline=True,
                        add_grid = True,
                        ha = "center",
                        va = "center",
                        hline_color = "black",
                        hline_style = "dashed",
                        vline_color = "black",
                        vline_style = "dashed",
                        ggtheme=pn.theme_minimal()) 
    ```

    Parameters
    ----------
    see fviz_efa_ind, fviz_efa_var

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> 
    ```
    """
    # Check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Individuals coordinates
    ind = self.ind_["coord"].iloc[:,axis]
    ind.columns = ["x","y"]
    # variables coordinates
    var = self.var_["coord"].iloc[:,axis]
    var.columns = ["x","y"]

    # Rescale variables coordinates
    xscale = (np.max(ind["x"]) - np.min(ind["x"]))/(np.max(var["x"]) - np.min(var["x"]))
    yscale = (np.max(ind["y"]) - np.min(ind["y"]))/(np.max(var["y"]) - np.min(var["y"]))
    rscale = min(xscale, yscale)

    #### Extract individuals coordinates
    ind_coord = self.ind_["coord"]

    # Variables coordinates
    var_coord = self.var_["coord"]*rscale

    p = pn.ggplot()

    #####################################################################################################################################################
    # Individuals Informations
    #####################################################################################################################################################

    if "point" in ind_geom:
        p = p + pn.geom_point(data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),
                                color=ind_color,shape=marker,size=ind_point_size,show_legend=False)
    if "text" in ind_geom:
        if repel :
            p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_coord.index),
                                color=ind_color,size=ind_text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': ind_color,'lw':1.0}})
        else:
            p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_coord.index),
                                color=ind_color,size=ind_text_size,va=va,ha=ha)
    
    # Add supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_["coord"]
            if "point" in ind_geom:
                p = p + pn.geom_point(ind_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                      color = ind_color_sup,shape = ind_marker_sup,size=ind_point_size)
            if "text" in ind_geom:
                if repel:
                    p = p + text_label(ind_text_type,data=ind_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                        color=ind_color_sup,size=ind_text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': ind_color_sup,'lw':1.0}})
                else:
                    p = p + text_label(ind_text_type,data=ind_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                        color = ind_color_sup,size=ind_text_size,va=va,ha=ha)
      
    #########################################################################################################################################################
    #   Variables informations
    ##########################################################################################################################################################

    if "arrow" in var_geom:
            p = p + pn.geom_segment(data=var_coord,mapping=pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=var_color)
    if "text" in var_geom:
        p = p + text_label(var_text_type,data=var_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_coord.index),
                           color=var_color,size=var_text_size,va=va,ha=ha)
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=var_color_circle, fill=None)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set title
    if title is None:
        title = "EFA - Biplot"
    p = p + pn.labs(title=title,x=x_label,y = y_label)
    
    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0,colour=hline_color,linetype =hline_style)    
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0,colour=vline_color,linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    # Add theme
    p = p + ggtheme
    
    return p

def fviz_efa(self,choice="ind",**kwargs)->pn:
    """
    Visualize Exploratory Factor Analysis (EFA)
    -------------------------------------------

    Description
    -----------
    Exploratory factor analysis is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. It is used to identify the structure of the relationship between the variable and the respondent. fviz_efa() provides plotnine-based elegant visualization of EFA outputs
    
        * fviz_efa_ind(): Graph of individuals
        * fviz_efa_var(): Graph of variables
        * fviz_efa_biplot() : Biplot of individuals and variables

    Usage
    -----
    ```python
    >>> fviz_efa(self,choice=("ind","var","biplot"))
    ```

    Parameters
    ----------
    `self` : an object of class EFA

    `choice` : the element to subset. Allowed values are : 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs (correlation circle)
        * 'biplot' for biplot of individuals and variables
    
    `**kwargs` : further arguments passed to or from other methods

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> 
    ```
    """
    # check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")

    if choice not in ["ind","var","biplot"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'biplot'")

    if choice == "ind":
        return fviz_efa_ind(self,**kwargs)
    elif choice == "var":
        return fviz_efa_var(self,**kwargs)
    elif choice == "biplot":
        return fviz_efa_biplot(self,**kwargs)