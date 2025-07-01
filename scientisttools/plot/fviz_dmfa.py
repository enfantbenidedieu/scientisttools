# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from .colors import list_colors
from .text_label import text_label
from .gg_circle import gg_circle

def fviz_dmfa_ind(self,
                 axis = [0,1],
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 geom = ["point","text"],
                 palette = None,
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 add_grid =True,
                 add_ellipses = False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 quali_sup = True,
                 color_quali_sup = "red",
                 marker_quali_sup = ">",
                 add_hline = True,
                 add_vline = True,
                 hline_color = "black",
                 hline_style = "dashed",
                 vline_color = "black",
                 vline_style = "dashed",
                 ha = "center",
                 va = "center",
                 lim_cos2 = None,
                 lim_contrib = None,
                 repel = False,
                 ggtheme = pn.theme_minimal()) -> pn:
    """
    Visualize Dual Multiple Factor Analysis (DMFA) - Graph of individuals
    ---------------------------------------------------------------------

    Description
    -----------
    Performs Dual Multiple Factor Analysis (DMFA) with supplementary quantitative variables and supplementary categorical variables. fviz_dmfa_ind provides plotnine based elegant visualization of DMFA outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_dmfa_ind(self,
                      axis=[0,1],
                      x_lim = None,
                      y_lim = None,
                      x_label = None,
                      y_label = None,
                      title = None, 
                      geom = ["point","text"],
                      gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                      point_size = 1.5,
                      text_size = 8,
                      text_type = "text",
                      add_grid = True,
                      add_ellipses=False, 
                      ellipse_type = "t",
                      confint_level = 0.95,
                      geom_ellipse = "polygon",
                      quali_sup = True,
                      color_quali_sup = "red",
                      marker_quali_sup = ">",
                      add_hline = True,
                      add_vline = True,
                      hline_color = "black",
                      hline_style = "dashed",
                      vline_color = "black",
                      vline_style = "dashed",
                      ha = "center",
                      va = "center",
                      lim_cos2 = None,
                      lim_contrib = None,
                      repel = False,
                      ggtheme = pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".
    
    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_ellipses` : a boolean to either add or not ellipses (by default = False). 

    `ellipse_type` : ellipse multivariate distribution (by default = "t" for t-distribution). However, you can set type = "norm" to assume a multivariate normal distribution or type = "euclid" for an euclidean ellipse.

    `confint_level` : ellipse confindence level (by default = 0.95).

    `geom_ellipse` : ellipse geometry (by default = "polygon").

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `lim_cos2` : a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

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
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_ind
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Graph of individuals
    >>> p = fviz_pca_ind(res_pca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_.n_components-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Set palette
    index = list(self.group_.name)
    if palette is None:
        palette = [x for x in list_colors if x != color_quali_sup][:len(index)]
    elif not isinstance(palette,(list,tuple)):
        raise TypeError("'palette' must be a list or a tuple of colors")
    elif len(palette) != len(index):
        raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    # Extract individuals coordinates
    coord = self.ind_.coord

    # Add Active Data
    num_fact_label = self.call_.num_fact
    coord = pd.concat([coord,self.call_.Xtot.loc[:,num_fact_label]],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_.cos2.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "point" in geom:
        p = p + pn.geom_point(pn.aes(color = num_fact_label,linetype = num_fact_label),size=point_size)
    if "text" in geom:
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(color=num_fact_label),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=num_fact_label),size=text_size,va=va,ha=ha)
    
    # Add ellipse
    if add_ellipses:
        p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=num_fact_label),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    # scale color manual
    p = p + pn.scale_color_manual(values=palette)
    
    # Add supplementary qualitatives/categories
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_.coord
            if "point" in geom:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                      color=color_quali_sup,shape = marker_quali_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                        color=color_quali_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,'lw':1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                        color =color_quali_sup,size=text_size,va=va,ha=ha)

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
        title = "Individuals Factor Map - DMFA"
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

# Variables Factor Map
def fviz_dmfa_var(self,
                 axis=[0,1],
                 x_label = None,
                 y_label = None,
                 title =None,
                 geom = ["arrow","text"],
                 palette = None,
                 scale = 1,
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "black",
                 linestyle_sup="dashed",
                 add_hline = True,
                 add_vline = True,
                 hline_color = "black",
                 hline_style = "dashed",
                 vline_color = "black",
                 vline_style ="dashed",
                 ha = "center",
                 va = "center",
                 add_circle = True,
                 color_circle = "gray",
                 arrow_angle = 10,
                 arrow_length = 0.1,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Dual Multiple Factor Analysis (DMFA) - Graph of variables
    -------------------------------------------------------------------

    Description
    -----------
    Performs Dual Multiple Factor Analysis (DMFA) with supplementary quantitative variables and supplementary categorical variables. fviz_dmfa_var provides plotnine based elegant visualization of DMFA outputs for variables.

    Usage
    -----
    ```python
    >>> fviz_dmfa_var(self,
                    axis = [0,1],
                    x_label = None,
                    y_label = None,
                    title = None,
                    geom = ["arrow","text"],
                    scale = 1,
                    text_type = "text",
                    text_size = 8,
                    add_grid = True,
                    quanti_sup=True,
                    color_sup = "black",
                    linestyle_sup="dashed",
                    add_hline = True,
                    add_vline = True,
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style ="dashed",
                    ha = "center",
                    va = "center",
                    add_circle = True,
                    color_circle = "gray",
                    arrow_angle = 10,
                    arrow_length = 0.1,
                    lim_cos2 = None,
                    lim_contrib = None,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `scale` : a numeric specifying scale the variables coordinates (by default 1)
    
    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `text_size` : a numeric value specifying the label size (by default = 8).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `quanti_sup` : a boolean to either add or not supplementary quantitatives variables (by default = True).

    `color_sup` : a color for the supplementary quantitatives variables (by default = "black").

    `linestyle_sup` : a string specifying the supplementary variables line style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `add_circle` : a boolean, whether to add or not a circle to plot.

    `color_circle` : a string specifying the color for the correlation circle (by default = "gray")

    `arrow_angle` : a numeric specifying the angle in degrees between the tail a single edge (by default = 10)

    `arrow_length` : a numeric specifying the length of the edge in "inches" (by default = 0.1)

    `lim_cos2` : a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

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
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_var
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Graph of variables
    >>> p = fviz_pca_var(res_pca)
    >>> print(p)
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_.n_components-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Set palette
    index = list(self.group_.name)
    if palette is None:
        palette = [x for x in list_colors if x != color_sup][:(len(index)+1)]
    elif not isinstance(palette,(list,tuple)):
        raise TypeError("'palette' must be a list or a tuple of colors")
    elif len(palette) != (len(index)+1):
        raise TypeError(f"'palette' must be a list or tuple with length {len(index)+1}.")

    coord = self.var_.coord.mul(scale)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_.cos2.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")
        
    # insert group
    coord.insert(0,"group","var")

    group_name = list(self.var_partiel_._fields)
    for i,k in enumerate(group_name):
        var_p = self.var_partiel_[i].mul(scale)
        var_p.insert(0,"group",k) 
        coord = pd.concat((coord,var_p),axis=0)
    
    # convert to categorical
    coord["group"] = pd.Categorical(coord["group"],categories=group_name+["var"],ordered=True)
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color="group",label=coord.index))

    if "arrow" in geom:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color="group"), arrow = pn.arrow(length=arrow_length,angle=arrow_angle))
    if "text" in geom:
        p = p + text_label(text_type,mapping=pn.aes(color="group"),size=text_size,va=va,ha=ha)

    # set color manual
    p = p + pn.scale_color_manual(values=palette)
    
    # Add supplmentary continuous variables
    if quanti_sup:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_.coord.mul(scale)
            if "arrow" in geom:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
            if "text" in geom:
                p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),color=color_sup,size=text_size,va=va,ha=ha)
    
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
        title = "Variables Factor Map - DMFA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

# Groups Factor Map
def fviz_dmfa_group(self,
                    choice = "absolute",
                    axis=[0,1],
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None, 
                    title =None,
                    color ="black",
                    geom = ["point","text"],
                    point_size = 1.5,
                    text_type = "text",
                    text_size = 8,
                    marker = "o",
                    add_grid =True,
                    add_hline = True,
                    add_vline = True,
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style ="dashed",
                    ha = "center",
                    va = "center",
                    repel = False,
                    ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Dual Multiple Factor Analysis (DMFA) - Graph of groups
    -------------------------------------------------------------------

    Description
    -----------
    Performs Dual Multiple Factor Analysis (DMFA) with supplementary quantitative variables and supplementary categorical variables. fviz_dmfa_group provides plotnine based elegant visualization of DMFA outputs for groups.

    Usage
    -----
    ```python
    >>> fviz_dmfa_group(self,
                        choice = "absolute",
                        axis = [0,1],
                        x_lim = None,
                        y_lim = None,
                        x_label = None,
                        y_label = None,
                        title = None,
                        color = "black",
                        geom = ["arrow","text"],
                        point_size = 1.5,
                        text_type = "text",
                        text_size = 8,
                        marker = "o",
                        add_grid = True,
                        add_hline = True,
                        add_vline = True,
                        hline_color = "black",
                        hline_style = "dashed",
                        vline_color = "black",
                        vline_style ="dashed",
                        ha = "center",
                        va = "center",
                        ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    `choice` : a text specifying the data to be plotted. Allowed values are "absolute" or "normalized"

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active individuals points (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").

    `text_size` : a numeric value specifying the label size (by default = 8).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

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
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_var
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Graph of variables
    >>> p = fviz_pca_var(res_pca)
    >>> print(p)
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_.n_components-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    if choice == "absolute":
        coord = self.group_.coord
    elif choice == "normalized":
        coord = self.group_.coord_n
    else:
        raise ValueError("'choice' must be either 'absolute' or 'normalized'")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "point" in geom:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Groups Factor Map - DMFA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)

    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

# Qualit Factor Map
def fviz_dmfa_quali_sup(self,
                        axis=[0,1],
                        x_lim = None,
                        y_lim = None,
                        x_label = None,
                        y_label = None, 
                        title =None,
                        color ="black",
                        geom = ["point","text"],
                        palette = None,
                        point_size = 1.5,
                        text_type = "text",
                        text_size = 8,
                        marker = "o",
                        add_grid =True,
                        add_hline = True,
                        add_vline = True,
                        hline_color = "black",
                        hline_style = "dashed",
                        vline_color = "black",
                        vline_style ="dashed",
                        ha = "center",
                        va = "center",
                        repel = False,
                        ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Dual Multiple Factor Analysis (DMFA) - Graph of groups
    -------------------------------------------------------------------

    Description
    -----------
    Performs Dual Multiple Factor Analysis (DMFA) with supplementary quantitative variables and supplementary categorical variables. fviz_dmfa_quali_sup provides plotnine based elegant visualization of DMFA outputs for groups.

    Usage
    -----
    ```python
    >>> fviz_dmfa_quali_sup(self,
                            axis = [0,1],
                            x_lim = None,
                            y_lim = None,
                            x_label = None,
                            y_label = None,
                            title = None,
                            color = "black",
                            geom = ["arrow","text"],
                            palette = 
                            point_size = 1.5,
                            text_type = "text",
                            text_size = 8,
                            marker = "o",
                            add_grid = True,
                            add_hline = True,
                            add_vline = True,
                            hline_color = "black",
                            hline_style = "dashed",
                            vline_color = "black",
                            vline_style ="dashed",
                            ha = "center",
                            va = "center",
                            ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    `choice` : a text specifying the data to be plotted. Allowed values are "absolute" or "normalized"

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active individuals points (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").

    `text_size` : a numeric value specifying the label size (by default = 8).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

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
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_var
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Graph of variables
    >>> p = fviz_pca_var(res_pca)
    >>> print(p)
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_.n_components-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Set palette
    index = list(self.group_.name)
    if palette is None:
        palette = [x for x in list_colors if x != color][:len(index)]
    elif not isinstance(palette,(list,tuple)):
        raise TypeError("'palette' must be a list or a tuple of colors")
    elif len(palette) != len(index):
        raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    quali_sup_coord = self.quali_sup_.coord
    #suppres
    to_drop = [x for x in quali_sup_coord.index for j in index if x.__contains__(j)]
    coord = quali_sup_coord.drop(index=to_drop)
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "point" in geom:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # add
    for i,j in enumerate(coord.index):
        for k, l in enumerate(index):
            if not "_".join([j,l]) in to_drop:
                pass
            else:
                x, y = coord.loc[j,f"Dim.{axis[0]+1}"], quali_sup_coord.loc[j,f"Dim.{axis[1]+1}"]
                xend, yend =  quali_sup_coord.loc["_".join([j,l]),f"Dim.{axis[0]+1}"], quali_sup_coord.loc["_".join([j,l]),f"Dim.{axis[1]+1}"]
                p = p + pn.annotate("point",x=xend,y=yend,color=palette[k], shape = marker,size=point_size) + pn.annotate("segment",x=x,y=y,xend=xend,yend=yend,color=palette[k])
            
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Qualitative representation - DMFA"
    p = p + pn.labs(title=title,x=x_label,y=y_label,color="group")

    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_dmfa(self,choice="biplot",**kwargs)->pn:
    """
    Visualize Dual Multiple Factor Analysis (DMFA)
    ----------------------------------------------

    Description
    -----------
    Plot the graphs for a Dual Multiple Factor Analysis (DMFA) with supplementary quantitative variables and supplementary categorical variables.

        * fviz_dmfa_ind() : Graph of individuals
        * fviz_dmfa_var() : Graph of variables (Correlation circle)
        * fviz_dmfa_group() : Graph of groups
        * fviz_dmfa_quali_sup() : graph of supplementary cetegoricals variables

    Usage
    -----
    ```python
    >>> fviz_dmfa(self,choice=("ind","var","group","quali_sup"))
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    `choice` : the element to plot from the output. Possible value are : 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs (= Correlation circle)
        * 'group' for groups graphs
        * 'quali_sup' for supplementary categoricals variables
    
    `**kwargs` : further arguments passed to or from other methods

    Returns
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    seff fviz_dmfa_ind, fviz_dmfa_var, fviz_dmfa_group, fviz_dmfa_quali_sup
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    
    if choice not in ["ind","var","group","quali_sup"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'group', 'quali_sup'")

    if choice == "ind":
        return fviz_dmfa_ind(self,**kwargs)
    elif choice == "var":
        return fviz_dmfa_var(self,**kwargs)
    elif choice == "group":
        return fviz_dmfa_group(self,**kwargs)
    elif choice == "quali_sup":
        return fviz_dmfa_quali_sup(self,**kwargs)