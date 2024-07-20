# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np
import plotnine3d as pn3d

from .text_label import text_label, text3d_label
from .gg_circle import gg_circle

def fviz_pca_ind(self,
                 axis = [0,1],
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 color ="black",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 legend_title = None,
                 habillage = None,
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
    Visualize Principal Component Analysis (PCA) - Graph of individuals
    -------------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. fviz_pca_ind provides plotnine based elegant visualization of PCA outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_pca_ind(self,
                    axis=[0,1],
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = None,
                    color ="black",
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
                    add_grid = True,
                    ind_sup = True,
                    color_sup = "blue",
                    marker_sup = "^",
                    legend_title = None,
                    habillage = None,
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
    `self` : an object of class PCA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active individuals points (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").
    
    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `ind_sup` : a boolean to either add or not supplementary individuals (by default = True).

    `color_sup` : a color for the supplementary individuals points (by default = "blue").

    `marker_sup` :  a marker style for the supplementary individuals points (by default = "^").

    `legend_title` : a string corresponding to the title of the legend (by default = None).

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
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Extract individuals coordinates
    coord = self.ind_["coord"]

    # Add Active Data
    coord = pd.concat([coord,self.call_["X"]],axis=1)

    # Add supplementary quantitatives columns
    if self.quanti_sup is not None:
        X_quanti_sup = self.call_["Xtot"].loc[:,self.call_["quanti_sup"]].astype("float")
        if self.ind_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat([coord,X_quanti_sup],axis=1)
    
    ################ Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = self.ind_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.ind_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise TypeError("'color' must me a numeric variable.")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    if habillage is None :  
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(color=c),size=point_size,show_legend=False)+ 
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
            c = [str(x+1) for x in color.labels_]
            if legend_title is None:
                legend_title = "Cluster"
                if "point" in geom:
                    p = (p + pn.geom_point(pn.aes(color=c,linetype = c),size=point_size)+
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
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        # Add ellipse
        if add_ellipses:
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    ##### Add supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(ind_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                    color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=ind_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                        color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text_label(text_type,data=ind_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                        color = color_sup,size=text_size,va=va,ha=ha)
                    
    # Add supplementary qualitatives/categories
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            if habillage is None:
                quali_sup_coord = self.quali_sup_["coord"]
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
        title = "Individuals Factor Map - PCA"
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
def fviz_pca_var(self,
                 axis=[0,1],
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["arrow","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 scale = 1,
                 legend_title = None,
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "blue",
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
    Visualize Principal Component Analysis (PCA) - Graph of variables
    -----------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. fviz_pca_var provides plotnine based elegant visualization of PCA outputs for variables.

    Usage
    -----
    ```python
    >>> fviz_pca_var(self,
                    axis = [0,1],
                    x_label = None,
                    y_label = None,
                    title = None,
                    color = "black",
                    geom = ["arrow","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    scale = 1,
                    legend_title = None,
                    text_type = "text",
                    text_size = 8,
                    add_grid = True,
                    quanti_sup=True,
                    color_sup = "blue",
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
    `self` : an object of class PCA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active variables (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).
    
    `scale` : a numeric specifying scale the variables coordinates (by default 1)

    `legend_title` : a string corresponding to the title of the legend (by default = None).
    
    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `text_size` : a numeric value specifying the label size (by default = 8).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `quanti_sup` : a boolean to either add or not supplementary quantitatives variables (by default = True).

    `color_sup` : a color for the supplementary quantitatives variables (by default = "blue").

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
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    coord = self.var_["coord"]*scale

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")

    if isinstance(color,str):
        if color == "cos2":
            c = self.var_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.var_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
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
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if "text" in geom:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add supplmentary continuous variables
    if quanti_sup:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_["coord"]*scale
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
        title = "Variables Factor Map - PCA"
    
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

def fviz_pca_biplot(self,
                    axis=[0,1],
                    x_label = None,
                    y_label = None,
                    x_lim = None,
                    y_lim = None,
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
                    habillage = None,
                    add_circle = False,
                    var_color_circle="gray",
                    ind_sup = True,
                    ind_color_sup = "blue",
                    ind_marker_sup = "^",
                    quali_sup = True,
                    quali_sup_color = "red",
                    quali_sup_marker = "v",
                    quanti_sup = True,
                    var_color_sup = "blue",
                    var_linestyle_sup="dashed",
                    repel = True,
                    add_ellipses=False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
                    title = "PCA - Biplot",
                    arrow_angle=10,
                    arrow_length =0.1,
                    add_hline = True,
                    add_vline=True,
                    add_grid = True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    ggtheme=pn.theme_minimal()) ->pn :
    """
    Visualize Principal Component Analysis (PCA) - Biplot of individuals and variables
    ----------------------------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. fviz_pca_biplot provides plotnine based elegant visualization of PCA outputs for individuals and variables.

    Usage
    -----
    ```python
    >>> fviz_pca_biplot(self,
                        axis=[0,1],
                        x_label = None,
                        y_label = None,
                        x_lim = None,
                        y_lim = None,
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
                        habillage = None,
                        add_circle = False,
                        var_color_circle="gray",
                        ind_sup = True,
                        ind_color_sup = "blue",
                        ind_marker_sup = "^",
                        quali_sup = True,
                        quali_sup_color = "red",
                        quali_sup_marker = "v",
                        quanti_sup = True,
                        var_color_sup = "blue",
                        var_linestyle_sup="dashed",
                        repel = True,
                        add_ellipses=False, 
                        ellipse_type = "t",
                        confint_level = 0.95,
                        geom_ellipse = "polygon",
                        title = "PCA - Biplot",
                        arrow_angle=10,
                        arrow_length =0.1,
                        add_hline = True,
                        add_vline=True,
                        add_grid = True,
                        ha="center",
                        va="center",
                        hline_color="black",
                        hline_style="dashed",
                        vline_color="black",
                        vline_style ="dashed",
                        ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    see fviz_pca_ind, fviz_pca_var

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    ```python
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_biplot
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Biplot of individuals and variables
    >>> p = fviz_pca_biplot(res_pca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
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

    # Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
        ind_coord = pd.concat([ind_coord,X_quali_sup],axis=1)
    
    # Variables coordinates
    var_coord = self.var_["coord"]*rscale

    p = pn.ggplot()

    #####################################################################################################################################################
    # Individuals Informations
    #####################################################################################################################################################

    if habillage is None :  
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
    else:
        if habillage not in ind_coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in ind_geom:
            p = p + pn.geom_point(data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,linetype = habillage),
                                  size=ind_point_size)
        if "text" in ind_geom:
            if repel:
                p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,label=ind_coord.index),
                                   size=ind_text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,label=ind_coord.index),
                                   size=ind_text_size,va=va,ha=ha)
        
        if add_ellipses:
            p = p + pn.stat_ellipse(data=ind_coord,geom=geom_ellipse,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,fill=habillage),
                                    type = ellipse_type,alpha = 0.25,level=confint_level)
    
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
    
    # Add supplementary qualitatives coordinates
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            if habillage is None:
                quali_sup_coord = self.quali_sup_["coord"]
                if "point" in ind_geom:
                    p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                          color=quali_sup_color,size=ind_point_size,shape=quali_sup_marker)
                if "text" in ind_geom:
                    if repel:
                        p = p + text_label(ind_text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                           color=quali_sup_color,size=ind_text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': quali_sup_color,'lw':1.0}})
                    else:
                        p = p + text_label(ind_text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                           color =quali_sup_color,size=ind_text_size,va=va,ha=ha)
    
    #########################################################################################################################################################
    #   Variables informations
    ##########################################################################################################################################################

    if "arrow" in var_geom:
            p = p + pn.geom_segment(data=var_coord,mapping=pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=var_color)
    if "text" in var_geom:
        p = p + text_label(var_text_type,data=var_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_coord.index),
                           color=var_color,size=var_text_size,va=va,ha=ha)
    
    # Add supplmentary continuous variables
    if quanti_sup:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_["coord"]*rscale
            if "arrow" in var_geom:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=var_color_sup,linetype=var_linestyle_sup)
            if "text" in var_geom:
                p  = p + text_label(var_text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=var_color_sup,size=var_text_size,va=va,ha=ha)
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
        title = "PCA - Biplot"
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

def fviz_pca3d_ind(self,
                   axis=[0,1,2],
                   x_lim=None,
                   y_lim=None,
                   x_label = None,
                   y_label = None,
                   z_label = None,
                   title =None,
                   color ="black",
                   geom = ["point","text"],
                   gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                   point_size = 1.5,
                   text_size = 8,
                   text_type = "text",
                   marker = "o",
                   add_grid =True,
                   ind_sup=True,
                   color_sup = "blue",
                   marker_sup = "^",
                   legend_title=None,
                   habillage = None,
                   quali_sup = True,
                   color_quali_sup = "red",
                   ha="center",
                   va="center",
                   repel=False,
                   lim_cos2 = None,
                   lim_contrib = None,
                   ggtheme=pn.theme_minimal()) -> pn3d:
    
    """
    Visualize Principal Component Analysis (PCA) - 3D graph of individuals
    ----------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_pca3d_ind(self,
                        axis=[0,1,2],
                        x_lim = None,
                        y_lim = None,
                        x_label = None,
                        y_label = None,
                        z_label = None,
                        title = None,
                        color = "black",
                        geom = ["point","text"],
                        gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                        point_size = 1.5,
                        text_size = 8,
                        text_type = "text",
                        marker = "o",
                        add_grid = True,
                        ind_sup = True,
                        color_sup = "blue",
                        marker_sup = "^",
                        legend_title = None,
                        habillage = None,
                        quali_sup = True,
                        color_quali_sup = "red",
                        ha = "center",
                        va = "center",
                        repel = False,
                        lim_cos2 = None,
                        lim_contrib = None,
                        ggtheme = pn.theme_minimal())
    ```

    Parameters
    ----------
    see fviz_pca_ind, fviz_pca_var

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

     ```python
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca3d_ind
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # 3D Graph of individuals
    >>> p = fviz_pca3d_ind(res_pca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if ((len(axis) !=3) or 
        (axis[0] < 0) or 
        (axis[2] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1]) or 
        (axis[0] > axis[2]) or 
        (axis[1] > axis[2])) :
        raise ValueError("You must pass a valid 'axis'.")

    #### Extract individuals coordinates
    coord = self.ind_["coord"]

    # Add Active Data
    coord = pd.concat([coord,self.call_["X"]],axis=1)

    ################ Add supplementary quantitatives columns
    if self.quanti_sup is not None:
        X_quanti_sup = self.call_["Xtot"].loc[:,self.call_["quanti_sup"]].astype("float")
        if self.ind_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat([coord,X_quanti_sup],axis=1)
    
    ################ Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")
    
    if isinstance(color,str):
        if color == "cos2":
            coord["cos2"] = self.ind_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Cos2"
        elif color == "contrib":
            coord["contrib"] = self.ind_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise ValueError("Error : 'color' must me a numeric variable.")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        coord["num_var"] = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(color, "labels_"):
        coord["cluster"] = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"

    # Initialize
    p = pn3d.ggplot_3d(data=coord) + pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",z=f"Dim.{axis[2]+1}",label=coord.index)

    if habillage is None :  
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn3d.geom_point_3d(pn.aes(color=color),shape=marker,size=point_size,show_legend=False)+
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,mapping=pn.aes(color=color),size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,mapping=pn.aes(color=color),size=text_size,va=va,ha=ha)
        elif isinstance(color,np.ndarray):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn3d.geom_point_3d(pn.aes(color="num_var"),shape=marker,size=point_size,show_legend=False)+
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,mapping=pn.aes(color="num_var"),size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
            if "point" in geom:
                p = (p + pn3d.geom_point_3d(pn.aes(color="cluster",linetype = "cluster"),size=point_size)+
                         pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,mapping=pn.aes(color="cluster"),size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text3d_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom:
                p = p + pn3d.geom_point_3d(color=color,shape=marker,size=point_size,show_legend=False)
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in geom:
            p = p + pn3d.geom_point_3d(pn.aes(color = habillage,linetype = habillage),size=point_size)
        if "text" in geom:
            if repel:
                p = p + text3d_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text3d_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
    
    # Add supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_["coord"]
            if "point" in geom:
                p = p + pn3d.geom_point_3d(ind_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                           color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text3d_label(text_type,data=ind_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                         color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,data=ind_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),
                                         color = color_sup,size=text_size,va=va,ha=ha)
    
    # Add supplementary qualitatives coordinates
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            if habillage is None:
                quali_sup_coord = self.quali_sup_["coord"]
                if "point" in geom:
                    p = p + pn3d.geom_point_3d(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                               color=color_quali_sup,size=point_size)
                if "text" in geom:
                    if repel:
                        p = p + text3d_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                             color=color_quali_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,'lw':1.0}})
                    else:
                        p = p + text3d_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                             color =color_quali_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set Z label
    if z_label is None:
        z_label = "Dim."+str(axis[2]+1)+" ("+str(round(proportion[axis[2]],2))+"%)"
    
    # Set title
    if title is None:
        title = "Individuals 3D Factor Map - PCA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)+pn3d.zlab(z_label)
    
    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    # Add theme
    p = p + ggtheme
    
    return p

def fviz_pca(self,choice="biplot",**kwargs)->pn:
    """
    Visualize Principal Component Analysis (PCA)
    --------------------------------------------

    Description
    -----------
    Plot the graphs for a Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

        * fviz_pca_ind() : Graph of individuals
        * fviz_pca_var() : Graph of variables (Correlation circle)
        * fviz_pca_biplot() : Biplot of individuals and variables
        * fviz_pca3d_ind() : 3D Graph of individuals

    Usage
    -----
    ```python
    >>> fviz_pca(self,choice=("ind","var","biplot","3D"))
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    `choice` : the element to plot from the output. Possible value are : 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs (= Correlation circle)
        * 'biplot' for biplot of individuals and variables
        * '3D' for 3D graph of individuals
    
    `**kwargs` : further arguments passed to or from other methods

    Returns
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    seff fviz_pca_ind, fviz_pca_var, fviz_pca_biplot, fviz_pca3d_ind
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if choice not in ["ind","var","biplot","3D"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'biplot', '3D'")

    if choice == "ind":
        return fviz_pca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_pca_var(self,**kwargs)
    elif choice == "biplot":
        return fviz_pca_biplot(self,**kwargs)
    elif choice == "3D":
        return fviz_pca3d_ind(self,**kwargs)