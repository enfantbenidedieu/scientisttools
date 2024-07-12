# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd

from .text_label import text_label
from .gg_circle import gg_circle

def fviz_cca_ind(self,
                 axis=[0,1],
                 which = "X",
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["point","text"],
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
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
    Visualize Canonical Correlation Analysis (CCA) - Graph of individuals
    ---------------------------------------------------------------------

    Description
    -----------
    Performs Canonical Correlation Analysis (CCA) to highlight correlations between two dataframes. fviz_cca_ind provides plotnine based elegant visualization of CCA outputs for individuals.

    Usage
    -----
    ```python 
    >>> fviz_cca_ind(self,
                    axis=[0,1],
                    which = "X",
                    x_lim=None,
                    y_lim=None,
                    x_label = None,
                    y_label = None,
                    title =None,
                    color ="black",
                    geom = ["point","text"],
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
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
    `self` : an object of class CCA

    see fviz_pca_ind

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load lifecyclesavings dataset
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    >>> from scientisttools import CCA, fviz_cca_ind
    >>> res_cca = CCA(lifecyclesavings,vars=[1,2])
    >>> p = fviz_cca_ind(res_cca)
    >>> print(p)
    ```
    """
    if self.model_ != "cca":
        raise TypeError("'self' must be an object of class CCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    if which not in ["X","Y"]:
        raise ValueError("'which' should be one of 'X' ot 'Y'")

    #### Extract individuals coordinates
    if which == "X":
        scores = self.ind_["xscores"]
    elif which == "Y":
        scores = self.ind_["yscores"]

    # Initialize
    p = pn.ggplot(data=scores,mapping=pn.aes(x = f"V.{axis[0]+1}",y=f"V.{axis[1]+1}",label=scores.index))

    if "point" in geom:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)

    # Set x label
    if x_label is None:
        x_label = which + " dimension " + str(axis[0]+1)
    # Set y label
    if y_label is None:
        y_label = which + " dimension " + str(axis[1]+1)
    # Set title
    if title is None:
        title = "Individuals Factor Map - CCA"
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

def fviz_cca_var(self,
                 axis=[0,1],
                 which = "X",
                 x_label = None,
                 y_label = None,
                 title =None,
                 xcolor = "black",
                 ycolor = "blue",
                 xmarker = "o",
                 ymarker = "^",
                 geom = ["point","text"],
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 add_circle = True,
                 color_circle="gray",
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Canonical Correlation Analysis (CCA) - Graph of variables
    -------------------------------------------------------------------

    Description
    -----------
    Performs Canonical Correlation Analysis (CCA) to highlight correlations between two dataframes. fviz_cca_var provides plotnine based elegant visualization of CCA outputs for variables.

    Usage
    -----
    ```python 
    >>> fviz_cca_var(self,
                    axis=[0,1],
                    which = "X",
                    x_label = None,
                    y_label = None,
                    title =None,
                    xcolor = "black",
                    ycolor = "blue",
                    xmarker = "o",
                    ymarker = "^",
                    geom = ["point","text"],
                    text_type = "text",
                    text_size = 8,
                    add_grid =True,
                    add_hline = True,
                    add_vline=True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    add_circle = True,
                    color_circle="gray",
                    repel=False,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class CCA

    see fviz_pca_var

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load lifecyclesavings dataset
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    >>> from scientisttools import CCA, fviz_cca_var
    >>> res_cca = CCA(lifecyclesavings,vars=[1,2])
    >>> p = fviz_cca_var(res_cca)
    >>> print(p)
    ```
    """
    if self.model_ != "cca":
        raise ValueError("'self' must be an object of class CCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")
    
    if which not in ["X","Y"]:
        raise ValueError("'which' should be one of 'X' ot 'Y'")

    # Extract scores
    if which == "X":
        xscores = self.var_["corr_X_xscores"]
        yscores = self.var_["corr_Y_xscores"]
    elif which == "Y":
        xscores = self.var_["corr_X_yscores"]
        yscores = self.var_["corr_Y_yscores"]
    
    # Rename columns
    xscores.columns = ["X."+str(x+1) for x in range(xscores.shape[1])]
    yscores.columns = ["Y."+str(x+1) for x in range(yscores.shape[1])]

    # Initialize
    p = pn.ggplot()

    ###### Add x scores coordinates
    if "point" in geom:
        p = p + pn.geom_point(data=xscores,mapping=pn.aes(x = f"X.{axis[0]+1}",y=f"X.{axis[1]+1}"),color=xcolor,shape=xmarker)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,data=xscores,mapping=pn.aes(x = f"X.{axis[0]+1}",y=f"X.{axis[1]+1}",label=xscores.index),
                               color=xcolor,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': xcolor,'lw':1.0}})
        else:
            p = p + text_label(text_type,data=xscores,mapping=pn.aes(x = f"V.{axis[0]+1}",y=f"V.{axis[1]+1}",label=xscores.index),
                               color=xcolor,size=text_size,va=va,ha=ha)
    
    ######## Add y scores coordinates
    if "point" in geom:
        p = p + pn.geom_point(data=yscores,mapping=pn.aes(x = f"Y.{axis[0]+1}",y=f"Y.{axis[1]+1}"),color=ycolor,shape=ymarker)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,data=yscores,mapping=pn.aes(x = f"Y.{axis[0]+1}",y=f"Y.{axis[1]+1}",label=yscores.index),
                               color=ycolor,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': ycolor,'lw':1.0}})
        else:
            p = p + text_label(text_type,data=yscores,mapping=pn.aes(x = f"V.{axis[0]+1}",y=f"V.{axis[1]+1}",label=yscores.index),
                               color=ycolor,size=text_size,va=va,ha=ha)

    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=color_circle, fill=None)
        p = p + gg_circle(r=0.5, xc=0.0, yc=0.0, color=color_circle, fill=None)
     
    # Set x label
    if x_label is None:
        x_label = which + " dimension " + str(axis[0]+1)
    # Set y label
    if y_label is None:
        y_label = which + " dimension " + str(axis[1]+1)
    if title is None:
        title = "Variables Factor Map - CCA"
    
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

def fviz_cca_scatterplot(self,
                         which=0,
                         x_lim = None,
                         y_lim = None,
                         x_label = None,
                         y_label = None,
                         title = None,
                         geom = ["point","text"],
                         text_type = "text",
                         text_size = 8,
                         color = "black",
                         marker = "o",
                         smooth = False,
                         smooth_color = "green",
                         abline = True,
                         abline_color = "red",
                         add_ellipse = False,
                         ellipse_color = "blue",
                         ha="center",
                         va="center",
                         repel=False,
                         ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Canonical Correlation Analysis (CCA) - Scatter plot
    -------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_cca_scatterplot(self,
                            which=0,
                            x_lim = None,
                            y_lim = None,
                            x_label = None,
                            y_label = None,
                            title = None,
                            geom = ["point","text"],
                            text_type = "text",
                            text_size = 8,
                            color = "black",
                            marker = "o",
                            smooth = False,
                            smooth_color = "green",
                            abline = True,
                            abline_color = "red",
                            add_ellipse = False,
                            ellipse_color = "blue",
                            ha="center",
                            va="center",
                            repel=False,
                            ggtheme=pn.theme_minimal()) 
    ```

    Parameters
    ----------
    `self` : an object of class CCA

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load lifecyclesavings dataset
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    >>> from scientisttools import CCA, fviz_cca_scatterplot
    >>> res_cca = CCA(lifecyclesavings,vars=[1,2])
    >>> p = fviz_cca_scatterplot(res_cca)
    >>> print(p)
    ```
    """
    if self.model_ != "cca":
        raise TypeError("'self' must be an object of class CCA")
    
    if which < 0 or which > 1:
        raise ValueError("'which' should be either 0 or 1")
    
    #### Individuals scores in the two dimensions
    scores = pd.concat((self.ind_["xscores"].iloc[:,which],self.ind_["yscores"].iloc[:,which]),axis=1)
    scores.columns = ["X","Y"]

    # Initialize
    p = pn.ggplot(data=scores,mapping=pn.aes(x = "X",y="Y",label=scores.index))

    ###### Add x scores coordinates
    if "point" in geom:
        p = p + pn.geom_point(color=color,shape=marker)
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add abline line
    if abline:
        p = p + pn.geom_abline(color = abline_color)
    
    # Add loess line
    if smooth:
        p = p + pn.stat_smooth(method="loess", se=False, color=smooth_color) 
    
    # Add ellipse circle
    if add_ellipse:
        p = p + pn.stat_ellipse(color=ellipse_color)
    
    # Set x label
    if x_label is None:
        x_label = "X dimension "+str(which+1)
    # Set y label
    if y_label is None:
        y_label = "Y dimension "+str(which+1)
    # Set title
    if title is None:
        title = "Canonical Correlation Analysis (Cor = " + str(round(self.can_corr_.iloc[which,0],2)) + ")"
    p = p + pn.labs(title=title,x=x_label,y = y_label)
    
    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
    
    # Add theme
    p = p + ggtheme

    return p
    
def fviz_cca(self,choice="ind",**kwargs)->pn:
    """
    Visualize Canonical Correlation Analysis (CCA)
    ----------------------------------------------

    Description
    -----------
    Plot the graphs for a Canonical Correlation Analysis (CCA).

        * fviz_cca_ind() : Graph of individuals
        * fviz_cca_var() : Graph of variables (Correlation circle)
        * fviz_cca_scatterplot() : Scatter plot

    Usage
    -----
    ```python
    >>> fviz_cca(self, choice = ("ind", "var", "scatter"))
    ```

    Parameters
    ----------
    `self` : an object of class CCA

    `choice` : the element to plot from the output. Possible value are : 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs
        * 'scatter' for scatter plot

    `**kwargs` : further arguments passed to or from other methods

    Returns
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if self is an object of class CCA
    if self.model_ != "cca":
        raise TypeError("'self' must be an instance of class CCA")
    
    if choice not in ["ind","var","scatter"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'scatter'")

    if choice == "ind":
        return fviz_cca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_cca_var(self,**kwargs)
    elif choice == "scatter":
        return fviz_cca_scatterplot(self,**kwargs)