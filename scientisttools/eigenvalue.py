# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotnine as pn
from mizani.formatters import percent_format

def get_eig(self) -> pd.DataFrame:
    """
    Extract the eigenvalues/variances of dimensions
    -----------------------------------------------

    Description
    -----------
    Eigenvalues correspond to the amount of the variation explained by each principal component.

    Parameters:
    -----------
    self : an object of class PCA, PartialPCA, CA, MCA, FAMD, MFA, MFAQUAL, MFAMIX, MFACT, CMDSCALE

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ not in ["pca","partialpca","ca","mca","famd","efa","mfa","mfaqual","mfamix","mfact","cmdscale"]:
        raise TypeError("'self' must be an object of class PCA, PartialPCA, CA, MCA, FAMD, EFA, MFA, MFAQUAL, MFAMIX, MFACT, CMDSCALE")
    
    return self.eig_
        

def get_eigenvalue(self) -> pd.DataFrame:
    """
    Extract the eigenvalues/variances of dimensions
    -----------------------------------------------

    Description
    -----------
    Eigenvalues correspond to the amount of the variation explained by each principal component.

    Parameters:
    -----------
    self : an object of class PCA, PartialPCA, CA, MCA, FAMD, MFA, MFAQUAL, MFAMIX, MFACT, CMDS,  HMFA

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    return get_eig(self)

def fviz_screeplot(self,
                   choice="proportion",
                   geom_type=["bar","line"],
                   y_lim=None,
                   bar_fill = "steelblue",
                   bar_color="steelblue",
                   line_color="black",
                   line_type="solid",
                   bar_width=None,
                   ncp=10,
                   add_labels=False,
                   ha = "center",
                   va = "bottom",
                   title=None,
                   x_label=None,
                   y_label=None,
                   ggtheme=pn.theme_minimal())-> pn:
    """
    Extract and visualize the eigenvalues/proportions of dimensions
    -------------------------------------------------------------

    Parameters
    ----------
    self : an object of class PCA, CA, MCA, FAMD, MFA, MFAQUAL, MFAMIX, MFACT, CMDS, DISQUAL, MIXDISC, CMDSCALE

    choice : a text specifying the data to be plotted. Allowed values are "proportion" or "eigenvalue".

    geom_type : a text specifying the geometry to be used for the graph. Allowed values are "bar" for barplot, 
                "line" for lineplot or ["bar", "line"] to use both types.

    ylim : y-axis limits, default = None

    barfill : 	fill color for bar plot.

    barcolor : outline color for bar plot.

    linecolor : color for line plot (when geom contains "line").

    linetype : line type

    barwidth : float, the width(s) of the bars

    ncp : a numeric value specifying the number of dimensions to be shown.

    addlabels : logical value. If TRUE, labels are added at the top of bars or points showing the information retained by each dimension.

    ha : horizontal adjustment of the labels.

    va : vertical adjustment of the labels.

    title : title of the graph

    xlabel : x-axis title

    ylabel : y-axis title
    
    ggtheme : function plotnine theme name. Default value is theme_gray(). Allowed values include plotnine official themes: 
                theme_gray(), theme_bw(), theme_minimal(), theme_classic(), theme_void(), ....
    
    Return
    ------
    figure : a plotnine graphs

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
        
    if self.model_ not in ["pca","ca","mca","famd","partialpca","efa","mfa","mfaqual","mfamix","mfact","cmdscale"]:
        raise ValueError("'self' must be an object of class PCA, CA, MCA, FAMD, PartialPCA, EFA, MFA, MFAQUAL, MFAMIX, MFACT, CMDSCALE")

    eig = get_eigenvalue(self)
    eig = eig.iloc[:min(ncp,self.call_["n_components"]),:]

    if choice == "eigenvalue":
        eig = eig["eigenvalue"]
        text_labels = list([str(np.around(x,3)) for x in eig.values])
        if y_label is None:
            y_label = "Eigenvalue"
    elif choice == "proportion":
        eig = (1/100)*eig["proportion"]
        text_labels = list([str(np.around(100*x,2))+"%" for x in eig.values])
    else:
        raise ValueError("'choice' must be one of 'proportion', 'eigenvalue'")

    if isinstance(geom_type,str):
        if geom_type not in ["bar","line"]:
            raise ValueError("The specified value for the argument geomtype are not allowed ")
    elif (isinstance(geom_type,list) or isinstance(geom_type,tuple)):
        intersect = [x for x in geom_type if x in ["bar","line"]]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed ")
    
    df_eig = pd.DataFrame({"dim" : pd.Categorical(np.arange(1,len(eig)+1)),"eig" : eig.values})
    
    p = pn.ggplot(df_eig,pn.aes(x = "dim",y="eig",group = 1))
    if "bar" in geom_type :
        p = p   +   pn.geom_bar(stat="identity",fill=bar_fill,color=bar_color,width=bar_width)
    if "line" in geom_type :
        p = p  +   pn.geom_line(color=line_color,linetype=line_type) + pn.geom_point(shape="o",color=line_color)
    if add_labels:
        p = p + pn.geom_text(label=text_labels,ha = ha,va = va)
    
    # Scale y continuous
    if choice == "proportion":
        p = p + pn.scale_y_continuous(labels=percent_format())

    if title is None:
        title = "Scree plot"
    if x_label is None:
        x_label = "Dimensions"
    if y_label is None:
        y_label = "Percentage of explained variances"
    
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    p = p + ggtheme
    return p

def fviz_eig(self,**kwargs) -> pn:
    """
    Extract and visualize the eigenvalues/proportions of dimensions
    -------------------------------------------------------------

    see fviz_screeplot(...)
    """
    return fviz_screeplot(self,**kwargs)