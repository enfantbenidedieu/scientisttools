# -*- coding: utf-8 -*-
from pandas import DataFrame, Categorical
from plotnine import ggplot, aes, theme_minimal, geom_bar, geom_line, geom_point, geom_text,scale_y_continuous,labs,ylim

def fviz_screeplot(self,
                   element = "proportion",
                   geom = ["bar","line"],
                   fill_bar = "steelblue",
                   col_bar = "steelblue",
                   width_bar = None,
                   col_line = "black",
                   type_line = "solid",
                   add_labels = False,
                   ncp = 10,
                   y_lim = None,
                   ha = "center",
                   va = "bottom",
                   x_label = None,
                   y_label = None,
                   title = None,
                   ggtheme = theme_minimal()):
    """
    Visualize the eigenvalues/proportions/cumulative of dimensions
    --------------------------------------------------------------

    Description
    -----------
    This function support the results of multiple general factor analysis methods such as PCA (Principal Component Analysis), CA (Correspondence Analysis), MCA (Multiple Correspondence Analysis), etc...

    Usage
    -----
    ```python
    >>> fviz_screeplot(self,
                       element = "proportion",
                       geom = ["bar","line"],
                       fill_bar = "steelblue",
                       col_bar = "steelblue",
                       width_bar = None,
                       col_line = "black",
                       type_line = "solid",
                       add_labels = False,
                       ncp = 10,
                       y_lim = None,
                       ha = "center",
                       va = "bottom",
                       x_label = None,
                       y_label = None,
                       title = None,
                       ggtheme = theme_minimal())
    ```
    
    Parameters
    ----------
    `self`: an object of class PCA, PartialPCA, FactorAnalysis, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT, DMFA, MIXDISC, CMDSCALE

    `element`: a text specifying the data to be plotted. Allowed values are "proportion", "eigenvalue" or "cumulative"

    `geom`: a text specifying the geometry to be used for the graph. Allowed values are "bar" for barplot, "line" for lineplot or ["bar", "line"] to use both types.

    `fill_bar`: fill color for bar plot.

    `col_bar`: outline color for bar plot.

    `width_bar`: float, the width(s) of the bars

    `col_line`: color for line plot (when geom contains "line").

    `type_line`: line type. Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `add_labels`: logical value. If True, labels are added at the top of bars or points showing the information retained by each dimension.

    `ncp`: a numeric value specifying the number of dimensions to be shown.

    `y_lim`: a numeric list/tuple of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `ha`: horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va`: vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `x_label`: a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label`: a string specifying the label text of y (by default = None and a x_label is chosen).

    `title`: a string corresponding to the title of the graph you draw (by default = None and a title is chosen).
    
    `ggtheme`: function plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes: theme_gray(), theme_bw(), theme_gray(), theme_classic(), theme_void(), ....
    
    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_screeplot
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> print(fviz_screeplot(res_ca))
    ```
    """
    if self.model_ not in ["pca","partialpca","fa","ca","mca","specificmca","famd","mpca","pcamix","mfa","mfaqual","mfamix","mfact","dmfa","cmdscale"]:
        raise ValueError("'self' must be an object of class PCA, PartialPCA, FactorAnalysis, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT, DMFA, CMDSCALE")

    eig = self.eig_.iloc[:min(ncp,self.call_.n_components),:]

    if element == "eigenvalue":
        eig = eig["Eigenvalue"]
        text_labels = [str(round(x,3)) for x in eig.values]
        if y_label is None:
            y_label = "Eigenvalue"
    elif element == "proportion":
        eig = (1/100)*eig["Proportion"]
        text_labels = [str(round(100*x,2))+"%" for x in eig.values]
    elif element == "cumulative":
        eig = (1/100)*eig["Cumulative"]
        text_labels = [str(round(100*x,2))+"%" for x in eig.values]
        if y_label is None:
            y_label = "Cumulative % of explained variances"
    else:
        raise ValueError("'choice' must be one of 'proportion', 'eigenvalue', 'cumulative'")

    if isinstance(geom,str):
        if geom not in ["bar","line"]:
            raise ValueError("The specified value for the argument 'geom' are not allowed ")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in ["bar","line"]]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed ")
    
    df_eig = DataFrame({"dim" : Categorical(range(1,len(eig)+1)),"eig" : eig.values})
    
    p = ggplot(df_eig,aes(x = "dim",y="eig",group = 1))
    if "bar" in geom :
        p = p +  geom_bar(stat="identity",fill=fill_bar,color=col_bar,width=width_bar)
    if "line" in geom :
        p = p +  geom_line(color=col_line,linetype=type_line) + geom_point(shape="o",color=col_line)
    if add_labels:
        p = p + geom_text(label=text_labels,ha=ha,va=va)
    
    # Scale y continuous
    if element in ["proportion","cumulative"]:
        p = p + scale_y_continuous(labels=lambda l: ["%d%%" % (v * 100) for v in l])

    if title is None:
        title = "Scree plot"
    if x_label is None:
        x_label = "Dimensions"
    if y_label is None:
        y_label = "% of explained variances"
    p = p + labs(title = title, x = x_label, y = y_label)
    if y_lim is not None:
        p = p + ylim(y_lim)

    return p + ggtheme

def fviz_eig(self,**kwargs):
    """
    Visualize the eigenvalues/proportions/cumulative of dimensions
    --------------------------------------------------------------

    see `fviz_screeplot`

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    return fviz_screeplot(self,**kwargs)