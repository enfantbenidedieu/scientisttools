# -*- coding: utf-8 -*-
from pandas import DataFrame, Categorical
from plotnine import (
    ggplot,
    geom_bar, 
    geom_line, 
    geom_point, 
    geom_text, 
    aes, 
    theme_minimal, 
    scale_y_continuous,
    labs,
    ylim
)

def fviz_screeplot(obj,
                   choice = "proportion",
                   geom = ("bar","line"),
                   bar_args = dict(color="steelblue",fill = "steelblue",width=None),
                   line_args = dict(color="black",linetype="solid"),
                   show_labels = False,
                   ncp = 10,
                   y_lim = None,
                   x_label = None,
                   y_label = None,
                   title = None,
                   pntheme = theme_minimal()):
    """
    Visualize the eigenvalues/variances of dimensions
    
    This function support the results of multiple general factor analysis methods such as PCA (Principal Component Analysis), CA (Correspondence Analysis), MCA (Multiple Correspondence Analysis), etc...

    Parameters
    ----------
    obj : class
        An object of class which have ``eig_`` as attribute.

    choice : {"proportion", "eigenvalue","cumulative"}, default = "proportion"
        A text specifying the data to be plotted. 

    geom : str, list, tuple, default = ("bar","line")
        The geometry to be used for the graph. Allowed values are the combinaison of ("bar","line"). 

        * "bar" to show only bar.
        * "line" to show only line.
        * ("bar", "line") to use both types.

    bar_args : dict, default = dict(color="steelblue",fill = "steelblue",width=None)
        A dictionary containing others keyword arguments for barplot (see https://plotnine.org/reference/geom_bar.html).

    line_args : dict, default = dict(color="black",linetype="solid")
        A dictionary containing others keyword arguments for line plot (see https://plotnine.org/reference/geom_line.html).

    show_labels : bool, default = False
        If True, labels are added at the top of bars or points showing the information retained by each dimension.

    ncp : int, default = 10
        The number of dimensions to be shown.

    y_lim : list, tuple, default = None
        The range of the plotted y values.

    x_label : str, default = None
        The label text of x. If None, then x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then y_label is chosen.

    title : str, default = None
        The title of the graph you draw. If None, then a title is chosen.
    
    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).
    
    Returns
    -------
    A plotnine object.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_screeplot
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> print(fviz_screeplot(clf))
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj has eig_ as attribute
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not hasattr(obj, "eig_"):
        raise TypeError("obj must have 'eig_' as attribute")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    eig = obj.eig_.iloc[:min(ncp,obj.call_.ncp),:]

    if choice == "eigenvalue":
        eig = eig["Eigenvalue"]
        text_labels = [str(round(x,3)) for x in eig.values]
        if y_label is None:
            y_label = "Eigenvalue"
    elif choice == "proportion":
        eig = (1/100)*eig["Proportion (%)"]
        text_labels = [str(round(100*x,2))+"%" for x in eig.values]
    elif choice == "cumulative":
        eig = (1/100)*eig["Cumulative (%)"]
        text_labels = [str(round(100*x,2))+"%" for x in eig.values]
        if y_label is None:
            y_label = "Cumulative % of explained variances"
    else:
        raise ValueError("'choice' must be one of 'proportion', 'eigenvalue', 'cumulative'")

    if isinstance(geom,str):
        if geom not in ("bar","line"):
            raise ValueError("The specified value for the argument 'geom' are not allowed ")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in ("bar","line")]
        if len(intersect) == 0:
            raise ValueError("The specified value(s) for the argument geom are not allowed ")
    
    df_eig = DataFrame({"dim" : Categorical(range(1,len(eig)+1)),"eig" : eig.values})
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # initialization
    p = ggplot(df_eig,aes(x = "dim",y="eig",group = 1))

    # show barplot
    if "bar" in geom :
        p = p +  geom_bar(stat="identity",**bar_args)
    # show line
    if "line" in geom :
        p = (
            p 
            + geom_line(**line_args) 
            + geom_point(shape="o",color=line_args["color"])
        )
    
    # show labels
    if show_labels:
        p = p + geom_text(label=text_labels,ha="center",va="bottom")
    
    # scale y continuous
    if choice in ("proportion","cumulative"):
        p = p + scale_y_continuous(labels=lambda l: ["%d%%" % (v * 100) for v in l])

    # set title
    if title is None:
        title = "Scree plot"
    # set x_label
    if x_label is None:
        x_label = "Dimensions"
    # set y_label
    if y_label is None:
        y_label = "% of explained variances"
    p = p + labs(title = title, x = x_label, y = y_label)
    # set y_lim
    if y_lim is not None:
        p = p + ylim(y_lim)
    return p + pntheme

def fviz_eig(obj,**kwargs):
    """
    Visualize the eigenvalues/variances of dimensions

    see :class:`~scientisttools.fviz_screeplot`
    """
    return fviz_screeplot(obj,**kwargs)