# -*- coding: utf-8 -*-
from numpy import triu_indices
from pandas import DataFrame
from scipy.spatial.distance import pdist,squareform
from plotnine import (
    aes,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme,
    theme_minimal,
    xlim,
    ylim
)

# intern functions
from ._fviz import add_scatter, set_axis

def fviz_pcoa_ind(obj,
                  axis = [0,1],
                  geom = ("point","text"),
                  repel = False,
                  col_ind = "black",
                  point_args = dict(size=1.5),
                  text_args = dict(size=8),
                  ind_sup = True,
                  col_ind_sup = "blue",
                  point_args_ind_sup = dict(size=1.5),
                  text_args_ind_sup = dict(size=8),
                  x_label = None,
                  y_label = None,
                  x_lim = None,
                  y_lim = None,
                  title = None,
                  subtitle = None,
                  pntheme = theme_minimal(),
                  **kwargs):
    """
    Visualize Principal Coordinates Analysis - Graph of individuals

    Principal Coordinates Analysis (from :class:`~scientisttools.PCoA`), also known as classical multidimensional scaling (MDS), is a method used to explore and visualize similarities or dissimilarities among a set of objects or samples. 
    :class:`~scientisttools.fviz_pcoa_ind` provides plotnine-based elegant visualization of from :class:`~scientisttools.PCoA` outputs for individuals.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCoA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting individuals text labels or not.

    col_ind : str, default = "black"
        Color for individuals.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for individuals points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for individuals texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    ind_sup : bool, default = True
        If True, show supplementary individuals points and/or texts.

    col_ind_sup : str, default = "blue"
        Color for supplementary individuals.

    point_args_ind_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary individuals points.

    text_args_ind_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary individuals texts.

    x_lim : list, tuple, default = None
        The range of the plotted x values.

    y_lim : list, tuple, default = None
        The range of the plotted y values.

    x_label : str, default = None
        The label text of x. If None, then a x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then a y_label is chosen.
    
    title : str, default = None 
        The title of the graph you draw. If None, then a title is chosen.

    subtitle : str, default = None
        The subtitle of the graph you draw.

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.
    
    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_pcoa`
        Visualize Principal Coordinates Analysis.

    Examples
    --------
    >>> from scientisttools.datasets import autosmds
    >>> from scientisttools import PCoA, fviz_pcoa_ind
    >>> clf = PCoA(ncp=2,ind_sup=(12,13,14))
    >>> clf.fit(autosmds)
    PCoA(ind_sup=(12,13,14),ncp=2)
    >>> p = fviz_pcoa_ind(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class PCoA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCoA":
        raise TypeError("'obj' must be an object of class PCoA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active indivdiuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(
            p = ggplot(),
            data = obj.ind_.coord,
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_ind,
            point_args = point_args,
            text_args = text_args
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ind_sup and hasattr(obj, "ind_sup_"):
        p = add_scatter(
            p = p,
            data = obj.ind_sup_.coord,
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_ind_sup,
            point_args = point_args_ind_sup,
            text_args = text_args_ind_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "PCoA - Graph of individuals"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show other points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = set_axis(
        p=p,
        obj=obj,
        axis=axis,
        x_lim=x_lim,
        y_lim=y_lim,
        x_label=x_label,
        y_label=y_label,
        title=title,
        subtitle=subtitle,
        pntheme=pntheme,
        **kwargs
    )
    return p

def fviz_pcoa_shepard(obj,
                      geom = ("point","line"),
                      color="black",
                      point_args = dict(size=1.5),
                      line_args = dict(linetype="dashed"),
                      x_lim=None,
                      y_lim=None,
                      x_label=None,
                      y_label=None,
                      title=None,
                      subtitle = None,
                      pntheme=theme_minimal(),
                      **kwargs):
    """
    Visualize Principal Coordinates Analysis - Shepard Diagram
    
    Principal Coordinates Analysis (from :class:`~scientisttools.PCoA`), also known as classical multidimensional scaling (MDS), is a method used to explore and visualize similarities or dissimilarities among a set of objects or samples. 
    :class:`~scientisttools.fviz_pcoa_shepard` provides plotnine-based elegant visualization of from :class:`~scientisttools.PCoA` outputs. It plots a Shepard diagram which is a scatter plot of InputDist and OutputDist.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCoA`.

    geom : str, list, tuple, default = ("point","line")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","line"). 

        * "point" to show only points.
        * "line" to show only line.
        * ("point","line") to show both points and line.

    color : str, default = "black"
        Color for scatter.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    line_args : dict, default = dict(linetype="dashed")
        A dictionary containing parameters (except color) for line (see `plotnine.geom_line <https://plotnine.org/reference/geom_line.html>`).

    x_lim : list, tuple, default = None
        The range of the plotted 'x' values.

    y_lim : list, tuple, default = None
        The range of the plotted 'y' values.

    x_label : str, default = None
        The label text of x. If None, then a x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then a y_label is chosen.
    
    title : str, default = None 
        The title of the graph you draw. If None, then a title is chosen.

    subtitle : str, default = None
        The subtitle of the graph you draw.

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_pcoa`
         Visualize Principal Coordinates Analysis.

    References
    ----------
    [1] `Sheparddiagram <https://github.com/Mthrun/DataVisualizations/blob/master/R/Sheparddiagram.R>`

    Examples
    --------
    >>> from scientisttools.datasets import autosmds
    >>> from scientisttools import PCoA, fviz_pcoa_shepard
    >>> clf = PCoA(ncp=2,ind_sup=(12,13,14))
    >>> clf.fit(autosmds)
    PCoA(ind_sup=(12,13,14),ncp=2)
    >>> p = fviz_pcoa_shepard(clf)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if PCoA class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCoA":
        raise TypeError("'obj' must be a PCoA class.")
    
    # input and output distance
    in_dist = obj.call_.dist.values
    out_dist = squareform(pdist(X=obj.ind_.coord,metric="euclidean"))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # create data
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    coord = DataFrame(dict(InDist=in_dist[triu_indices(in_dist.shape[0], k = 1)],OutDist=out_dist[triu_indices(out_dist.shape[0], k = 1)]))
    
    # initialization
    p = ggplot(coord,aes(x = "InDist",y = "OutDist"))

    # show points
    if "point" in geom:
        p = p + geom_point(color=color,**point_args)
    # show line
    if "line" in geom:
        p = p + geom_line(color=color,**line_args)
   
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ste x_label, y_label, title and subtitle
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x label
    if x_label is None:
        x_label = "Input Distances"
    # set y label
    if y_label is None:
        y_label = "Output Distances"
    # set title
    if title is None:
        title = "PCoA - Shepard Diagram"
    # set subtitle
    if subtitle is None:
        subtitle = ""
    p = p + labs(title = title, x = x_label, y = y_label, subtitle=subtitle)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x range and y range
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x range
    if x_lim is not None:
        p = p + xlim(x_lim)
    # set y range
    if y_lim is not None:
        p = p + ylim(y_lim)

    # set theme
    p = p + pntheme

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if kwargs is not None:
        p = p + theme(**kwargs)
    return p

def fviz_pcoa(obj,
              choice="ind",
              **kwargs):
    """
    Visualize Principal Coordinates Analysis

    Principal Coordinates Analysis (from :class:`~scientisttools.PCoA`), also known as classical multidimensional scaling (MDS), is a method used to explore and visualize similarities or dissimilarities among a set of objects or samples. 
    :class:`~scientisttools.fviz_pcoa` provides plotnine-based elegant visualization of from :class:`~scientisttools.PCoA` outputs.

    Parameters
    ----------
    obj : class
        an object of class :class:`~scientisttools.PCoA`.
    
    choice : {"ind","shepard"}, default = "ind"
        The graph to plot. Allowed values include one of ("ind","shepard"):

        * "ind" for the individuals graphs
        * "shepard" for Shepard Diagram
    
    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_pcoa_ind`: Graph of individuals
        * :class:`scientisttools.fviz_pcoa_shepard`: Shepard Diagram

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_pcoa_ind`
        Visualize Principal Coordinates Analysis - Graph of individuals
    :class:`~scientisttools.fviz_pcoa_shepard`
        Visualize Principal Coordinates Analysis - Shepard Diagram

    Examples
    --------
    >>> from scientisttools.datasets import autosmds
    >>> from scientisttools import PCoA, fviz_pcoa
    >>> clf = PCoA(ncp=2,ind_sup=(12,13,14))
    >>> clf.fit(autosmds)
    PCoA(ind_sup=(12,13,14),ncp=2)
    >>> # graph of individuals
    >>> p = fviz_pcoa(clf,choice="ind",repel=True)
    >>> print(p.show())
    >>> # shepard diagram
    >>> p = fviz_pcoa(clf,choice="shepard")
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_pcoa_ind(obj,**kwargs)
    elif choice == "shepard":
        return fviz_pcoa_shepard(obj,**kwargs)
    else:
        raise ValueError(f"{choice} is not supported.")