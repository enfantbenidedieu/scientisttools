# -*- coding: utf-8 -*-
from plotnine import ggplot, theme_minimal

#intern functions
from ._fviz import (
    add_arrow, 
    check_is_valid_axis, 
    check_is_valid_geom,
    fviz_circle,
    set_axis
)

def fviz_corcircle(obj,
                   axis = [0,1],
                   geom = ("arrow","text"),
                   repel = False,
                   col_var = "black",
                   segment_args = dict(size=0.5,alpha=1),
                   point_args = dict(size=1.5),
                   text_args = dict(size=8),
                   quanti_sup = True,
                   col_quanti_sup = "blue",
                   segment_args_quanti_sup = dict(linetype="dashed",size=0.5,alpha=1),
                   point_args_quanti_sup = dict(size=1.5),
                   text_args_quanti_sup = dict(size=8),
                   scale = 1,
                   circle = True,
                   col_circle = "gray",
                   x_lim = (-1.1,1.1),
                   y_lim = (-1.1,1.1),
                   x_label = None,
                   y_label = None,
                   title = None,
                   subtitle = None,
                   pntheme = theme_minimal(),
                   **kwargs):
    """
    Correlation Circle

    This function produces a correlation circle.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.CA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`,:class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`, :class:`~scientisttools.DMFA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","text"). 

        * "arrow" to plot only arrows.
        * "text" to show only labels.
        * ("arrow","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    col_var : str, default = "black"
        Color for variables segments and/or texts.

    segment_args : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color) for segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables segments and/or texts.

    col_quanti_sup : str, default = "blue"
        Color for supplementary continuous variables segments and/or texts.

    segment_args_quanti_sup : dict, default = dict(linetype="dashed",size=0.5,alpha=1)
        A dictionary containing parameters (except color) for supplementary continuous variables segments.

    point_args_quanti_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary continuous variables points.

    text_args_quanti_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary continuous variables texts.

    scale : int, default = 1
        The scale of factor coordinates.

    circle : bool, default = True
        If True, draw a circle.

    col_circle : str, default = "gray"
        Color for the circle.

    x_lim : list, tuple, default = (-1.1,1.1)
        The range of the plotted x values.

    y_lim : list, tuple, default = (-1.1,1.1)
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

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_corcircle
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,13))
    >>> clf.fit(decathlon.data)
    >>> # graph of variables
    >>> p = fviz_corcircle(clf)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid object
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__  in ("PCA","FA","CA","MCA","FAMD","PCAmix","MPCA","MFA","DMFA")):
        raise TypeError("'obj' must be an object of class PCA, FA, CA, MCA, FAMD, MPCA, PCAmix, MFA, DMFA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    check_is_valid_axis(obj=obj,axis=axis)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    check_is_valid_geom(geom=geom,axis=1)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # extract factor coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("PCA","FA","FAMD","PCAmix","MPCA","MFA","DMFA")) and hasattr(obj, "quanti_var_sup_"):
        coord = obj.quanti_var_sup_.coord
    else:
        coord = obj.quanti_var_.coord

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_arrow(
        p = ggplot(),
        data = coord.mul(scale),
        axis = axis,
        geom = geom,
        repel = repel,
        color = col_var,
        segment_args = segment_args,
        point_args = point_args,
        text_args = text_args
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    if (quanti_sup and 
        (obj.__class__.__name__ in ("PCA","FA","FAMD","PCAmix","MPCA","MFA","DMFA")) and 
        hasattr(obj,"quanti_var_sup_")):
        p = add_arrow(
            p = p,
            data = obj.quanti_var_sup_.coord.mul(scale),
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_quanti_sup,
            segment_args = segment_args_quanti_sup,
            point_args = point_args_quanti_sup,
            text_args = text_args_quanti_sup
        )
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show correlation circle
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if circle:
        p = fviz_circle(p=p,color=col_circle)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Correlation circle"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = set_axis(
        p = p,
        obj = obj,
        axis = axis,
        x_lim = x_lim,
        y_lim = y_lim,
        x_label = x_label,
        y_label = y_label,
        title = title,
        subtitle = subtitle,
        pntheme = pntheme,
        **kwargs
    )
    return p