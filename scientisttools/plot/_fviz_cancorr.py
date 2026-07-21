# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from mizani.palettes import brewer_pal
from plotnine import (
    aes,
    arrow,
    geom_abline,
    geom_point,
    geom_segment,
    geom_polygon,
    geom_text,
    ggplot,
    guides,
    guide_legend,
    scale_color_manual,
    stat_ellipse,
    stat_smooth,
    theme_minimal,
)

# intern functions
from ._fviz import (
    check_is_valid_axis,
    check_is_valid_geom,
    fviz_circle,
    overlap_coord,
    set_axis
)
from ..methods.others import ellipse, convexhull

def fviz_cancorr_ind(obj,
                     element = "X",
                     axis = [0,1],
                     geom = ("point","text"),
                     repel = False,
                     col_ind = "black",
                     point_args = dict(size=1.5),
                     text_args = dict(size=8),
                     palette = "Dark2",
                     x_lim = None,
                     y_lim = None,
                     x_label = None,
                     y_label = None,
                     title = None,
                     subtitle = None,
                     pntheme = theme_minimal(),
                     **kwargs):
    """
    Visualize Canonical Correlation Analysis - Graph of individuals

    Canonical correlation analysis (:class:`~scientisttools.CANCORR) seeks a linear combination of one set of variables and a linear combination of a second set of variables such that the correlation is maximized. 
    It is similar to regression, which seeks a linear combination of a set of variables that maximizes the correlation with a single (response) variable.
    :class:`~scientisttools.fviz_cancorr_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.CANCORR` outputs for individuals.

    Parameters
    ----------
    obj : class
        An object of class:class:`~scientisttools.CANCORR`.

    element : str, default = "X"
        The element to be used for points. Allowed values are :

        * "X" for first group.
        * "Y" for second group.
        * "XY" for both groups.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    col_ind : str, default = "black"
        Color for individuals.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

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
    :class:`~scientisttools.fviz_cancorr`
        Visualize Canonical Correlation Analysis.

    Examples
    --------
    >>> from scientisttools.datasets import fitnessclub
    >>> from scientisttools import CANCORR, fviz_cancorr_ind
    >>> clf = CANCORR(scale_unit=False,ncp=3,group=(3,3),name_group=("Physiological","Exercises"))
    >>> clf.fit(fitnessclub)
    >>> # graph of individuals
    >>> p = fviz_cancorr_ind(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a CANCORR class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CANCORR":
        raise TypeError("'obj' must be a CANCORR class.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid element
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (element in ("X","Y","XY")):
        raise ValueError("'element' should be one of 'X', 'Y' or 'XY'.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=0)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set text arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and ("text" in geom):
        text_args["adjust_text"] = dict(arrowprops=dict(lw=1.0))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # coordinates for individuals
    if element in ("X","Y"):
        coord = obj.ind_[0] if element == "X" else obj.ind_[1]
        coord.columns = [f"Can{x+1}" for x in range(coord.shape[1])]
        coord = coord.reset_index().rename(columns={"index" : "rownames"})
    else:
        coord = DataFrame().astype("float")
        for i, k in enumerate(obj.call_.name_group):
            data = obj.ind_[i].copy()
            data.columns = [f"Can{x+1}" for x in range(data.shape[1])]
            data.loc[:,"habillage"] = k
            data = data.reset_index().rename(columns={"index" : "rownames"})
            coord = concat((coord,data),axis=0,ignore_index=True)
        coord["habillage"] = coord["habillage"].astype("category")
    
    # initialize
    p = ggplot(data=coord,mapping=aes(x=f"Can{axis[0]+1}", y=f"Can{axis[1]+1}",label="rownames"))
    # show points
    if "point" in geom:
        p = p + (geom_point(color=col_ind,**point_args) if element in ("X","Y") else geom_point(aes(color="habillage"),**point_args))
    # show texts
    if "text" in geom:
        p = p + (geom_text(color=col_ind,**text_args) if element in ("X","Y") else geom_text(aes(color="habillage"),**text_args))

    # set color
    if element == "XY":
        # set colors
        index = coord["habillage"].unique().tolist()
        if isinstance(palette,str):
            colors = brewer_pal(type="qual", palette=palette)(len(index))
        elif isinstance(palette,(list,tuple)):
            if len(palette) != len(index):
                raise TypeError("Not convenient palette definition")
            colors = palette
        else:
            raise TypeError("palette should be one of str, list or tuple")
        # set color mapping
        colors_mapping = dict(zip(index,colors))

        # set color
        p = (
            p 
            + scale_color_manual(values=colors_mapping)
            + guides(color=guide_legend(title=""))
        )

    # set x label
    if x_label is None:
        x_label = f"Can{str(axis[0]+1)} ({round(obj.eig_.iloc[axis[0],2],1)}%)"
    # set y label
    if y_label is None:
        y_label = f"Can{str(axis[1]+1)} ({round(obj.eig_.iloc[axis[1],2],1)}%)"
    # set title
    if title is None:
        title = "CANCORR - Graph of individuals"
    # set subtitle
    if subtitle is None:
        if element in ("X","Y"):
            name = obj.call_.name_group[0] if element == "X" else obj.call_.name_group[1]
        else:
            name = f"{obj.call_.name_group[0]} and the {obj.call_.name_group[1]}"
        subtitle = f"Among the {name}"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show others points
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

def fviz_cancorr_var(obj,
                     element = "X",
                     axis = [0,1],
                     geom = ("arrow","text"),
                     repel = False,
                     segment_args = dict(size=0.5,alpha=1),
                     text_args = dict(size=8),
                     palette = "Dark2",
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
    Visualize Canonical Correlation Analysis - Graph of variables
    
    Canonical correlation analysis (:class:`~scientisttools.CANCORR) seeks a linear combination of one set of variables and a linear combination of a second set of variables such that the correlation is maximized. 
    It is similar to regression, which seeks a linear combination of a set of variables that maximizes the correlation with a single (response) variable.
    :class:`~scientisttools.fviz_cancorr_var` provides plotnine-based elegant visualization of :class:`~scientisttools.CANCORR` outputs for variables.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CANCORR`.

    element : str, default = "X"
        The element to be used for points. Allowed values are :

        * "X" for first group.
        * "Y" for second group.

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
        A dictionary containing parameters  (except color and arrow) for segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

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
    :class:`~scientisttools.fviz_cancorr`
        Visualize Canonical Correlation Analysis.

    Examples
    --------
    >>> from scientisttools.datasets import fitnessclub
    >>> from scientisttools import CANCORR, fviz_cancorr_var
    >>> clf = CANCORR(scale_unit=False,ncp=3,group=(3,3),name_group=("Physiological","Exercises"))
    >>> clf.fit(fitnessclub)
    >>> # graph of variables
    >>> p = fviz_cancorr_var(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class CANCOR
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CANCORR":
        raise TypeError("'obj' must be a CANCORR object")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid element
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (element in ("X","Y")):
        raise ValueError("'element' should be one of 'X' ot 'Y'")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=1)
    
    # Extract scores
    if element == "X":
        xcoord, ycoord = obj.quanti_var_[0].xscores, obj.quanti_var_[0].yscores
    else:
        xcoord, ycoord = obj.quanti_var_[1].xscores, obj.quanti_var_[1].yscores
    # set columns
    xcoord.columns, ycoord.columns = [f"Dim{x+1}" for x in range(xcoord.shape[1])], [f"Dim{x+1}" for x in range(ycoord.shape[1])]
    # add 
    xcoord.loc[:,"habillage"] = obj.call_.name_group[0]
    ycoord.loc[:,"habillage"] = obj.call_.name_group[1]
    # concatenate
    coord = concat((xcoord,ycoord),axis=0,ignore_index=False).reset_index().rename(columns={"index" : "rownames"})

    # set colors
    index = coord["habillage"].unique().tolist()
    if isinstance(palette,str):
        colors = brewer_pal(type="qual", palette=palette)(len(index))
    elif isinstance(palette,(list,tuple)):
        if len(palette) != len(index):
            raise TypeError("Not convenient palette definition")
        colors = palette
    else:
        raise TypeError("palette should be one of str, list or tuple")
    # set color mapping
    colors_mapping = dict(zip(index,colors))

    # define text coordinates
    coord = overlap_coord(coord=coord,axis=axis,repel=repel)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x, y for texts
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel:
        x_text, y_text = "xnew", "ynew"
    else:
        x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"
    
    # initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",label="rownames"))

    # show segments
    if "arrow" in geom:
        p = (
            p 
            + geom_segment(
                mapping = aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}",color="habillage"),
                arrow = arrow(angle=30,length=0.2/2.54),
                **segment_args
            ) 
        )
    # show texts
    if "text" in geom:
        p = p + geom_text(aes(x=x_text,y=y_text,color="habillage"),**text_args,show_legend=False)

    # set color manual
    p = (
        p 
        + scale_color_manual(values=colors_mapping)
        + guides(color=guide_legend(title=""))
    )
    
    # create circle
    if circle:
        p = fviz_circle(p=p,color=col_circle)

    # set x label
    if x_label is None:
        x_label = f"Can{str(axis[0]+1)} ({round(obj.eig_.iloc[axis[0],2],1)}%)"
    # set y label
    if y_label is None:
        y_label = f"Can{str(axis[1]+1)} ({round(obj.eig_.iloc[axis[1],2],1)}%)"
    # set title
    if title is None:
        title = "CANCORR - Graph of variables"
    # set subtitle
    if subtitle is None:
        if element in ("X","Y"):
            name = obj.call_.name_group[0] if element == "X" else obj.call_.name_group[1]
        else:
            name = f"{obj.call_.name_group[0]} and the {obj.call_.name_group[1]}"
        subtitle = f"Among the {name}"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show others points
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

def fviz_cancorr_scatter(obj,
                         axis = 0,
                         geom = ("point","text"),
                         repel = False,
                         col_ind = "black",
                         point_args = dict(size=1.5),
                         text_args = dict(size=8),
                         smooth = True,
                         col_smooth = "green",
                         smooth_args = dict(method="loess",se=False),
                         abline = True,
                         col_abline = "red",
                         abline_args = dict(linetype="dashed",size=1.5),
                         add_ellipses = True,
                         ellipse_type = "confidence",
                         col_ellipse = "blue",
                         level = 0.95,
                         x_lim = None,
                         y_lim = None,
                         x_label = None,
                         y_label = None,
                         title = None,
                         subtitle = None,
                         pntheme = theme_minimal(),
                         **kwargs):
    """
    Visualize Canonical Correlation Analysis - Scatter plot

    Canonical correlation analysis (:class:`~scientisttools.CANCORR) seeks a linear combination of one set of variables and a linear combination of a second set of variables such that the correlation is maximized. 
    It is similar to regression, which seeks a linear combination of a set of variables that maximizes the correlation with a single (response) variable.
    :class:`~scientisttools.fviz_cancorr_scatter` provides plotnine-based elegant visualization of :class:`~scientisttools.CANCORR` outputs to help visualize X, Y data in canonical space.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CANCORR`.

    axis : int, default = 1
        The dimension to plot.

    geom : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","text"). 

        * "arrow" to plot only arrows.
        * "text" to show only labels.
        * ("arrow","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    col_ind : str, default = "black"
        Color for individuals.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    smooth : bool, default = True
        If True, draw a (loess) smoothed curve for Ycan.iloc[,axis] on Xcan.iloc[,axis].

    col_smooth : str, default = "green"
        The color for loess smoothed curve.
 
    smooth_args : dict, default = dict(method="loess",se=False)
        A dictionary containing parameters (except color) for smoothed curve (see `plotnine.stat_smooth <https://plotnine.org/reference/stat_smooth.html>`).

    abline : bool, default = True
        If True, draw the linear regression line for Ycan.iloc[,axis] on Xcan.iloc[,axis].

    col_abline : str, default = "red"
        Color for the linear regression line.

    abline_args : dict, default = dict(linetype="dashed",size=1.5)
        A dictionary containing parameters (except color) for linear regression (see `plotnine.geom_abline <https://plotnine.org/reference/geom_abline.html>`).

    add_ellipses : bool, default = False
        If True, draws ellipses around the canonical scores.

    ellipse_type : str, default = "confidence"
        String specifying frame type. Possible values are : "convex", "confidence" or types supported by `plotnine.stat_ellipse <https://plotnine.org/reference/stat_ellipse.html>` including one of "t", "norm" or "euclid" for plotting concentration ellipses.

        * "convex": plot convex hull of a set of points as :class:`~scientisttools.convexhull`.
        * "confidence": plot confidence ellipses around group mean points as :class:`~scientisttools.ellipse`.
        * "t": assumes a multivariate t-distribution.
        * "norm": assumes a multivariate normal distribution.
        * "eulclid": draws a circle with the radius equal to `level`, representing the euclidean distance from the center.

    col_ellipse : str, default = "blue"
        Color for ellipse.

    level : float, default = 0.95
        The size of the concentration ellipse in normal probability.

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
    :class:`~scientisttools.fviz_cancorr`
        Visualize Canonical Correlation Analysis.

    Examples
    --------
    >>> from scientisttools.datasets import fitnessclub
    >>> from scientisttools import CANCORR, fviz_cancorr_scatter
    >>> clf = CANCORR(scale_unit=False,ncp=3,group=(3,3),name_group=("Physiological","Exercises"))
    >>> clf.fit(fitnessclub)
    >>> # canonical correlation analysis scatter points
    >>> p = fviz_cancorr_scatter(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class CANCOR
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CANCORR":
        raise TypeError("'obj' must be a CANCORR object")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(axis,int):
        raise TypeError("'axis' must be an integer")
    if not (axis in list(range(obj.call_.ncp))):
        raise ValueError("'which' should be either 0 or 1")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=0)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set text arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and ("text" in geom):
        text_args["adjust_text"] = dict(arrowprops=dict(arrowstyle='-',lw=1.0))
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    scores = (
        concat((obj.ind_[0].iloc[:,axis].to_frame("X"),obj.ind_[1].iloc[:,axis].to_frame("Y")),
               axis=1)
        .reset_index()
        .rename(columns={"index":"rownames"})
    )

    # initialize
    p = ggplot(data=scores,mapping=aes(x="X",y="Y",label="rownames"))
    # show points
    if "point" in geom:
        p = p + geom_point(color=col_ind,**point_args)
    # show texts
    if "text" in geom:
        p = p + geom_text(color=col_ind,**text_args)
    # show abline line
    if abline:
        p = p + geom_abline(color=col_abline,**abline_args)
    # show loess line
    if smooth:
        p = p + stat_smooth(color=col_smooth,**smooth_args)
    # show ellipse circle
    if add_ellipses:
        if ellipse_type in ("confidence","convex"):
            if ellipse_type == "confidence":
                data = ellipse(X=scores,level=level)
            else:
                data = convexhull(X=scores)

            # add to plot
            p = (
                p 
                + geom_polygon(
                    data = data,
                    mapping = aes(
                        x = "X",
                        y = "X",
                    ), 
                    color = col_ellipse,
                    fill = col_ellipse,
                    inherit_aes = False
                )
            )
        else:
            p = (
                p 
                + stat_ellipse(
                    geom  = "polygon",
                    type = ellipse_type,
                    color = col_ellipse,
                    fill = col_ellipse
                )
            )
    
    # set x label
    if x_label is None:
        x_label = f"{obj.call_.name_group[0]}{axis+1}"
    # set y label
    if y_label is None:
        y_label = f"{obj.call_.name_group[1]}{axis+1}"
    # set title
    if title is None:
        title = "Canonical Correlation Analysis"
    # set subtitle
    if subtitle is None:
        subtitle = f"Cor = {round(obj.cancorr_.iloc[axis,0],2)}"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show others points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = set_axis(
        p = p,
        obj = obj,
        axis = [axis,axis],
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

def fviz_cancorr(obj,
                 choice="ind",
                 **kwargs):
    """
    Visualize Canonical Correlation Analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CANCORR`.

    choice : {"ind","scatter","var"}, default = "ind"
        The graph to plot. Allowed values are:

        * 'ind' for graph of individuals
        * 'scatter' for canonical correlation analysis scatter points
        * 'var' for graph of variables (=correlation circle)
    
    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_cancorr_ind`: Graph of individuals
        * :class:`scientisttools.fviz_cancorr_var`: Graph of variables (=correlation circle)
        * :class:`scientisttools.fviz_cancorr_scatter`: Canonical correlation analysis scatter points

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_cancorr_ind`
        Visualize Canonical Correlation Analysis - Graph of individuals
    :class:`~scientisttools.fviz_cancorr_scatter`
        Visualize Canonical Correlation Analysis - Canonical correlation analysis scatter plot
    :class:`~scientisttools.fviz_cancorr_var`
        Visualize Canonical Correlation Analysis - Graph of variables.

    Examples
    --------
    >>> from scientisttools.datasets import fitnessclub
    >>> from scientisttools import CANCORR, fviz_cancorr
    >>> clf = CANCORR(scale_unit=False,ncp=3,group=(3,3),name_group=("Physiological","Exercises"))
    >>> clf.fit(fitnessclub)
    >>> # graph of individuals
    >>> p = fviz_cancorr(clf,choice="ind",repel=True)
    >>> print(p.show())
    >>> # graph of variables
    >>> p = fviz_cancorr(clf,choice="var",repel=True)
    >>> print(p.show())
    >>> # scatter points
    >>> p = fviz_cancorr(clf,choice="scatter",repel=True)
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_cancorr_ind(obj=obj,**kwargs)
    elif choice == "scatter":
        return fviz_cancorr_scatter(obj=obj,**kwargs)
    elif choice == "var":
        return fviz_cancorr_var(obj=obj,**kwargs)
    else:
        raise ValueError("choice should be one of 'ind', 'scatter' or 'var'")