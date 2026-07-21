# -*- coding: utf-8 -*-
from pandas import concat, Categorical
from mizani.palettes import brewer_pal
from plotnine import (
    aes,
    arrow,
    ggplot,
    geom_segment,
    geom_text,
    scale_color_manual,
    guides,
    guide_legend,
    theme_minimal
)

# intern functions
from ._fviz import (
    check_is_valid_axis,
    overlap_coord,
    fviz_arrow, 
    fviz_scatter, 
    add_scatter, 
    add_arrow, 
    set_axis, 
    fviz_circle
)

def fviz_dmfa_ind(obj,
                  axis = [0,1],
                  geom = ("point","text"),
                  repel = False,
                  point_args = dict(size=1.5),
                  text_args = dict(size=8),
                  palette = "Dark2",
                  add_ellipses = False, 
                  ellipse_type = "confidence",
                  level = 0.95,
                  alpha = 0.1,
                  ind_sup = True,
                  col_ind_sup = "blue",
                  point_args_ind_sup = dict(size=1.5),
                  text_args_ind_sup = dict(size=8),
                  quali_sup = True,
                  col_quali_sup = "violet",
                  point_args_quali_sup = dict(size=1.5),
                  text_args_quali_sup = dict(size=8),
                  lim_cos2 = None,
                  lim_contrib = None,
                  x_lim = None,
                  y_lim = None,
                  x_label = None,
                  y_label = None,
                  title = None,
                  subtitle = None,
                  pntheme = theme_minimal(),
                  **kwargs):
    """
    Visualize Dual Multiple Factor Analysis - Graph of individuals
    
    Dual Multiple Factor Analysis (:class:`~scientisttools.DMFA`) is used to analyze a data set in which variables are described by several sets of individuals structured into groups.
    :class:`~scientisttools.fviz_dmfa_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.DMFA` outputs for individuals.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.DMFA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting individuals text labels or not.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for individuals points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for individuals texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    add_ellipses : bool, default = False
        If True, draws ellipses around the points when habillage is not None.

    ellipse_type : str, default = "confidence"
        String specifying frame type. Possible values are : "convex", "confidence" or types supported by `plotnine.stat_ellipse <https://plotnine.org/reference/stat_ellipse.html>` including one of "t", "norm" or "euclid" for plotting concentration ellipses.

        * "convex": plot convex hull of a set of points as :class:`~scientisttools.data_ellipse`.
        * "confidence": plot confidence ellipses around group mean points as :class:`~scientisttools.data_ellipse`.
        * "t": assumes a multivariate t-distribution.
        * "norm": assumes a multivariate normal distribution.
        * "eulclid": draws a circle with the radius equal to `level`, representing the euclidean distance from the center.

    level : float, default = 0.95
        The size of the concentration ellipse in normal probability.
    
    alpha : float, default = 0.1
        The transparency level of fill color. Use alpha = 0 for no fill color.

    ind_sup : bool, default = True
        If True, show supplementary individuals points and/or texts.

    col_ind_sup : str, default = "blue"
        Color for supplementary individuals.

    point_args_ind_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary individuals points.

    text_args_ind_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary individuals texts.

    quali_sup : bool, default = True
        If True, then show supplementary variable categories points and/or texts.

    col_quali_sup : str, default = "violet"
        Color for supplementary variable categories points and/or texts.

    point_args_quali_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary variable categories points.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary variable categories texts.

    lim_cos2 : float, default = None
        The cos2 limit.

    lim_contrib : float, default = None
        The relative contribution limit.

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

    hline : bool, default = True
        If True, then add a horizontal line.

    vline : bool, default = True
        If True, then add a vertical line.

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.
    
    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_dmfa`
        Visualize Dual Multiple Factor Analysis
    :class:`~scientisttools.get_dmfa`
        Extract the results for individuals/variables/group - DMFA

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, fviz_dmfa_ind
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    >>> # graph of individuals
    >>> p = fviz_dmfa_ind(clf,repel=True)
    >>> print(p.show())
    """
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class DMFA
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj = obj,
        choice = "ind",
        axis = axis,
        geom = geom,
        repel = repel,
        color = "black",
        point_args = point_args,
        text_args = text_args,
        habillage = obj.call_.group[0],
        palette = palette,
        add_ellipses = add_ellipses, 
        ellipse_type = ellipse_type,
        level = level,
        alpha = alpha,
        lim_cos2 = lim_cos2,
        lim_contrib = lim_contrib
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ind_sup and hasattr(obj,"ind_sup_"):
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
    # show supplementary categories points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quali_sup and hasattr(obj,"levels_sup_"):
        p = add_scatter(
            p = p,
            data = obj.levels_sup_.coord,
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_quali_sup,
            point_args = point_args_quali_sup,
            text_args = text_args_quali_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "DMFA - Graph of individuals"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show others elements
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

def fviz_dmfa_var(obj,
                  choice = "var",
                  axis = [0,1],
                  geom = ("arrow","point","text"),
                  repel = False,
                  col_var = "black",
                  point_args = dict(size=1.5),
                  segment_args = dict(size=0.5),
                  text_args = dict(size=8),
                  gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                  legend_title = None,
                  palette = "Dark2",
                  quanti_sup = True,
                  col_quanti_sup = "violet",
                  segment_args_quanti_sup = dict(linetype="dashed",size=0.5),
                  text_args_quanti_sup = dict(size=8),
                  scale = 1,
                  lim_cos2 = None,
                  lim_contrib = None,
                  circle = True,
                  col_circle = "gray",
                  x_lim = None,
                  y_lim = None,
                  x_label = None,
                  y_label = None,
                  title = None,
                  subtitle = None,
                  pntheme = theme_minimal(),
                  **kwargs):
    """
    Visualize Dual Multiple Factor Analysis - Graph of variables

    Dual Multiple Factor Analysis (:class:`~scientisttools.DMFA`) is used to analyze a data set in which variables are described by several sets of individuals structured into groups.
    :class:`~scientisttools.fviz_dmfa_var` provides plotnine-based elegant visualization of :class:`~scientisttools.DMFA` outputs for variables and groups.
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.DMFA`.

    choice : {"group","var"}, default = "group"
        The graph to plot. Allowed values include:

        * "group" for groups
        * "var" for variables (=correlation circle)
    
    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("arrow","point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text"). 

        * "arrow" to show only arrows
        * "point" to show only points.
        * "text" to show only labels.
        * ("arrow","text") to show both arrows and texts.
        * ("point","text") to show both points and texts.
    
    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    col_var : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for variables or variable categories. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for variablescategories are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 

    segment_args : dict, default = dict(size = 0.5)
        A dictionary containing parameters  (except color and arrow) for variables segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for variables points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for variables texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high values.

    legend_title : str, defaut = None
        The title of the legend. If None, then a legend title is chosen.

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    add_ellipses : bool, default = False
        If True, draws ellipses around the points.

    ellipse_type : str, default = "confidence"
        String specifying frame type. Possible values are : "convex", "confidence" or types supported by `plotnine.stat_ellipse <https://plotnine.org/reference/stat_ellipse.html>` including one of "t", "norm" or "euclid" for plotting concentration ellipses.

        * "convex": plot convex hull of a set of points as :class:`~scientisttools.data_ellipse`.
        * "confidence": plot confidence ellipses around group mean points as :class:`~scientisttools.data_ellipse`.
        * "t": assumes a multivariate t-distribution.
        * "norm": assumes a multivariate normal distribution.
        * "eulclid": draws a circle with the radius equal to `level`, representing the euclidean distance from the center.

    level : float, default = 0.95
        The size of the concentration ellipse in normal probability.
    
    alpha : float, default = 0.1
        The transparency level of fill color. Use alpha = 0 for no fill color.

    var_sup : bool, default = True
        If True, then show supplementary variables points and/or texts.

    col_var_sup : str, default = "blue"
        Color for supplementary variables or variables categories points and/or texts.

    segment_args_var_sup : dict, default = dict(linetype="dashed",size = 0.5)
        A dictionary containing parameters (except color and arrow) for supplementar variables segments.

    point_args_var_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary variables points.

    text_args_var_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary variables texts.

    scale : int, default = 1
        The scale of factor coordinates.

    circle : bool, default = True
        If True, draw a circle.

    col_circle : str, default = "gray"
        Color for the circle.

    lim_cos2 : float, default = None
        The cos2 limit.

    lim_contrib : float, default = None
        The relative contribution limit.

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
    :class:`~scientisttools.fviz_dmfa`
        Visualize Dual Multiple Factor Analysis
    :class:`~scientisttools.get_dmfa`
        Extract the results for individuals/variables/group - DMFA

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, fviz_dmfa_var
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    >>> # graph of variables
    >>> p = fviz_dmfa_var(clf,repel=True)
    >>> print(p.show())
    >>> # graph of groups
    >>> p = fviz_dmfa_var(clf,choice="group",repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class DMFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)
    
    # show group points
    if choice == "group":
        # show active groups points
        p = fviz_scatter(
            obj = obj,
            choice = "group",
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_var,
            point_args = point_args,
            text_args = text_args,
            gradient_cols = gradient_cols,
            legend_title = legend_title,
            palette = palette
        )

        # set title
        if title is None:
            title = "DMFA - Graph of groups"
    elif choice == "var":
        if col_var == "group":
            # compromise space
            coord = obj.quanti_var_.coord.mul(scale)
            # insert habillage
            coord["habillage"] = "var"
            # add partial coordinates
            index = list(obj.var_partiel_._fields)
            for i,k in enumerate(index):
                data = obj.var_partiel_[i].mul(scale)
                # insert habillage
                data["habillage"] = k 
                # concatenate
                coord = concat((coord,data),axis=0)
            # convert to categorical
            coord["habillage"] = Categorical(coord["habillage"],categories=index+["var"])

            # set colors
            if isinstance(palette,str):
                colors = brewer_pal(type="qual", palette=palette)(len(index))
            elif isinstance(palette,(list,tuple)):
                if len(palette) != len(index):
                    raise TypeError("Not convenient palette definition")
                colors = palette
            else:
                raise TypeError("palette should be one of str, list of tuple")
            # set color mapping
            colors_mapping = dict(zip(index,colors))
            colors_mapping["var"] = "black"

            # define text coordinates
            coord = overlap_coord(coord=coord,axis=axis,repel=repel)
            # set x, y for texts
            if repel:
                x_text, y_text = "xnew", "ynew"
            else:
                x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"

            # initialization
            p = ggplot(
                data=coord,
                mapping=aes(x=f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",color="habillage",label=coord.index)
            )
            # show segments
            if "arrow" in geom:
                p = (
                    p 
                    + geom_segment(
                        mapping=aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}",color="habillage"), 
                        arrow = arrow(angle=30,length=0.2/2.54),
                        **segment_args
                    )
                )
            # show points
            if "point" in geom:
                p = p + geom_text(aes(color="habillage"),**point_args)
            # show texts
            if "text" in geom:
                p = p + geom_text(aes(x=x_text,y=y_text,color="habillage"),**text_args,show_legend=False)

            # set color manual
            p = (
                p 
                + scale_color_manual(values=colors_mapping) 
                + guides(color=guide_legend(title=""))
            )

            # show correlation circle
            if circle:
                p = fviz_circle(p=p,color=col_circle)
        else:
            p = fviz_arrow(
                obj = obj,
                choice="quanti_var",
                axis = axis,
                geom = geom,
                repel = repel,
                lim_cos2 = lim_cos2,
                lim_contrib = lim_contrib,
                color = col_var,
                segment_args = segment_args,
                text_args = text_args,
                gradient_cols = gradient_cols,
                legend_title = legend_title,
                palette = palette,
                scale = scale,
                circle = circle,
                col_circle = col_circle
            )

        # show supplementary continuous variables
        if quanti_sup and hasattr(obj,"quanti_var_sup_"):
            p = add_arrow(
                p=p,
                data=obj.quanti_var_sup_.coord.mul(scale),
                axis=axis,
                geom=geom,
                repel=repel,
                color=col_quanti_sup,
                segment_args=segment_args_quanti_sup,
                text_args=text_args_quanti_sup
            )
        
        # set x limits
        if x_lim is None:
            x_lim = (-1.1,1.1)
        # set y limits
        if y_lim is None:
            y_lim = (-1.1,1.1)
        # set title
        if title is None:
            title = "DMFA - Graph of variables"
    else:
        raise ValueError("choice should be one of 'var', 'group'")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add others elements
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

def fviz_dmfa(obj,
              choice="ind",
              **kwargs):
    """
    Visualize Dual Multiple Factor Analysis
    
    Dual Multiple Factor Analysis (:class:`~scientisttools.DMFA`) is used to analyze a data set in which variables are described by several sets of individuals structured into groups.
    :class:`~scientisttools.fviz_dmfa` provides plotnine-based elegant visualization of :class:`~scientisttools.DMFA` outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.DMFA`.

    choice : {"ind","var","group"}, default = "ind"
        The graph to plot. Allowed values include one of : 

        * "ind" for the individuals graphs
        * "var" for the variables graphs (= Correlation circle)
        * "group" for groups graphs

    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_dmfa_ind`: Graph of individuals
        * :class:`scientisttools.fviz_dmfa_var`: Graph of variables

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_dmfa_ind`
        Visualize Dual Multiple Factor Analysis - Graph of individuals
    :class:`~scientisttools.fviz_dmfa_var`
        Visualize Dual Multiple Factor Analysis - Graph of variables
    :class:`~scientisttools.get_dmfa`
        Extract the results for individuals/variables/group - DMFA

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, fviz_dmfa
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    >>> # graph of individuals
    >>> p = fviz_dmfa(clf,choice="ind",repel=True)
    >>> print(p.show())
    >>> # graph of variables
    >>> p = fviz_dmfa(clf,choice="var",repel=True)
    >>> print(p.show())
    >>> # graph of groups
    >>> p = fviz_dmfa(clf,choice="group",repel=True)
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_dmfa_ind(obj,**kwargs)
    elif choice in ("var","group"):
        return fviz_dmfa_var(obj,choice=choice,**kwargs)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'group'")