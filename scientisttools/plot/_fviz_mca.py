# -*- coding: utf-8 -*-
from plotnine import ggplot, theme_minimal

#intern functions
from ._fviz import (
    add_scatter,
    check_is_valid_axis, 
    check_is_valid_geom,
    fviz_scatter,
    set_axis
)
from ._fviz_corcircle import fviz_corcircle

def fviz_mca_ind(obj,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 col_ind ="black",
                 point_args = dict(size=1.5),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 habillage = None,
                 palette = "Dark2",
                 add_ellipses = False, 
                 ellipse_type = "confidence", 
                 level = 0.95,
                 alpha = 0.1,
                 ind_sup = True,
                 col_ind_sup = "blue",
                 point_args_ind_sup = dict(size=1.5),
                 text_args_ind_sup = dict(size=8),
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
    Visualize Multiple Correspondence Analysis - Graph of individuals
    
    Multiple Correspondence Analysis (:class:`~scientisttools.MCA`) is an extension of simple (:class:`~scientisttools.CA) to analyse a data table containing more than two categorical variables. 
    :class:`~scientisttools.fviz_mca_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.MCA` outputs for individuals.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting individuals text labels or not.

    col_ind : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for individuals. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for individuals are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 
        To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for individuals points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for individuals texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high values.

    legend_title : str, defaut = None
        The title of the legend. If None, then a legend title is chosen.

    habillage : str, int, default = None 
        The name of variable for coloring the observations by groups.

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
    :class:`~scientisttools.fviz_mca`
        Visualize Multiple Correspondence Analysis
    :class:`~scientisttools.get_mca`
        Extract the results for individuals/variables - MCA

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, fviz_mca_ind
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison.data)
    >>> # graph of individuals
    >>> p = fviz_mca_ind(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class MCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MCA":
        raise TypeError("'obj' must be a MCA class")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj = obj,
        choice = "ind",
        axis = axis,
        geom = geom,
        repel = repel,
        color = col_ind,
        point_args = point_args,
        text_args = text_args,
        gradient_cols = gradient_cols,
        legend_title = legend_title,
        habillage = habillage,
        palette = palette,
        add_ellipses = add_ellipses, 
        ellipse_type = ellipse_type,
        level = level,
        alpha = alpha,
        lim_cos2 = lim_cos2,
        lim_contrib = lim_contrib
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add supplementary individuals
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ind_sup and hasattr(obj,"ind_sup_"):
        p = add_scatter(
            p=p,
            data=obj.ind_sup_.coord,
            axis=axis,
            geom=geom,
            repel=repel,
            color=col_ind_sup,
            point_args=point_args_ind_sup,
            text_args=text_args_ind_sup
        )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "MCA - Graph of individuals"

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

def fviz_mca_var(obj,
                 choice = "var",
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 col_var = "black",
                 point_args = dict(size=1.5),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = "Dark2",
                 add_ellipses = False, 
                 ellipse_type = "convex",
                 level = 0.95,
                 alpha = 0.1,
                 quali_sup = True,
                 col_quali_sup = "blue",
                 point_args_quali_sup = dict(size=1.5),
                 text_args_quali_sup = dict(size=8),
                 quanti_sup = True,
                 col_quanti_sup = "red",
                 point_args_quanti_sup = dict(size=1.5),
                 text_args_quanti_sup = dict(size=8),
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
    Visualize Multiple Correspondence Analysis - Graph of variables
    
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. 
    :class:`~scientisttools.fviz_mca_var` provides plotnine-based elegant visualization of :class:`~scientisttools.MCA` outputs for categories.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    choice : {"levels","var"}, default = "var"
        The graph to plot. Allowed values include:

        * "levels" for variable categories
        * "var" (default) for plotting the correlation between variables and principal dimensions
    
    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting variables or variable categories text labels or not.

    col_var : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for variables or variable categories. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for variablescategories are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

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

        * "convex": plot convex hull of a set of points as :class:`~scientisttools.convex_ellipse`.
        * "confidence": plot confidence ellipses around group mean points as :class:`~scientisttools.confidence_ellipse`.
        * "t": assumes a multivariate t-distribution.
        * "norm": assumes a multivariate normal distribution.
        * "eulclid": draws a circle with the radius equal to `level`, representing the euclidean distance from the center.

    level : float, default = 0.95
        The size of the concentration ellipse in normal probability.
    
    alpha : float, default = 0.1
        The transparency level of fill color. Use alpha = 0 for no fill color.

    quali_sup : bool, default = True
        If True, then show supplementary variables or variable categories points and/or texts.

    col_quali_sup : str, default = "blue"
        Color for supplementary variables or variable categories points and/or texts.

    point_args_quali_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary variables or variable categories points.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary variables or variable categories texts.

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables points and/or texts.

    col_quanti_sup : str, default = "red"
        Color for supplementary continuous variables points and/or texts.

    point_args_quanti_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary continuous variables points.

    text_args_quanti_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary continuous variables texts.

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
    :class:`~scientisttools.fviz_mca`
        Visualize Multiple Correspondence Analysis
    :class:`~scientisttools.get_mca`
        Extract the results for individuals/variables - MCA

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, fviz_mca_var
    >>> clf = MCA(sup_var=(0,1,2,3))
    >>> clf.fit(poison.data)
    >>> # graph of categories
    >>> p = fviz_mca_var(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class MCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MCA":
        raise TypeError("'obj' must be a MCA class")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # graph of variable categories
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if choice == "levels":
        # show active variable categories points and texts
        p = fviz_scatter(
            obj = obj,
            choice = "levels",
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_var,
            point_args = point_args,
            text_args = text_args,
            gradient_cols = gradient_cols,
            legend_title = legend_title,
            palette = palette,
            add_ellipses = add_ellipses, 
            ellipse_type = ellipse_type,
            level = level,
            alpha = alpha,
            lim_cos2 = lim_cos2,
            lim_contrib = lim_contrib
        )
        
        # show supplementary variable categories points
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
    
        # set title
        if title is None:
            title = "MCA - Graph of variable categories"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # graph of variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif choice == "var":
        # check if valid axis
        check_is_valid_axis(obj=obj,axis=axis)
        # cehck if valid geom
        check_is_valid_geom(geom=geom,axis=0)

        # show active categorical variables points
        p = add_scatter(
            p=ggplot(),
            data=obj.quali_var_.coord,
            axis=axis,
            geom=geom,
            repel=repel,
            color=col_var,
            point_args=point_args,
            text_args=text_args
        )

        # show supplementary categorical variables points
        if quali_sup and hasattr(obj,"quali_var_sup_"):
            p = add_scatter(
                p=p,
                data=obj.quali_var_sup_.coord,
                axis=axis,
                geom=geom,
                repel=repel,
                color=col_quali_sup,
                point_args=point_args_quali_sup,
                text_args=text_args_quali_sup
            )

        # show supplementary continuous variables points
        if quanti_sup and hasattr(obj,"quanti_var_sup_"):
            p = add_scatter(
                p=p,
                data=obj.quanti_var_sup_.cos2,
                axis=axis,
                geom=geom,
                repel=repel,
                color=col_quanti_sup,
                point_args=point_args_quanti_sup,
                text_args=text_args_quanti_sup
            )
        
        # set title
        if title is None:
            title = "MCA - Graph of variables"
    else:
        raise ValueError("choice should be one of levels, var")

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

def fviz_mca_biplot(obj,
                    axis=[0,1],
                    geom_ind = ("point","text"),
                    geom_var = ("point","text"),
                    repel_ind = False,
                    repel_var = False,
                    col_ind = "black",
                    point_args_ind = dict(size=1.5),
                    text_args_ind = dict(size=8),
                    col_var = "steelblue",
                    point_args_var = dict(size=1.5),
                    text_args_var = dict(size=8),
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    legend_title = None,
                    habillage = None,
                    palette = None,
                    add_ellipses = False, 
                    ellipse_type = "confidence",
                    level = 0.95,
                    alpha = 0.1,
                    ind_sup = False,
                    col_ind_sup = "red",
                    point_args_ind_sup = dict(size=1.5),
                    text_args_ind_sup = dict(size=8),
                    quali_sup = False,
                    col_quali_sup = "blue",
                    point_args_quali_sup = dict(size=1.5),
                    text_args_quali_sup = dict(size=8),
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = None,
                    subtitle = None,
                    pntheme = theme_minimal(),
                    **kwargs):
    """
    Visualize Multiple Correspondence Analysis - Biplot of individuals and variable categories
    
    Multiple Correspondence Analysis (:class:`~scientisttools.MCA`) is an extension of simple CA to analyse a data table containing more than two categorical variables. 
    :class:`~scientisttools.fviz_mca_biplot` provides plotnine-based elegant visualization of :class:`~scientisttools.MCA` outputs for individuals and variable categories.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom_ind : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.

    geom_var : str, list, tuple, default = ("point","text")
        See ``geom_ind``.
    
    repel_ind : bool, default = True
        Whether to avoid overplotting individuals text labels or not.

    repel_var : bool, default = True
        See ``repel_ind``.

    color_ind : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for individuals. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for individuals are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 
        To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    point_args_ind : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for individuals points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args_ind : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for individuals texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    col_var : str, default = "steelblue"
        Color for column variables

    point_args_var : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for variable categories points.

    text_args_var : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for variable categories texts.

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high values.

    legend_title : str, defaut = None
        The title of the legend. If None, then a legend title is chosen.

    habillage : str, int, default = None 
        The name of variable for coloring the observations by groups.

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

    col_ind_sup : str, default = "red"
        Color for supplementary individuals.

    point_args_ind_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary individuals points.

    text_args_ind_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary individuals texts.

    quali_sup : bool, default = True
        If True, then show supplementary variables or variables categories points and/or texts.

    col_quali_sup : str, default = "blue"
        Color for supplementary variables or variable categories points and/or texts.

    point_args_quali_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary variables or variable categories points.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary variables or variable categories texts.

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
    :class:`~scientisttools.fviz_mca`
        Visualize Multiple Correspondence Analysis
    :class:`~scientisttools.get_mca`
        Extract the results for individuals/variables - MCA

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, fviz_mca_biplot
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison.data)
    >>> # biplot of individuals and categories
    >>> p = fviz_mca_biplot(clf,repel_ind=True,repel_var=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class MCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MCA":
        raise TypeError("'obj' must be a MCA class")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj = obj,
        choice = "ind",
        axis = axis,
        geom = geom_ind,
        repel = repel_ind,
        color = col_ind,
        point_args = point_args_ind,
        text_args = text_args_ind,
        gradient_cols = gradient_cols,
        legend_title = legend_title,
        habillage = habillage,
        palette = palette,
        add_ellipses = add_ellipses, 
        ellipse_type = ellipse_type,
        level = level,
        alpha = alpha
    )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ind_sup and hasattr(obj,"ind_sup_"):
        p = add_scatter(
            p=p,
            data=obj.ind_sup_.coord,
            axis=axis,
            geom=geom_ind,
            repel=repel_ind,
            color=col_ind_sup,
            point_args=point_args_ind_sup,
            text_args=text_args_ind_sup
        )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active categories points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(
        p=p,
        data=obj.levels_.coord,
        axis=axis,
        geom=geom_var,
        repel=repel_var,
        color=col_var,
        point_args=point_args_var,
        text_args=text_args_var
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary categories points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quali_sup and hasattr(obj,"levels_sup_"):
        p = add_scatter(
            p=p,
            data=obj.levels_sup_.coord,
            axis=axis,
            geom=geom_var,
            repel=repel_var,
            color=col_quali_sup,
            point_args=point_args_quali_sup,
            text_args=text_args_quali_sup
        )
  
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "MCA - Biplot of individuals and variable categories"

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
    
def fviz_mca(obj, 
             choice = "biplot",
             **kwargs):
    """
    Visualize Multiple Correspondence Analysis
    
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. 
    :class:`~scientisttols.fviz_mca` provides plotnine-based elegant visualization of :class:`~scientisttools.MCA` outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    choice : {"ind","levels","var","biplot","quanti_sup"}, default = "biplot"
        The graph to plot. Allowed values include:

        * 'ind' for the individuals graphs
        * 'levels' for the variables categories graphs
        * 'var' for the variables graphs 
        * 'biplot' for biplot of individuals and variable categories
        * 'quanti_sup' for the supplementary continuous variables (=correlation circle)

    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_mca_ind`: Graph of individuals
        * :class:`scientisttools.fviz_mca_var`: Graph of variables and variable categories
        * :class:`scientisttools.fviz_mca_biplot`: Biplot of individuals and variable categories
        * :class:`scientisttools.fviz_corcircle`: Graph of variables (=correlation circle)
    
    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_mca_ind`
        Visualize Multiple Correspondence Analysis - Graph of individuals
    :class:`~scientisttools.fviz_mca_var`
        Visualize Multiple Correspondence Analysis - Graph of variables
    :class:`~scientisttools.fviz_mca_biplot`
        Visualize Multiple Correspondence Analysis - Biplot of individuals and variables
    :class:`~scientisttools.get_mca`
        Extract the results for individuals/variables - MCA

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, fviz_mca_biplot
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison.data)
    >>> # graph of individuals
    >>> p = fviz_mca(clf,choice="ind",repel=True)
    >>> print(p.show())
    >>> # graph of variables categories
    >>> p = fviz_mca(clf,choice="levels",repel=True)
    >>> print(p.show())
    >>> # graph of variables correlations (=eta squared)
    >>> p = fviz_mca(clf,choice="var",repel=True)
    >>> print(p.show())
    >>> # biplot of individuals and variable categories
    >>> p = fviz_mca(clf,choice="biplot",repel=True)
    >>> print(p.show())
    >>> # graph of supplementary continuous variables (=correlation circle)
    >>> p = fviz_mca(clf,choice="quanti_sup",repel=True)
    >>> print(p.show())
    """    
    if choice == "ind":
        return fviz_mca_ind(obj,**kwargs)
    elif choice in ("levels","var"):
        return fviz_mca_var(obj,choice=choice,**kwargs)
    elif choice == "biplot":
        return fviz_mca_biplot(obj,**kwargs)
    elif choice == "quanti_sup":
        return fviz_corcircle(obj,**kwargs)
    else:
        raise ValueError("choice should be one of 'ind', 'levels', 'var', 'biplot', 'quanti_sup'")