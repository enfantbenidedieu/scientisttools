# -*- coding: utf-8 -*-
from plotnine import theme_minimal

#intern functions
from ._fviz import (
    add_arrow,
    add_scatter,
    fviz_arrow, 
    fviz_scatter, 
    set_axis
)

def fviz_pca_ind(obj,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 col_ind = "black",
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
    Visualize Principal Component Analysis - Graph of individuals
    
    Principal components analysis (:class:`~scientisttools.PCA`) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. 
    :class:`~scientisttools.fviz_pca_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.PCA` outputs for individuals.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`.

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

    quali_sup : bool, default = True
        If True, then show supplementary variable categories points and/or texts.

    col_quali_sup : str, default = "red"
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

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_pca`
        Visualize Principal Component Analysis
    :class:`~scientisttools.get_pca`
        Extract the results for individuals/variables - PCA

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_pca_ind
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> # graph of individuals
    >>> p = fviz_pca_ind(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class PCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCA":
        raise TypeError("'obj' must be an object of class PCA")

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
    # show supplementary individuals points
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
        title = "PCA - Graph of individuals"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show others elements
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

def fviz_pca_var(obj,
                 axis = [0,1],
                 geom = ("arrow","text"),
                 repel = False,
                 col_var = "black",
                 segment_args = dict(size=0.5),
                 point_args = dict(size=1.5),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = "Dark2",
                 quanti_sup = True,
                 col_quanti_sup = "blue",
                 segment_args_quanti_sup = dict(linetype="dashed",size=0.5),
                 point_args_quanti_sup = dict(size=1.5),
                 text_args_quanti_sup = dict(size=8),
                 scale = 1,
                 lim_cos2 = None,
                 lim_contrib = None,
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
    Visualize Principal Component Analysis - Graph of variables
    
    Principal components analysis (:class:`~scientisttools.PCA`) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. 
    :class:`~scientisttools.fviz_pca_var` provides plotnine-based elegant visualization of :class:`~scientisttools.PCA` outputs for variables.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text"). 

        * "arrow" to show only arrows.
        * "point" to show only points
        * "text" to show only labels.
        * ("arrow","text") to show both arrows and labels.
        * ("point","text") to show both points and labels.
    
    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    col_var : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for variables. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for variables are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 

    segment_args : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color) for variables segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for variables points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for variables texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables segments, points and/or texts.

    col_quanti_sup : str, default = "blue"
        Color for supplementary continuous variables segments, points and/or texts.

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

    See also
    --------
    :class:`~scientisttools.fviz_pca`
        Visualize Principal Component Analysis
    :class:`~scientisttools.get_pca`
        Extract the results for individuals/variables - PCA

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_pca_var
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> # graph of variables
    >>> p = fviz_pca_var(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class PCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCA":
        raise TypeError("'obj' must be an object of class PCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_arrow(
        obj = obj,
        choice = "quanti_var",
        axis = axis,
        geom = geom,
        repel = repel,
        color = col_var,
        segment_args = segment_args,
        point_args = point_args,
        text_args = text_args,
        gradient_cols = gradient_cols,
        legend_title = legend_title,
        palette = palette,
        scale = scale,
        lim_cos2 = lim_cos2,
        lim_contrib = lim_contrib,
        circle = circle,
        col_circle = col_circle
    )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary continuous variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quanti_sup and hasattr(obj,"quanti_var_sup_"):
        p = add_arrow(
            p = p,
            data = obj.quanti_var_sup_.coord.mul(scale),
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_quanti_sup,
            segment_args = segment_args_quanti_sup,
            point_args = point_args_quanti_sup,
            text_args=text_args_quanti_sup
        )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "PCA - Graph of variables"

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
        title=title,
        subtitle=subtitle,
        pntheme=pntheme,
        **kwargs
    )
    return p
    
def fviz_pca_biplot(obj,
                    axis = [0,1],
                    geom_ind = ("point","text"),
                    geom_var = ("arrow","text"),
                    repel_ind = False,
                    repel_var = False,
                    col_ind = "black",
                    point_args_ind = dict(size=1.5),
                    text_args_ind = dict(size=8),
                    col_var = "steelblue",
                    segment_args_var = dict(size=0.5),
                    point_args_var = dict(size=1.5),
                    text_args_var = dict(size=8),
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
                    quali_sup = True,
                    col_quali_sup = "violet",
                    point_args_quali_sup = dict(size=1.5),
                    text_args_quali_sup = dict(size=8),
                    quanti_sup = True,
                    col_quanti_sup = "darkblue",
                    segment_args_quanti_sup = dict(linetype="dashed",size=0.5),
                    point_args_quanti_sup = dict(size=1.5),
                    text_args_quanti_sup = dict(size=8),
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = None,
                    subtitle = None,
                    pntheme = theme_minimal(),
                    **kwargs):
    """
    Visualize Principal Component Analysis - Biplot of individuals and variables
    
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. 
    :class:`~scientisttools.fviz_pca_biplot` provides plotnine-based elegant visualization of PCA outputs for individuals and variables.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom_ind : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.

    geom_var : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text"). 

        * "arrow" to show only arrows.
        * "point" to show only points
        * "text" to show only labels.
        * ("arrow","text") to show both arrows and labels.
        * ("point","text") to show both points and labels.
    
    repel_ind : bool, default = False
        Whether to avoid overplotting individuals text labels or not.

    repel_var : bool, default = False
        Whether to avoid overplotting variables text labels or not.

    col_ind : str, 1darray, km class, list, tuple, Series, default = "black"
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
        Color for variables.

    segment_args_var : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color) for variables segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    point_args_var : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for variables points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args_var : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for variables texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

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

    point_args_ind_sup : dict, default = dict(ize = 1.5)
        A dictionary containing parameters (except color) for supplementary individuals points.

    text_args_ind_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary individuals texts.
    
    quali_sup : bool, default = True
        If True, then show supplementary variable categories points and/or texts.

    col_quali_sup : str, default = "red"
        Color for supplementary variable categories points and/or texts.

    point_args_quali_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary variable categories points.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary variable categories texts.

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables segments, points and/or texts.

    col_quanti_sup : str, default = "blue"
        Color for supplementary continuous variables segments, points and/or texts.

    segment_args_quanti_sup : dict, default = dict(linetype="dashed",size=0.5,alpha=1)
        A dictionary containing parameters (except color) for supplementary continuous variables segments.

    point_args_quanti_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary continuous variables points.

    text_args_quanti_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary continuous variables texts.

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
    :class:`~scientisttools.fviz_pca`
        Visualize Principal Component Analysis
    :class:`~scientisttools.get_pca`
        Extract the results for individuals/variables - PCA

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_pca_biplot
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> # biplot of individuals and variables
    >>> p = fviz_pca_biplot(clf,repel_ind=True,repel_var=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a PCA class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCA":
        raise TypeError("'obj' must be an object of class PCA")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj=obj,
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
    # show supplementary individuals
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
    # show supplementary categories
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quali_sup and hasattr(obj,"levels_sup_"):
        p = add_scatter(
            p=p,
            data=obj.levels_sup_.coord,
            axis=axis,
            geom=geom_ind,
            repel=repel_ind,
            color=col_quali_sup,
            point_args=point_args_quali_sup,
            text_args=text_args_quali_sup
        )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # rescale variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    xscale = (max(obj.ind_.coord.iloc[:,axis[0]])-min(obj.ind_.coord.iloc[:,axis[0]]))/(max(obj.quanti_var_.coord.iloc[:,axis[0]])-min(obj.quanti_var_.coord.iloc[:,axis[0]]))
    yscale = (max(obj.ind_.coord.iloc[:,axis[1]])-min(obj.ind_.coord.iloc[:,axis[1]]))/(max(obj.quanti_var_.coord.iloc[:,axis[1]])-min(obj.quanti_var_.coord.iloc[:,axis[1]]))
    scale = min(xscale, yscale)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show continuous variables segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_arrow(
        p = p,
        data = obj.quanti_var_.coord.mul(scale),
        axis = axis,
        geom = geom_var,
        repel = repel_var,
        color = col_var,
        segment_args = segment_args_var,
        point_args = point_args_var,
        text_args = text_args_var
    )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary continuous variables segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quanti_sup and hasattr(obj,"quanti_var_sup_"):
        p = add_arrow(
            p = p,
            data = obj.quanti_var_sup_.coord.mul(scale),
            axis = axis,
            geom = geom_var,
            repel = repel_var,
            color = col_quanti_sup,
            segment_args = segment_args_quanti_sup,
            point_args = point_args_quanti_sup,
            text_args = text_args_quanti_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "PCA - Biplot of individuals and variables"

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
    
def fviz_pca(obj,
             choice="biplot",
             **kwargs):
    """
    Visualize Principal Component Analysis
    
    Principal components analysis (:class:`~scientisttools.PCA`) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. 
    :class:`~scientisttools.fviz_pca_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.PCA` outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`.

    choice : {"ind","var","biplot"}, default = "biplot"
        The graph to plot. Allowed values include one of ("ind","var","biplot"):

        * "ind" for the individuals graphs
        * "var" for the variables graphs (= Correlation circle)
        * "biplot" for biplot of individuals and variables

    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_pca_ind`: Graph of individuals
        * :class:`scientisttools.fviz_pca_var`: Graph of variables
        * :class:`scientisttools.fviz_pca_biplot`: Biplot of individuals and variables
    
    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_pca_ind`
        Visualize Principal Component Analysis - Graph of individuals
    :class:`~scientisttools.fviz_pca_var`
        Visualize Principal Component Analysis - Graph of variables
    :class:`~scientisttools.fviz_pca_biplot`
        Visualize Principal Component Analysis - Biplot of individuals and variables
    :class:`~scientisttools.get_pca`
        Extract the results for individuals/variables - PCA
    
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_pca
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> # graph of individuals
    >>> p = fviz_pca(clf,choice="ind",repel=True)
    >>> print(p.show())
    >>> # graph of variables
    >>> p = fviz_pca(clf,choice="var",repel=True)
    >>> print(p.show())
    >>> # biplot of individuals and variables
    >>> p = fviz_pca(clf,repel_ind=True,repel_var=True)
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_pca_ind(obj,**kwargs)
    elif choice == "var":
        return fviz_pca_var(obj,**kwargs)
    elif choice == "biplot":
        return fviz_pca_biplot(obj,**kwargs)
    else:
        raise ValueError("choice should be one of 'ind', 'var', 'biplot'")