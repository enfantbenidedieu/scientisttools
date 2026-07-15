# -*- coding: utf-8 -*-
from plotnine import theme_minimal

#intern functions
from ._fviz import fviz_scatter, add_scatter, set_axis
from ._fviz_corcircle import fviz_corcircle

def fviz_ca_row(obj,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                col_row = "black",
                point_args = dict(size=1.5),
                text_args = dict(size=8),
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                habillage = None,
                palette = "Dark2",
                ellipse = False, 
                ellipse_type = "confidence",
                level = 0.95,
                alpha = 0.1,
                row_sup = True,
                col_row_sup = "blue",
                point_args_row_sup = dict(shape="^",size=1.5),
                text_args_row_sup = dict(size=8),
                quali_sup = True,
                col_quali_sup = "red",
                point_args_quali_sup = dict(shape=">",size=1.5),
                text_args_quali_sup = dict(size=8),
                lim_cos2 = None,
                lim_contrib = None,
                x_lim = None,
                y_lim = None,
                x_label = None,
                y_label = None,
                title = None,
                subtitle = None,
                hline = True,
                vline = True,
                pntheme = theme_minimal(),
                **kwargs):
    """
    Visualize Correspondence Analysis - Graph of row variables
    
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. 
    :class:`~scientisttools.fviz_ca_row` provides plotnine-based elegant visualization of :class:`~scientisttools.CA` outputs for rows.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    col_row : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for row variables. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for row variables are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 
        To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`) except color.

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high values.

    legend_title : str, defaut = None
        The title of the legend. If None, then a legend title is chosen.

    habillage : str, int, default = None 
        The name of variable for coloring the observations by groups.

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    ellipse : bool, default = False
        If True, draws ellipses around the points when habillage is not None.

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

    row_sup : bool, default = True
        If True, then show supplementary row variables points and/or texts.

    col_row_sup : str, default = "blue"
        Color for supplementary row variables points and/or texts.

    point_args_row_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters for supplementary row variables points except color.

    text_args_row_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary row variables texts.

    quali_sup : bool, default = True
        If True, then show supplementary variables categories points and/or texts.

    col_quali_sup : str, default = "red"
        Color for supplementary variables categories points and/or texts.

    point_args_quali_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters for supplementary variables categories points except color.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary variables categories texts.

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
    :class:`~scientisttools.fviz_ca`
        Visualize Correspondence Analysis
    :class:`~scientisttools.get_ca`
        Extract the results for rows/columns - CA

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, fviz_ca_row
    >>> clf = CA(ncp=2,row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> # graph of row variables
    >>> p = fviz_ca_row(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a CA class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CA":
        raise TypeError("'obj' must be a CA class")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active row points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj = obj,
        choice = "row",
        axis = axis,
        geom = geom,
        repel = repel,
        lim_cos2 = lim_cos2,
        lim_contrib = lim_contrib,
        color = col_row,
        point_args = point_args,
        text_args = text_args,
        gradient_cols = gradient_cols,
        legend_title = legend_title,
        habillage = habillage,
        palette = palette,
        ellipse = ellipse, 
        ellipse_type = ellipse_type,
        level = level,
        alpha = alpha
    )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary rows points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if row_sup and hasattr(obj,"row_sup_"):
        p = add_scatter(
            p = p,
            data = obj.row_sup_.coord,
            axis = axis,
            geom = geom,
            repel= repel,
            color = col_row_sup,
            point_args = point_args_row_sup,
            text_args = text_args_row_sup
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
        title = "CA - Graph of row variables"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show other points
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
        hline = hline,
        vline = vline,
        pntheme = pntheme,
        **kwargs
    )
    return p

def fviz_ca_col(obj,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                col_col = "black",
                point_args = dict(size=1.5),
                text_args = dict(size=8),
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                palette = "Dark2",
                col_sup = True,
                col_col_sup = "blue",
                point_args_col_sup = dict(shape="^",size=1.5),
                text_args_col_sup = dict(size=8),
                lim_cos2 = None,
                lim_contrib = None,
                x_lim = None,
                y_lim = None,
                x_label = None,
                y_label = None,
                title = None,
                subtitle = None,
                hline = True,
                vline = True,
                pntheme = theme_minimal(),
                **kwargs):
    """
    Visualize Correspondence Analysis - Graph of column variables
   
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. 
    :class:`~scientisttools.fviz_ca_col` provides plotnine-based elegant visualization of :class:`~scientisttools.CA` outputs for columns.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    col_col : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for column variables. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for column variables are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y").

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`) except color.

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high values.

    legend_title : str, defaut = None
        The title of the legend. If None, then a legend title is chosen.

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    ellipse : bool, default = False
        If True, draws ellipses around the points when habillage is not None.

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

    col_sup : bool, default = True
        If True, then show supplementary column variables points and/or texts.

    col_col_sup : str, default = "blue"
        Color for supplementary column variables points and/or texts.

    point_args_col_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters for supplementary column variables points except color.

    text_args_col_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary column variables texts.

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
    :class:`~scientisttools.fviz_ca`
        Visualize Correspondence Analysis
    :class:`~scientisttools.get_ca`
        Extract the results for rows/columns - CA

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, fviz_ca_col
    >>> clf = CA(ncp=2,row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> # graph of column variables
    >>> p = fviz_ca_col(clf,repel=True)
    >>> print(p)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class CA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CA":
        raise TypeError("'obj' must be an object of class CA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active columns points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj = obj,
        choice = "col",
        axis = axis,
        geom = geom,
        repel = repel,
        lim_cos2 = lim_cos2,
        lim_contrib = lim_contrib,
        color = col_col,
        point_args = point_args,
        text_args = text_args,
        gradient_cols = gradient_cols,
        legend_title = legend_title,
        palette = palette
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary columns points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if col_sup and hasattr(obj,"col_sup_"):
        p = add_scatter(
            p = p,
            data = obj.col_sup_.coord,
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_col_sup,
            point_args = point_args_col_sup,
            text_args = text_args_col_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "CA - Graph of column variables"

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
        hline = hline,
        vline = vline,
        pntheme = pntheme,
        **kwargs
    )
    return p

def fviz_ca_biplot(obj,
                   axis = [0,1],
                   geom_row = ("point","text"),
                   geom_col = ("point","text"),
                   repel_row = True,
                   repel_col = True,
                   col_row = "black",
                   point_args_row = dict(size=1.5),
                   text_args_row = dict(size=8),
                   col_col = "steelblue",
                   point_args_col = dict(size=1.5),
                   text_args_col = dict(size=8),
                   gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                   legend_title = None,
                   habillage = None,
                   palette = "Dark2",
                   ellipse = False, 
                   ellipse_type = "convex", 
                   level = 0.95,
                   alpha = 0.1,
                   row_sup = True,
                   col_row_sup = "red",
                   point_args_row_sup = dict(shape="^",size=1.5),
                   text_args_row_sup = dict(size=8),
                   quali_sup = True,
                   col_quali_sup = "darkred",
                   point_args_quali_sup = dict(shape="v",size=1.5),
                   text_args_quali_sup = dict(size=8),
                   col_sup = True,
                   col_col_sup = "darkblue",
                   point_args_col_sup = dict(shape="x",size=1.5),
                   text_args_col_sup = dict(size=8),
                   x_lim = None,
                   y_lim = None,
                   x_label = None,
                   y_label = None,
                   title = None,
                   subtitle = None,
                   hline = True,
                   vline = True,
                   pntheme = theme_minimal(),
                   **kwargs):
    """
    Visualize Correspondence Analysis - Biplot of row and column variables

    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. 
    :class:`~scientisttools.fviz_ca_biplot` provides plotnine-based elegant visualization of :class:`~scientisttools.CA` outputs.
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom_row : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.

    geom_col : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel_row : bool, default = True
        Whether to avoid overplotting row variables text labels or not.

    repel_col : bool, default = True
        Whether to avoid overplotting column variables text labels or not.

    col_row : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for row variables. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for row variables are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 
        To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    point_args_row : dict, default = dict(size = 1.5)
        A dictionary containing parameters for row variables points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`) except color.

    text_args_row : dict, default = dict(size = 8)
        A dictionary containing parameters for row variables texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    col_col : str, default = "steelblue"
        Color for column variables

    point_args_col : dict, default = dict(size = 1.5)
        A dictionary containing parameters for column variables points except color.

    text_args_col : dict, default = dict(size = 8)
        A dictionary containing parameters for column variables texts.

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high values.

    legend_title : str, defaut = None
        The title of the legend. If None, then a legend title is chosen.

    habillage : str, int, default = None 
        The name of variable for coloring the observations by groups.

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    ellipse : bool, default = False
        If True, draws ellipses around the points when habillage is not None.

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

    row_sup : bool, default = True
        If True, then show supplementary row variables points and/or texts.

    col_row_sup : str, default = "red"
        Color for supplementary row variables points and/or texts.

    point_args_row_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters for supplementary row variables points except color.

    text_args_row_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary row variables texts.

    quali_sup : bool, default = True
        If True, then show supplementary variables categories points and/or texts.

    col_quali_sup : str, default = "darkred"
        Color for supplementary variables categories points and/or texts.

    point_args_quali_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters for supplementary variables categories points except color.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary variables categories texts.

    col_sup : bool, default = True
        If True, then show supplementary column variables points and/or texts.

    col_col_sup : str, default = "red"
        Color for supplementary column variables points and/or texts.

    point_args_col_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters for supplementary column variables points except color.

    text_args_col_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary column variables texts.

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
    :class:`~scientisttools.fviz_ca`
        Visualize Correspondence Analysis
    :class:`~scientisttools.get_ca`
        Extract the results for rows/columns - CA

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, fviz_ca_biplot
    >>> clf = CA(ncp=2,row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> # biplot of row and column variables
    >>> p = fviz_ca_biplot(clf,repel_row=True,repel_col=True)
    >>> print(p)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class CA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "CA":
        raise ValueError("'obj' must be an object of class CA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active rows points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_scatter(
        obj = obj,
        choice = "row",
        axis = axis,
        geom = geom_row,
        repel = repel_row,
        color = col_row,
        point_args = point_args_row,
        text_args = text_args_row,
        gradient_cols = gradient_cols,
        legend_title = legend_title,
        habillage = habillage,
        palette = palette,
        ellipse = ellipse, 
        ellipse_type = ellipse_type,
        level = level,
        alpha = alpha
    )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active columns points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(
        p = p,
        data = obj.col_.coord,
        axis = axis,
        geom = geom_col,
        repel = repel_col,
        color = col_col,
        point_args = point_args_col,
        text_args = text_args_col
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary rows points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if row_sup and hasattr(obj,"row_sup_"):
        p = add_scatter(
            p = p,
            data = obj.row_sup_.coord,
            axis = axis,
            geom = geom_row,
            repel = repel_row,
            color = col_row_sup,
            point_args = point_args_row_sup,
            text_args = text_args_row_sup
        )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary categories
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quali_sup and hasattr(obj,"quali_sup_"):
        p = add_scatter(
            p = p,
            data = obj.quali_sup_.coord,
            axis = axis,
            geom = geom_row,
            repel = repel_row,
            color = col_quali_sup,
            point_args = point_args_quali_sup,
            text_args = text_args_quali_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary columns points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if col_sup and hasattr(obj,"col_sup_"):
        p = add_scatter(
            p = p,
            data = obj.col_sup_.coord,
            axis = axis,
            geom = geom_col,
            repel = repel_col,
            color = col_col_sup,
            point_args = point_args_col_sup,
            text_args = text_args_col_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "CA - Biplot of rows and column variables"

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
        hline = hline,
        vline = vline,
        pntheme = pntheme,
        **kwargs
    )
    return p
    
def fviz_ca(obj, 
            choice = "biplot",
            **kwargs):
    """
    Visualize Correspondence Analysis

    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables.
    :class:`~scientisttools.fviz_ca` provides plotnine-based elegant visualization of :class:`scientisttools.CA` outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    choice : {"row","col","biplot"}, default = "biplot"
        The graph to plot. Allowed values are:

        * 'row' for graph of row variables
        * 'col' for graph of column variables
        * 'biplot' for biplot of row and columns variables
        * 'quanti_sup' for graph of variables (=correlation circle)

    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_ca_row`: Graph of row variables 
        * :class:`scientisttools.fviz_ca_col`: Graph of column variables
        * :class:`scientisttools.fviz_ca_biplot`: Biplot of row and column variables
        * :class:`scientisttools.fviz_corcircle`: Graph of variables (=correlation circle)

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_ca_biplot`
        Visualize Correspondence Analysis - Biplot of row and column variables
    :class:`~scientisttools.fviz_ca_col`
        Visualize Correspondence Analysis - Graph of column variables
    :class:`~scientisttools.fviz_ca_row`
        Visualize Correspondence Analysis - Graph of row variables
    :class:`~scientisttools.get_ca`
        Extract the results for rows/columns - CA

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, fviz_ca
    >>> clf = CA(ncp=2,row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> # graph of row variables
    >>> p1 = fviz_ca(clf, choice = "row")
    >>> print(p1.show())
    >>> # graph of colum variables
    >>> p2 = fviz_ca(clf, choice = "col")
    >>> print(p2.show())
    >>> # biplot of row and colum variables
    >>> p3 = fviz_ca(clf, choice = "biplot")
    >>> print(p3.show())
    """
    if choice == "row":
        return fviz_ca_row(obj,**kwargs)
    elif choice == "col":
        return fviz_ca_col(obj,**kwargs)
    elif choice == "biplot":
        return fviz_ca_biplot(obj,**kwargs)
    elif choice == "quanti_sup" and hasattr(obj,"quanti_var_sup_"):
        return fviz_corcircle(obj,**kwargs)
    else:
        raise ValueError("'choice' should be one of 'row', 'col', 'biplot'")