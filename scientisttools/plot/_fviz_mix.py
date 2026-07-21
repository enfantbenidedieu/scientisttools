# -*- coding: utf-8 -*-
from plotnine import ggplot, theme_minimal

#intern functions
from ._fviz import (
    add_arrow,
    add_scatter,
    fviz_arrow,
    fviz_scatter, 
    set_axis
)

def fviz_mix_ind(obj,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 col_ind ="black",
                 point_args = dict(size=1.5),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB","#E7B800","#FC4E07"),
                 legend_title = None,
                 habillage = None,
                 palette = "Dark2",
                 add_ellipses = False,
                 ellipse_type = "confidence",
                 level = 0.95,
                 alpha = 0.1,
                 ind_sup = True,
                 col_ind_sup = "blue",
                 point_args_ind_sup = dict(size = 1.5),
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
    Visualize Mixed Data - Graph of individuals
    
    Factor Analysis of Mixed Data (:class:`~scientisttools.FAMD`), Principal Component Analysis of Mixed Data (:class:`~scientisttools.PCAmix`) and Mixed Principal Component Analysis (:class:`~scientisttools.MPCA`) are, a particular case of :class:`~scientisttools.PCA`, used to analyze a data set containing both continuous and categorical variables.
    :class:`~scientisttools.fviz_mix_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix` and :class:`~scientisttools.MPCA` outputs for individuals.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix` or :class:`~scientisttools.MPCA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

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
    :class:`~scientisttools.fviz_mix`
        Visualize Mixed Data
    :class:`~scientisttools.get_mix`
        Extract the results for individuals and variables - FAMD/PCAmix/MPCA

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, fviz_mix_ind
    >>> clf = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> clf.fit(autos2005)
    >>> # graph of individuals
    >>> p = fviz_mix_ind(clf)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid object class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("FAMD","PCAmix","MPCA")):
        raise TypeError("obj must be an object of class FAMD, PCAmix or MPCA")
    
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
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = f"{obj.__class__.__name__} - Graph of individuals"

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
        subtitle = subtitle,
        pntheme=pntheme,
        **kwargs
    )
    return p
    
def fviz_mix_var(obj,
                 choice = "var",
                 axis = [0,1],
                 geom = ("arrow","point","text"),
                 repel = False,
                 col_var ="black",
                 segment_args = dict(size=0.5,alpha=1),
                 point_args = dict(size=1.5),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = "Dark2",
                 add_ellipses = False,
                 ellipse_type = "confidence",
                 level = 0.95,
                 alpha = 0.1,
                 quanti_sup = True,
                 col_quanti_sup = "red",
                 segment_args_quanti_sup = dict(linetype="dashed",size=0.5,alpha=1),
                 point_args_quanti_sup = dict(size=1.5),
                 text_args_quanti_sup = dict(size=8),
                 quali_sup = True,
                 col_quali_sup = "blue",
                 point_args_quali_sup = dict( size=1.5),
                 text_args_quali_sup = dict(size=8),
                 lim_cos2 = None,
                 lim_contrib = None,
                 scale = 1,
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
    Visualize Mixed Data - Graph of variables
    
    Factor Analysis of Mixed Data (:class:`~scientisttools.FAMD`), Principal Component Analysis of Mixed Data (:class:`~scientisttools.PCAmix`) and Mixed Principal Component Analysis (:class:`~scientisttools.MPCA`) are, a particular case of :class:`~scientisttools.PCA`, used to analyze a data set containing both continuous and categorical variables.
    :class:`~scientisttools.fviz_mix_var` provides plotnine-based elegant visualization of :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix` and :class:`~scientisttools.MPCA` outputs for variables.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix` or :class:`~scientisttools.MPCA`.

    choice : {"levels","quanti_var","var"}, default = "var"
        The graph to plot. Allowed values include:

        * "levels" for variable categories
        * "quanti_var" for continuous variables (=correlation circle)
        * "var" for variables
    
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
        A dictionary containing parameters  (except color and arrow) for segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

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

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables points and/or texts.

    col_quanti_sup : str, default = "red"
        Color for supplementary continuous variables points, arrows and/or texts.

    segment_args_quanti_sup : dict, default = dict(linetype="dashed",size = 0.5)
        A dictionary containing parameters (except color and arrow) for supplementary continuous variables segments.

    point_args_quanti_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary continuous variables points.

    text_args_quanti_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary continuous variables texts.

    quali_sup : bool, default = True
        If True, then show supplementary categorical variables or variables categories points and/or texts.

    col_quali_sup : str, default = "blue"
        Color for supplementary categorical variables or variables categories points and/or texts.

    point_args_quali_sup : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for supplementary categorical variables or variables categories points.

    text_args_quali_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary categorical variables or variables categories texts.

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
    :class:`~scientisttools.fviz_mix`
        Visualize Mixed Data
    :class:`~scientisttools.get_mix`
        Extract the results for individuals and variables - FAMD/PCAmix/MPCA

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, fviz_mix_var
    >>> clf = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> clf.fit(autos2005)
    >>> # graph of continuous variables
    >>> p = fviz_mix_var(clf, choice = "quanti_var")
    >>> print(p.show())
    >>> # graph of variable categories
    >>> p = fviz_mix_var(clf, choice = "levels")
    >>> print(p.show())
    >>> # graph of variables
    >>> p = fviz_mix_var(clf, choice = "var")
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid object class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("FAMD","PCAmix","MPCA")):
        raise TypeError("'obj' must be an object of class FAMD, PCAmix or MPCA")

    if (choice == "quanti_var" and 
        hasattr(obj,"quanti_var_")):
        # show active variables segments
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
        
        # show supplementary continuous variables segments and/or texts
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
                text_args = text_args_quanti_sup
            )
    elif choice == "levels" and hasattr(obj,"levels_"):
        # show variable categories points
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
            ellipse_type=ellipse_type,
            level = level,
            alpha = alpha,
            lim_cos2 = lim_cos2,
            lim_contrib = lim_contrib
        )
            
        # show supplementary variable categories points
        if quali_sup and hasattr(obj,"levels_sup_"):
            p = add_scatter(
                p=p,
                data = obj.levels_sup_.coord,
                axis = axis,
                geom = geom,
                repel = repel,
                color = col_quali_sup,
                point_args = point_args_quali_sup,
                text_args = text_args_quali_sup
            )
    elif (choice == "var" and hasattr(obj,"var_")):
        # show variables points
        p = add_scatter(
            p = ggplot(),
            data = obj.var_.coord,
            axis = axis,
            geom = geom,
            repel = repel,
            color = col_var,
            point_args = point_args,
            text_args = text_args
        )
        
        # show supplementary categorical variables points
        if quali_sup and hasattr(obj,"quali_var_sup_"):
            p = add_scatter(
                p = p,
                data = obj.quali_var_sup_.coord,
                axis = axis,
                geom = geom,
                repel = repel,
                color = col_quali_sup,
                point_args = point_args_quali_sup,
                text_args = text_args_quali_sup
            )

        # show supplementary continuous variables points
        if quanti_sup and hasattr(obj,"quanti_var_sup_"):
            p = add_scatter(
                p = p,
                data = obj.quanti_var_sup_.cos2,
                axis = axis,
                geom = geom,
                repel = repel,
                color = col_quanti_sup,
                point_args = point_args_quanti_sup,
                text_args = text_args_quanti_sup
            )
        
    # set x and y limits
    if choice == "quanti_var":
        if x_lim is None:
            x_lim = (-1.1,1.1)
        if y_lim is None:
            y_lim = (-1.1,1.1)
        
    # set title    
    if title is None:
        if choice == "levels":
            title = f"{obj.__class__.__name__} - Graph of variable categories"
        elif choice == "quanti_var":
            title = f"{obj.__class__.__name__} - Graph of continuous variables"
        elif choice == "var":
            title = f"{obj.__class__.__name__} - Graph of variables"

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
    
def fviz_mix(obj, 
             choice="ind", 
             **kwargs):
    """
    Visualize Mixed Data

    Factor Analysis of Mixed Data (:class:`~scientisttools.FAMD`), Principal Component Analysis of Mixed Data (:class:`~scientisttools.PCAmix`) and Mixed Principal Component Analysis (:class:`~scientisttools.MPCA`) are, a particular case of :class:`~scientisttools.PCA`, used to analyze a data set containing both continuous and categorical variables.
    :class:`~scientisttools.fviz_mix` provides plotnine-based elegant visualization of :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix` and :class:`~scientisttools.MPCA` outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix` or :class:`~scientisttools.MPCA`.

    element : {"ind","levels","quanti_var","var"}, default = "ind"
        The graph to plot. Allowed values include one of:

        * "ind" for graph of individuals
        * "levels" for the graph of variable categories
        * "quanti_var" for graph of continuous variables
        * "var" for graph of variables

    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_mix_ind`: Graph of individuals
        * :class:`scientisttools.fviz_mix_var`: Graph of variables
        
    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_mix_ind`
        Visualize Mixed Data - Graph of individuals
    :class:`~scientisttools.fviz_mix_var`
        Visualize Mixed Data - Graph of variables
    :class:`~scientisttools.get_mix`
        Extract the results for individuals and variables - FAMD/PCAmix/MPCA

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, fviz_mix
    >>> clf = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> clf.fit(autos2005)
    >>> # graph of individuals
    >>> p = fviz_mix(clf,repel=True)
    >>> print(p.show())
    >>> # graph of variable categories
    >>> p = fviz_mix(clf,choice="levels",repel=True)
    >>> print(p.show())
    >>> # graph of continuous variables (=correlation circle)
    >>> p = fviz_mix(clf,choice="quanti_var",repel=True)
    >>> print(p.show())
    >>> # graph of variables
    >>> p = fviz_mix(clf,choice="var",repel=True)
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_mix_ind(obj,**kwargs)
    elif choice in ("levels","quanti_var","var"):
        return fviz_mix_var(obj,choice=choice,**kwargs)
    else:
        raise ValueError("choice should be one of 'ind', 'levels', 'quanti_var', 'var'.")