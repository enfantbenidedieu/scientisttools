# -*- coding: utf-8 -*-
from plotnine import ggplot, theme_minimal

# intern functions
from ._fviz import fviz_circle, add_scatter, add_arrow, set_axis

def fviz_fa_ind(obj,
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
                x_lim = None,
                y_lim = None,
                x_label = None,
                y_label = None,
                title = None,
                subtitle = None,
                pntheme = theme_minimal(),
                **kwargs):
    """
    Visualize Factor Analysis - Graph of individuals

    Factor analysis (:class:`~scientisttools.FA`) is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. 
    It is used to identify the structure of the relationship between the variable and the respondent. :class:`~scientisttools.fviz_fa_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.FA` outputs for individuals.
   
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`.

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
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    ind_sup : bool, default = True
        If True, then show supplementary individuals points and/or texts.

    col_ind_sup : str, default = "blue"
        Color for supplementary individuals points and/or texts.

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
    :class:`~scientisttools.fviz_fa`
        Visualize Factor Analysis.
    :class:`~scientisttools.get_fa`
        Extract the results for individuals/variables - FA
    
    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, fviz_fa_ind
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    >>> # graph of individuals
    >>> p = fviz_fa_ind(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a FA class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "FA":
        raise TypeError("'obj' must be an object of class FA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(
        p=ggplot(),
        data=obj.ind_.coord,
        axis=axis,
        geom=geom,
        repel=repel,
        color=col_ind,
        point_args=point_args,
        text_args=text_args
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
        title = "FA - Graph of individuals"

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

def fviz_fa_var(obj,
                axis = [0,1],
                geom = ("arrow","text"),
                repel = False,
                col_var ="black",
                segment_args = dict(size=0.5),
                text_args = dict(size=8),
                quanti_sup = True,
                col_quanti_sup = "blue",
                segment_args_quanti_sup = dict(linetype="dashed",size=0.5),
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
    Visualize Factor Analysis - Graph of variables

    Factor analysis is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. 
    It is used to identify the structure of the relationship between the variable and the respondent. :class:`~scientisttools.fviz_fa_var` provides plotnine-based elegant visualization of :class:`~scientisttools.FA` outputs for variables.
   
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","text"). 

        * "arrow" to plot only arrows.
        * "text" to show only labels.
        * ("arrow","text") to show both types.
    
    repel : bool, default = False
        Whether to avoid overplotting variables text labels or not.

    col_var : str, default = "black"
        Color for variables. 

    segment_args : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color and arrow ) for variables segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for variables texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables segments and/or texts.

    col_quanti_sup : str, default = "blue"
        Color for supplementary continuous variables segments and/or texts.

    segment_args_quanti_sup : dict, default = dict(linetype="dashed",size=0.5,alpha=1)
        A dictionary containing parameters for supplementary continuous variables segments except color.

    text_args_quanti_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary continuous variables texts.

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
    :class:`~scientisttools.fviz_fa`
        Visualize Factor Analysis.
    :class:`~scientisttools.get_fa`
        Extract the results for individuals/variables - FA

    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, fviz_fa_var
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    >>> # graph of variables
    >>> p = fviz_fa_var(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a FA class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "FA":
        raise TypeError("'obj' must be a FA class.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active continuous variables segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_arrow(
        p=ggplot(),
        data=obj.quanti_var_.coord.mul(scale),
        axis=axis,
        geom=geom,
        repel=repel,
        color=col_var,
        segment_args=segment_args,
        text_args=text_args
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add correlation circle
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if circle:
        p = fviz_circle(p=p,color=col_circle)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "FA - graph of variables"

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
    
# biplot of individuals and variables
def fviz_fa_biplot(obj,
                    axis = [0,1],
                    geom_ind = ("point","text"),
                    geom_var = ("arrow","text"),
                    repel_ind = False,
                    repel_var = True,
                    col_ind = "black",
                    point_args_ind = dict(size=1.5),
                    text_args_ind = dict(size=8),
                    col_var = "steelblue",
                    segment_args_var = dict(size=0.5),
                    text_args_var = dict(size = 8),
                    ind_sup = True,
                    col_ind_sup = "blue",
                    point_args_ind_sup = dict(size=1.5),
                    text_args_ind_sup = dict(size=8),
                    quanti_sup = True,
                    col_quanti_sup = "darkblue",
                    segment_args_quanti_sup = dict(linetype="dashed",size=0.5),
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
    Visualize Factor Analysis (FA) - Biplot of individuals and variables

    Factor analysis (:class:`~scientisttools.FA`) is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. 
    It is used to identify the structure of the relationship between the variable and the respondent. :class:`~scientisttools.fviz_fa_biplot` provides plotnine-based elegant visualization of :class:`~scientisttools.FA` outputs for individuals and variables.
   
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom_ind : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.

    geom_var : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","text"). 

        * "arrow" to show only segments.
        * "text" to show only labels.
        * ("arrow","text") to show both types.
    
    repel_ind : bool, default = True
        Whether to avoid overplotting individuals text labels or not.

    repel_var : bool, default = True
        Whether to avoid overplotting variables text labels or not.

    col_ind : str, default = "black"
        Color for individuals.

    point_args_ind : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for individuals points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args_ind : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for individuals texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    col_var : str, default = "steelblue"
        Color for variables.

    segment_args_var : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color) for variables segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    text_args_var : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for variables texts.

    ind_sup : bool, default = True
        If True, then show supplementary individuals points and/or texts.

    col_ind_sup : str, default = "blue"
        Color for supplementary individuals points and/or texts.

    point_args_ind_sup : dict, default = dict(shape="^",size = 1.5)
        A dictionary containing parameters (except color) for supplementary individuals points.

    text_args_ind_sup : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for supplementary individuals texts.

    quanti_sup : bool, default = True
        If True, then show supplementary continuous variables segments and/or texts.

    col_quanti_sup : str, default = "darkblue"
        Color for supplementary continuous variables segments and/or texts.

    segment_args_quanti_sup : dict, default = dict(linetype="dashed",size=0.5,alpha=1)
        A dictionary containing parameters for supplementary continuous variables segments except color.

    text_args_quanti_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary continuous variables texts.

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
    :class:`~scientisttools.fviz_fa`
        Visualize Factor Analysis.
    :class:`~scientisttools.get_fa`
        Extract the results for individuals/variables - FA
    
    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, fviz_fa_biplot
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    >>> # biplot - graph of individuals and variables
    >>> p = fviz_fa_biplot(clf,repel_ind=True,repel_var=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a FA class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "FA":
        raise TypeError("'obj' must be a FA class.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show active individuals points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(
        p=ggplot(),
        data=obj.ind_.coord,
        axis=axis,
        geom=geom_ind,
        repel=repel_ind,
        color=col_ind,
        point_args=point_args_ind,
        text_args=text_args_ind
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
    # rescale variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    xscale = (max(obj.ind_.coord.iloc[:,axis[0]])-min(obj.ind_.coord.iloc[:,axis[0]]))/(max(obj.quanti_var_.coord.iloc[:,axis[0]])-min(obj.quanti_var_.coord.iloc[:,axis[0]]))
    yscale = (max(obj.ind_.coord.iloc[:,axis[1]])-min(obj.ind_.coord.iloc[:,axis[1]]))/(max(obj.quanti_var_.coord.iloc[:,axis[1]])-min(obj.quanti_var_.coord.iloc[:,axis[1]]))
    scale = min(xscale, yscale)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_arrow(
        p=p,
        data=obj.quanti_var_.coord.mul(scale),
        axis=axis,
        geom=geom_var,
        repel=repel_var,
        color=col_var,
        segment_args=segment_args_var,
        text_args=text_args_var
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show supplementary continuous variables segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quanti_sup and hasattr(obj,"quanti_var_sup_"):
        p = add_arrow(
            p=p,
            data=obj.quanti_var_sup_.coord.mul(scale),
            axis=axis,
            geom=geom_var,
            repel=repel_var,
            color=col_quanti_sup,
            segment_args=segment_args_quanti_sup,
            text_args=text_args_quanti_sup
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "FA - Biplot of individuals and variables"

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

def fviz_fa(obj,
            choice="biplot",
            **kwargs):
    """
    Visualize Factor Analysis (FA)

    Factor analysis (:class:`~scientisttools.FA`) is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. 
    It is used to identify the structure of the relationship between the variable and the respondent. :class:`~scientisttools.fviz_fa` provides plotnine-based elegant visualization of :class:`~scientisttools.FA` outputs.
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`.

    choice : {"ind","var","biplot"}, default = "ind"
        The graph to plot. Allowed values include:

        * "ind" for the individuals graphs
        * "var" for the variables graphs (= Correlation circle)
        * "biplot" for biplot of individuals and variables

    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_fa_ind`: Graph of individuals
        * :class:`scientisttools.fviz_fa_var`: Graph of variables (=correlation circle)
        * :class:`scientisttools.fviz_fa_biplot`: Biplot of individuals and variables
    
    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_fa_ind`
        Visualize Factor Analysis - Graph of individuals
    :class:`~scientisttools.fviz_fa_var`
        Visualize Factor Analysis - Graph of variables (=correlation circle)
    :class:`~scientisttools.fviz_fa_biplot`
        Visualize Factor Analysis - Biplot of individuals and variables
    
    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, fviz_fa
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    >>> # graph of individuals
    >>> p = fviz_fa(clf, choice = "ind", repel=True)
    >>> print(p.show())
    >>> # graph of variables
    >>> p = fviz_fa(clf, choice = "var", repel=True)
    >>> print(p.show())
    >>> # biplot - graph of individuals and variables
    >>> p = fviz_fa(clf, choice = "biplot", repel_ind=True, repel_var=True)
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_fa_ind(obj,**kwargs)
    elif choice == "var":
        return fviz_fa_var(obj,**kwargs)
    elif choice == "biplot":
        return fviz_fa_biplot(obj,**kwargs)
    else:
        raise ValueError("choice should be one of 'ind', 'var', 'biplot'")