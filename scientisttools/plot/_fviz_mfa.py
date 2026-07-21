# -*- coding: utf-8 -*-
from pandas import DataFrame, concat, merge
from mizani.palettes import brewer_pal
from plotnine import (
    aes,
    arrow,
    ggplot,
    geom_point,
    geom_segment,
    geom_text,
    guides,
    guide_legend,
    scale_color_manual,
    theme_minimal
)

from ._fviz import (
    add_arrow,
    add_scatter,
    fviz_arrow,
    fviz_circle,
    fviz_scatter,
    overlap_coord,
    set_axis
)

from ..methods.functions.get_sup_label import get_sup_label

def fviz_mfa_ind(obj,
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
                 partiel = None,
                 geom_partiel = "arrow",
                 point_args_partiel = dict(size=1.5),
                 segment_args_partiel = dict(size=0.5,alpha=1),
                 text_args_partiel = dict(size=8),
                 add_ellipses = False,
                 ellipse_type = "convex",
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
    Visulize Multiple Factor Analysis - Graph of individuals

    Multiple factor analysis (:class:`~scientisttools.MFA`) is used to analyze a data set in which individuals are described by several sets of variables (continuous and/or categorical) structured into groups. 
    :class:`~scientisttools.fviz_mfa_ind` provides plotnine-based elegant visualization of :class:`~scientisttools.MFA` individuals outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.

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

    partiel : str, list, tuple, default = None
        If string, the name of the individuals for which the partial points should be drawn. Use partial = "all" to visualize partial points for all individuals.

    geom : str, list, tuple, default = "arrow"
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text"). 

        * "arrow" to show only segments.
        * "point" to show only points.
        * "text" to show only labels.
        * ("arrow","text") to show both arrows and texts.
        * ("point","text") to show both points and texts.

    point_args_partiel : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for partiel points.

    segment_args_partiel : dict, default = dict(size=0.5,alpha=1)
        A dictionary containing parameters (except color) for partiel segments.

    text_args_partiel : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for partiel texts.

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
    :class:`~scientisttools.fviz_mfa`
        Visualize Multiple Factor Analysis
    :class:`~scientisttools.get_mfa`
        Extract the results for individuals/variables/group/partial axes - MFA

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, fviz_mfa_ind
    >>> clf = MFA(group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    >>> # graph of individuals
    >>> p = fviz_mfa_ind(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class MFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be a MFA class object")

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
    
    # add partiel individuals
    if ((habillage is None) and 
        (partiel is not None)):
        # set partiel label
        if isinstance(partiel,str) and partiel == "all":
            partiel_label = obj.ind_.coord.index.tolist()
        else:
            partiel_label = get_sup_label(X=obj.ind_.coord,indexes=partiel,axis=0)
        # find partiel coordinates
        coord = DataFrame().astype("float")
        for i, g in enumerate(obj.ind_.coord_partiel._fields):
            data = obj.ind_.coord_partiel[i].loc[partiel_label,:]
            data.loc[:,"habillage"] = g
            data.index = [f"{x}.{g}" for x in data.index]
            coord = concat((coord,data),axis=0)
        # reset index
        coord["habillage"] = coord["habillage"].astype("category")
        # convert to string
        partiel_label = [f"{x}" for x in partiel_label]
        # compromise coordinates
        ind_coord = obj.ind_.coord
        # change index type to string
        coord.index, ind_coord.index = coord.index.astype("str"), ind_coord.index.astype("str")

        # add x and y
        coord["x"], coord["y"] = 0.0,0.0
        for i in partiel_label:
            n = [x for x in coord.index if x.startswith(i)]
            coord.loc[n,"x"] = ind_coord.loc[i,f"Dim{axis[0]+1}"]
            coord.loc[n,"y"] = ind_coord.loc[i,f"Dim{axis[1]+1}"]

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

        # remove point if arrow
        if all(x in geom_partiel for x in ("arrow","point")):
            geom_partiel = [x for x in geom_partiel if x != "point"]

        # overlapped texts
        if repel and ("text" in geom):
            text_args["adjust_text"] = dict(arrowprops=dict(arrowstyle='-',lw=1.0))
        
        # show points
        if "point" in geom_partiel:
            p = p + geom_point(
                data=coord,
                mapping=aes(
                    x = f"Dim{axis[0]+1}",
                    y = f"Dim{axis[1]+1}",
                    color = "habillage"
                ),
                inherit_aes=False,
                **point_args_partiel
            )
        
        # draw segment
        if "arrow" in geom_partiel:
            p = p + geom_segment(
                data=coord,
                mapping=aes(
                    x="x",
                    y="y",
                    xend = f"Dim{axis[0]+1}",
                    yend = f"Dim{axis[1]+1}",
                    color = "habillage"
                ),
                arrow = arrow(angle=30,length=0.2/2.54),
                inherit_aes=False,
                **segment_args_partiel
            )

        if "text" in geom_partiel:
            p = p + geom_text(
                data=coord,
                mapping=aes(
                    x = f"Dim{axis[0]+1}",
                    y = f"Dim{axis[1]+1}",
                    label = coord.index,
                    color = "habillage"
                ),
                inherit_aes=False,
                **text_args_partiel
            )
        
        # set color
        p = (
            p 
            + scale_color_manual(values=colors_mapping)
            + guides(color=guide_legend(title=""),linetype=guide_legend(title=""))
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
        title = "MFA - Graph of individuals"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show others choices
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
    
def fviz_mfa_var(obj,
                 choice = "group",
                 axis=[0,1],
                 geom = ("arrow","point","text"),
                 repel = False,
                 col_var = "black",
                 segment_args = dict(size=0.5),
                 point_args = dict(size=1.5),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = "Dark2",
                 add_ellipses = False,
                 ellipse_type = "convex",
                 level = 0.95,
                 alpha = 0.1,
                 var_sup = True,
                 col_var_sup = "blue",
                 segment_args_var_sup = dict(linetype="dashed",size=0.5),
                 point_args_var_sup = dict(size=1.5),
                 text_args_var_sup = dict(size=8),
                 scale = 1,
                 circle = True,
                 col_circle = "gray",
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
    Visualize Multiple Factor Analysis - Graph of variables/group/partial axes

    Multiple factor analysis (:class:`~scientisttools.MFA`) is used to analyze a data set in which individuals are described by several sets of variables (continuous, categorical or mixed) structured into groups. 
    :class:`~scientisttools.fviz_mfa_var` provides plotnine-based elegant visualization of :class:`~scientisttools.MFA` variables, groups and partial axes outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.

    choice : {"group","partial_axes","levels","freq","quanti_var"}, default = "group"
        The graph to plot. Allowed values include:

        * "group" for groups
        * "partial_axes" for partial axes
        * "levels" for variable categories
        * "freq" for frequencies
        * "quanti_var" for continuous variables (=correlation circle)
    
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
    :class:`~scientisttools.fviz_mfa`
        Visualize Multiple Factor Analysis
    :class:`~scientisttools.get_mfa`
        Extract the results for individuals/variables/group/partial axes - MFA

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, fviz_mfa_var
    >>> clf = MFA(group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    >>> # graph of variables (=correlation circle)
    >>> p = fviz_mfa_var(clf,choice="quanti_var",repel=True)
    >>> print(p.show())
    >>> # graph of groups
    >>> p = fviz_mfa_var(clf,choice="group",repel=True)
    >>> print(p.show())
    >>> # graph of partial axes
    >>> p = fviz_mfa_var(clf,choice ="partial_axes",repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is an object of class MFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be a MFA class")
    
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

        # show supplementary groups points
        if var_sup and ("coord_sup" in obj.group_._asdict().keys()):
            p = add_scatter(
                p = p,
                data = obj.group_.coord_sup,
                axis = axis,
                geom = geom,
                repel = repel,
                color = col_var_sup,
                point_args = point_args_var_sup,
                text_args = text_args_var_sup
            )
    elif choice in ("freq","levels","quanti_var","partial_axes") and hasattr(obj,f"{choice}_"):
        if col_var == "group":
            if choice in ("freq","levels","quanti_var"):
                # extract coordinates
                coord = getattr(obj,f"{choice}_").coord
                # reset index and left join with group
                coord = coord.reset_index().rename(columns={"index" : "variable"})
                coord = merge(coord,obj.call_.col_group,on="variable",how="left")
            else:
                coord = obj.partial_axes_.coord.copy()
                # add group
                coord["group"] = [x.split(".")[1] for x in coord.index]
                # reset index
                coord = coord.reset_index().rename(columns={"index" : "variable"})
            
            # set colors
            index = coord["group"].unique().tolist()
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
                
            # initialization
            p = ggplot(data=coord,mapping=aes(x = f"Dim{axis[0]+1}",y = f"Dim{axis[1]+1}"))

            if choice in ("levels","freq"):
                # overlapping texts
                if repel and "text" in geom:
                    text_args["adjust_text"] = dict(arrowprops=dict(arrowstyle='-',lw=1.0))
                # show points
                if "point" in geom:
                    p = p + geom_point(aes(color="group"),**point_args)
                # show texts
                if "text" in geom:
                    p = p + geom_text(aes(color="group",label = "variable"),**text_args)
            else:
                # define text coordinates
                coord = overlap_coord(coord=coord,axis=axis,repel=repel)
                if repel:
                    x_text, y_text = "xnew", "ynew"
                else:
                    x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"
                # show segments
                if "arrow" in geom:
                    p = (
                        p 
                        + geom_segment(
                            mapping = aes(x=0,y=0,xend = f"Dim{axis[0]+1}",yend = f"Dim{axis[1]+1}",color = "group"),
                            arrow = arrow(angle=30,length=0.2/2.54),
                            **segment_args
                        )
                    )
                # show points
                if "point" in geom:
                    p = p + geom_point(aes(color="group"),**point_args)
                # show texts
                if "text" in geom:
                    p = (
                        p 
                        + geom_text(
                            data = coord,
                            mapping = aes(
                                x = x_text,
                                y = y_text,
                                color = "group",
                                label = "variable"
                            ),
                            inherit_aes = False,
                            show_legend = False,
                            **text_args
                        )
                    )

                # add circle
                if circle:
                    p = fviz_circle(p=p,color=col_circle)

            # scale color manual
            p = (
                p 
                + scale_color_manual(values=colors_mapping,name="") 
                + guides(color=guide_legend(title=""))
            )
        else:
            if choice in ("levels","freq"):
                p = fviz_scatter(
                    obj = obj,
                    choice = choice,
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
            else:
                p = fviz_arrow(
                    obj = obj,
                    choice = choice,
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
            
            # add supplementary points or segment
            if (var_sup and 
                (choice in ("levels","freq","quanti_var")) and 
                hasattr(obj,f"{choice}_sup_")):
                coord_sup = getattr(obj,f"{choice}_sup_").coord
                if choice in ("levels","freq"):
                    p = add_scatter(
                        p = p,
                        data = coord_sup,
                        axis = axis,
                        geom = geom,
                        repel = repel,
                        color = col_var_sup,
                        point_args = point_args_var_sup,
                        text_args = text_args_var_sup
                    )
                else:
                    p = add_arrow(
                        p = p,
                        data = coord_sup,
                        axis = axis,
                        geom = geom,
                        repel = repel,
                        color = col_var_sup,
                        segment_args = segment_args_var_sup,
                        point_args = point_args_var_sup,
                        text_args = text_args_var_sup
                    )

    # set x and y limits
    if choice in ("quanti_var","partial_axes"):
        if x_lim is None:
            x_lim = (-1.1,1.1)
        if y_lim is None:
            y_lim = (-1.1,1.1)
        
    # set title    
    if title is None:
        if choice == "group":
            title = "MFA - Graph of groups"
        elif choice == "levels":
            title = "MFA - Graph of variable categories"
        elif choice == "freq":
            title = "MFA - Graph of frequencies"
        elif choice == "quanti_var":
            title = "MFA - Graph of continuous variables"
        elif choice == "partial_axes":
            title = "MFA - Graph of partial axes"
    
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

def fviz_mfa(obj,
             choice="ind",
             **kwargs):
    """
    Visualize Multiple Factor Analysis

    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (continous and/or categorical) structured into groups.
    :class:`~scientisttools.fviz_mfa` provides plotnine-based elegant visualization of :class:`~scientisttools.MFA` outputs.
    
        * :class:`~scientisttools.fviz_mfa_ind`: Graph of individuals
        * :class:`~scientisttools.fviz_mfa_var`: Graph of variables/group/partial axes

    Parameters
    ----------
    obj : class
        an object of class :class:`~scientisttools.MFA`.
    
    choice : {"ind","group","levels","quanti_var","freq","partial_axes"}, default = "ind"
        The graph to plot. Allowed values include one of : 

        * "ind" for the individuals graphs
        * "group" for groups graphs
        * "levels' for the variable categories graphs
        * "quanti_var" for the variables (= Correlation circle)
        * "freq" for frequencies  graphs
        * "partial_axes" for partial axes graphs
    
    **kwargs: Any
        Parameters use by one of this function. See:
        
        * :class:`scientisttools.fviz_mfa_ind`: Graph of individuals
        * :class:`scientisttools.fviz_mfa_var`: Graph of variables/group/partial axes

    Returns
    -------
    A plotnine object.

    See also
    --------
    :class:`~scientisttools.fviz_mfa_ind`
        Visualize Multiple Factor Analysis - Graph of individuals
    :class:`~scientisttools.fviz_mfa_var`
        Visualize Multiple Factor Analysis - Graph of variables/group/partial axes
    :class:`~scientisttools.get_mfa`
        Extract the results for individuals/variables/group/partial axes - MFA

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, fviz_mfa
    >>> clf = MFA(group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    >>> # graph of individuals
    >>> p = fviz_mfa(clf,choice="ind",repel=True)
    >>> print(p.show())
    >>> # graph of variables (=correlation circle)
    >>> p = fviz_mfa(clf,choice="quanti_var",repel=True)
    >>> print(p.show())
    >>> # graph of groups
    >>> p = fviz_mfa(clf,choice="group",repel=True)
    >>> print(p.show())
    >>> # graph of partial axes
    >>> p = fviz_mfa(clf,choice="partial_axes",repel=True)
    >>> print(p.show())
    """
    if choice == "ind":
        return fviz_mfa_ind(obj,**kwargs)
    elif choice in ("group","levels","quanti_var","freq","partial_axes"):
        return fviz_mfa_var(obj,choice=choice,**kwargs)
    else:
        raise ValueError(f"{choice} is not supported.")