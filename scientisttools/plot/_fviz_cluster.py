# -*- coding: utf-8 -*-
from pandas import concat
from mizani.palettes import brewer_pal
from plotnine import (
    aes,
    geom_point,
    geom_segment,
    geom_text,
    geom_polygon,
    arrow,
    ggplot,
    guide_legend,
    guides,
    scale_color_manual,
    scale_fill_manual,
    stat_ellipse,
    theme_minimal
)

#intern functions
from ._fviz import (
    add_arrow, 
    add_scatter,
    check_is_valid_axis,
    check_is_valid_geom,
    fviz_circle, 
    overlap_coord, 
    set_axis
)
from ..methods.others._confidence_ellipse import confidence_ellipse
from ..methods.others._convex_ellipse import convex_ellipse

def fviz_cluster(obj,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = True,
                 point_args = dict(size=1.5),
                 segment_args = dict(size=0.5),
                 text_args = dict(size=8),
                 cluster_center = False, 
                 center_marker_size = 5,
                 center_arrow_size = 1,
                 legend_title = None,
                 palette = "Dark2",
                 ellipse = False,
                 ellipse_type = "confidence",
                 level = 0.95,
                 alpha = 0.2,
                 add_sup = True,
                 color_sup = "black",
                 point_args_sup = dict(shape=">",size=1.5),
                 segment_args_sup = dict(linetype="dashed",size=0.5),
                 text_args_sup = dict(size=8),
                 circle = True,
                 col_circle = "gray",
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
    Visualize Clustering Analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CatVARHCPC`, :class:`~scientisttools.CatVARKMeansPC`, :class:`~scientisttools.HCPC`, :class:`~scientisttools.KMeansPC`, :class:`~scientisttools.VARHCPC`, :class:`~scientisttools.VARKMeansPC`.

    axis : list, default = [0,1]
        The dimensions to be plotted.
    
    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text"). 

        * "arrow" to plot only arrows.
        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show points and labels.
        * ("arrow","text") to show arrows and labels.

    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    segment_args : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color) for segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    cluster_center : bool, default = False
        If True, then plot cluster centers data points.

    center_marker_size : int, default = 5
        Marker size of cluster centers data points.

    center_arrow_size : 1
        Arrow size of cluster centers data points.

    legend_title : str, defaut = None
        The title of the legend.

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    ellipse : bool, default = False
        If True, draws ellipses around the points.

    ellipse_type : str, default = "confidence"
        String specifying frame type. Possible values are : "convex", "confidence" or types supported by `plotnine.stat_ellipse <https://plotnine.org/reference/stat_ellipse.html>` including one of "t", "norm" or "euclid" for plotting concentration ellipses.

        * "convex": plot convex hull of a set of points as :class:`~scientisttools.convex_ellipse`.
        * "confidence": plot confidence ellipses around group mean points as :class:`~scientisttools.confidence_ellipse`.
        * "t": assumes a multivariate t-distribution.
        * "norm": assumes a multivariate normal distribution.
        * "eulclid": draws a circle with the radius equal to `level`, representing the euclidean distance from the center.

    level : float, default = 0.95
        The confidence level at which to draw the ellipse.

    alpha : float, default = 0.2
        The transparency of ellipse fill.

    add_sup : bool, default = False
        If True, add supplementary data points or segments.

    color_sup : str, default = "black"
        The color name for the supplementary data points or segments.

    point_args_sup : dict, default = dict(shape = ">",size = 1.5)
        A dictionary containing parameters for supplementary data points.

    segment_args_sup : dict, default = dict(linetype="solid",size = 0.5)
        A dictionary containing parameters for supplementary segments.

    text_args_sup : dict, default = dict(size = 8)
        A dictionary containing parameters for supplementary texts.

    circle : bool, default = True
        If True, then draw a circle to graph.

    col_circle : str, default = "gray"
        The color name of the circle.
    
    x_lim : list, tuple, default = None
        The range of the plotted x values.

    y_lim : list, tuple, default = None
        The range of the plotted y values.

    x_label : str, default = None
        The label text of x. If None, then x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then y_label is chosen.

    title : str, default = None
        The title of the graph you draw. If None, then a title is chosen.

    subtitle : str, default = None
        The subtitle of the graph you draw.

    hline : bool, default = True
        A boolean to either add or not a horizontal line.

    vline : bool, default = True
        A boolean to either add or not a vertical line.

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.

    Returns
    -------
    A plotnine object.

    Examples
    --------
    >>> from scientisttools.datasets import usarrests
    >>> from scientisttools import PCA, HCPC, fviz_cluster
    >>> clf = PCA(ncp=3)
    >>> clf.fit(usarrests)
    >>> clf2 = HCPC(ncl=4,consol=False,order=False)
    >>> clf2.fit(clf)
    >>> p = fviz_cluster(obj=clf2,repel=True,show_clust_cent=True,ellipse=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check object class name
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("CatVARHCPC","CatVARKMeansPC","VARHCPC","VARKMeansPC","HCPC","KMeansPC")):
        raise TypeError("'obj' must be an object of class CatVARHCPC, CatVARKMeansPC, VARHCPC, VARKMeansPC, HCPC, KMeansPC")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj.call_.obj,axis=axis)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set legend title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if legend_title is None:
        legend_title = "cluster"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # extract coordinates
    coord = obj.call_.data_clust
    # rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set palette
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # unique cluster
    uk = coord[legend_title].unique().tolist()
    if isinstance(palette,str):
        colors = brewer_pal(type="qual", palette=palette)(len(uk))
    elif isinstance(palette,(list,tuple)):
        if len(palette) != len(uk):
            raise TypeError("Not convenient palette definition")
        colors = palette
    else:
        raise TypeError("palette should be one of str, list of tuple")
    # set color mapping
    colors_mapping = dict(zip(uk,colors))
    
    # define text coordinates
    if obj.__class__.__name__ in ("VARHCPC","VARKMeansPC"):
        coord = overlap_coord(coord=coord,axis=axis,repel=repel)
        # set x, y 
        if repel:
            x_text, y_text = "xnew", "ynew"
        else:
            x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"
    
    # Initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",label=coord.index,color=legend_title))
    
    if obj.__class__.__name__ in ("VARHCPC","VARKMeansPC"):
        # check if valid geom
        check_is_valid_geom(geom=geom,axis=1)

        # set x and y limits
        if x_lim is None:
            x_lim = (-1.1,1.1)
        if y_lim is None:
            y_lim = (-1.1,1.1)
        
        # show segments
        if "arrow" in geom:
            p = (p + 
                 geom_segment(
                     mapping= aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}"), 
                     arrow = arrow(angle=30,length=0.2/2.54,type="open"),
                     **segment_args
                 ) + guides(color=guide_legend(title=legend_title)))
        # show texts
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text,color=legend_title),**text_args,show_legend=False)

        # add supplementary continuous variables
        if add_sup and hasattr(obj,"quanti_var_sup_"):
            coord_sup = concat((obj.call_.obj.quanti_var_sup_.coord,obj.quanti_var_sup_.cluster),axis=1)
            p = add_arrow(
                p=p,
                data=coord_sup,
                axis=axis,
                geom=geom,
                repel=repel,
                color=color_sup,
                segment_args=segment_args_sup,
                text_args=text_args_sup
            )
        #add correlation circle
        if circle:
            p = fviz_circle(p=p,color=col_circle)
    else:
        # check if valid geom
        check_is_valid_geom(geom=geom,axis=0)

        # add adjust_text to text_args
        if repel and ("text" in geom):
            text_args["adjust_text"] = dict(arrowprops=dict(arrowstyle='-',lw=1.0))

        # show points
        if "point" in geom:
            p  = (
                p 
                + geom_point(aes(shape=legend_title),**point_args) 
                + guides(color=guide_legend(title=legend_title))
            )

        # show texts
        if "text" in geom:
            p = p + geom_text(mapping=aes(color=legend_title),**text_args)
        
        # show ellipse
        if ellipse:
            if ellipse_type in ("confidence","convex"):
                # data preparation
                data = coord.loc[:,[f"Dim{axis[0]+1}",f"Dim{axis[1]+1}",legend_title]]
                # confidence ellipse
                if ellipse_type == "confidence":
                    df_ells = confidence_ellipse(X=data,axis=axis,level=level)
                # convex ellipse
                else:
                    df_ells = convex_ellipse(X=data,axis=axis)
                # add to plot
                p = (
                    p 
                    + geom_polygon(
                        data = df_ells,
                        mapping = aes(
                            x = f"Dim{axis[0]+1}",
                            y = f"Dim{axis[1]+1}",
                            color = legend_title,
                            fill = legend_title,
                            group = legend_title
                        ), 
                        alpha = alpha,
                        inherit_aes=False
                    )
                )
            else:
                p = (
                    p 
                    + stat_ellipse(
                        mapping=aes(color=legend_title,fill=legend_title),
                        geom = "polygon",
                        type = ellipse_type,
                        alpha = alpha,
                        level = level
                    )
                )

        # add supplementary elements
        if add_sup:
            coord_sup = None
            if hasattr(obj,"ind_sup_"):
                coord_sup = concat((obj.call_.obj.ind_sup_.coord,obj.ind_sup_.cluster),axis=1)
            if hasattr(obj,"levels_sup_"):
                coord_sup = concat((obj.call_.obj.levels_sup_.coord,obj.levels_sup_.cluster),axis=1)
            if coord_sup is not None:
                p = add_scatter(
                    p = p,
                    data = coord_sup,
                    axis = axis,
                    geom = geom,
                    repel = repel,
                    color = color_sup,
                    point_args = point_args_sup,
                    text_args = text_args_sup
                )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add cluster centers
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if cluster_center:
        cluster_coord = obj.cluster_.coord.rename_axis(legend_title).reset_index()
        cluster_coord[legend_title] = cluster_coord[legend_title].astype("category")
        # show points
        if "point" in geom:
            p = (
                p
                + geom_point(
                    data = cluster_coord,
                    mapping = aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",shape=legend_title,color=legend_title),
                    size = center_marker_size,
                    inherit_aes=False
                )
            )
        # show segments
        if "arrow" in geom:
            p = (
                p 
                + geom_segment(
                    data = cluster_coord,
                    mapping = aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}",color=legend_title), 
                    arrow = arrow(angle=10,length=0.1,type="closed"),
                    size = center_arrow_size,
                    inherit_aes = False
                )
            )

    # set color manual
    p = p + scale_color_manual(values=colors_mapping)
    
    # set fill manual
    if ellipse and (obj.__class__.__name__ not in ("VARHCPC","VARKMeansPC")):
        p = p + scale_fill_manual(values=colors_mapping)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    if title is None:
        title = f"Factor map ({obj.call_.obj.__class__.__name__} - {obj.__class__.__name__})"

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = set_axis(
        p=p,
        obj = obj.call_.obj,
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