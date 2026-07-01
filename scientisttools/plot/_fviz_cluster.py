# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from scipy.spatial import ConvexHull
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
    scale_color_brewer,
    scale_fill_brewer,
    stat_ellipse,
    theme_minimal,
)

#intern functions
from .fviz import fviz_circle, overlap_coord, add_scatter, add_arrow, set_axis
from ..methods.functions.concat_empty import concat_empty

def fviz_cluster(obj,
                 axis = (0,1),
                 legend_title = None,
                 geom = ("point","text"),
                 point_args = dict(size=1.5),
                 text_args = dict(size=10),
                 segment_args = dict(linetype = "solid",size=0.5),
                 show_clust_cent = False, 
                 center_marker_size = 5,
                 center_arrow_size = 1,
                 repel = True,
                 palette = "Dark2",
                 ellipse = False,
                 geom_ellipse = "polygon",
                 ellipse_type = "convex",
                 ellipse_level = 0.95,
                 ellipse_alpha = 0.2,
                 add_sup = True,
                 color_sup = "black",
                 point_args_sup = dict(shape=">",size=1.5),
                 text_args_sup = dict(size=10),
                 segment_args_sup = dict(linetype = "solid",size=0.5),
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 add_hline = True,
                 add_vline = True,
                 add_grid = True,
                 add_circle = True,
                 col_circle = "gray",
                 ggtheme = theme_minimal()):
    """
    Visualize Clustering Analysis

    Parameters
    ----------
    obj : class
        an object of class :class:`~scientisttools.CatVARHCPC`, :class:`~scientisttools.CatVARKMeansPC`, :class:`~scientisttools.HCPC`, :class:`~scientisttools.KMeansPC`, :class:`~scientisttools.VARHCPC`, :class:`~scientisttools.VARKMeansPC`.

    axis : list, tuple, default = (0,1)
        The dimensions to be plotted.

    legend_title : str, defaut = None
        The title of the legend.

    geom : list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text"). 

        - "arrow" to plot only arrows.
        - "point" to show only points.
        - "text" to show only labels.
        - ("point","text") to show points and labels.
        - ("arrow","text") to show arrows and labels.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing others keyword arguments for active data points (see https://plotnine.org/reference/geom_point.html).

    text_args : dict, default = dict(size = 10)
        A dictionary containing keyword arguments for `geom_text` (see https://plotnine.org/reference/geom_text.html).

    segment_args : dict, default = dict(linetype = "solid",size = 0.5)
        A dictionary containing others keyword arguments for active data segments (see https://plotnine.org/reference/geom_segment.html).

    show_clust_center : bool, default = False
        If True, then plot cluster centers data points.

    center_marker_size : int, default = 5
        Marker size of cluster centers data points.

    center_arrow_size : 1
        Arrow size of cluster centers data points.

    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    palette : str, default = "Dark2"
        The color palette to be used for coloring or filling by groups.

    ellipse : bool, default = False
        If True, then add ellipse to graph.

    geom_ellipse : str, default = "polygon"
        The statistical transformation to use on the data for this layer.

    ellipse_type : str, default = "convex"
        The type of ellipse. 

    ellipse_level : float, default = 0.95
        The confidence level at which to draw the ellipse.

    ellipse_alpha : float, default = 0.2
        The transparency of ellipse fill.

    add_sup : bool, default = False
        If True, add supplementary data points.

    color_sup : str, default = "black"
        The color name for the supplementary data points.

    point_args_sup : dict, default = dict(shape = ">",size = 1.5)
        A dictionary containing others keyword arguments for supplementary data points (see https://plotnine.org/reference/geom_point.html).

    text_args_sup : dict, default = dict(size = 10)
        A dictionary containing keyword arguments for `geom_text` (see https://plotnine.org/reference/geom_text.html).

    segment_args_sup : dict, default = dict(linetype = "solid",size = 0.5)
        A dictionary containing others keyword arguments for supplementary data segments (see https://plotnine.org/reference/geom_segment.html).

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

    add_hline : bool, default = True
        A boolean to either add or not a horizontal line.

    add_vline : bool, default = True
        A boolean to either add or not a vertical line.

    add_grid : bool, default = True
        A boolean to either add or not a grid customization.

    add_circle : bool, default = True
        If True, then draw a circle to graph.

    col_circle : str, default = "gray"
        The color name of the circle.

    ggtheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    p : ggplot
        a plotnine

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
    # check object class name
    if not (obj.__class__.__name__ in ("CatVARHCPC","CatVARKMeansPC","VARHCPC","VARKMeansPC","HCPC","KMeansPC")):
        raise TypeError("'obj' must be an object of class CatVARHCPC, CatVARKMeansPC, VARHCPC, VARKMeansPC, HCPC, KMeansPC")
    
    # valid axis
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > obj.call_.obj.call_.ncp-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # set legend title
    if legend_title is None:
        legend_title = "cluster"

    # extract coordinates
    coord = obj.call_.data_clust
    # rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]
    
    #define text coordinates
    if obj.__class__.__name__ in ("VARHCPC","VARKMeansPC"):
        coord = overlap_coord(coord=coord,axis=axis,repel=repel)
        #set x, y 
        if repel:
            x_text, y_text = "xnew", "ynew"
        else:
            x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"
    
    # Initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",label=coord.index,color=legend_title))
    
    if obj.__class__.__name__ in ("VARHCPC","VARKMeansPC"):
        x_lim, y_lim = (-1.1,1.1), (-1.1,1.1)
        if "arrow" in geom:
            p = (p + 
                 geom_segment(
                     mapping= aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}"), 
                     arrow = arrow(angle=30,length=0.2/2.54,type="open"),
                     **segment_args
                 ) + guides(color=guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text,color=legend_title),**text_args)

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
        if add_circle:
            p = fviz_circle(p=p,color=col_circle)
    else:
        if repel:
            text_args["adjust_text"] = dict(arrowprops={'arrowstyle': '-','lw':1.0})
        if "point" in geom:
            p  = p + geom_point(aes(shape=legend_title),**point_args) + guides(color=guide_legend(title=legend_title))
        if "text" in geom:
            p = p + geom_text(mapping=aes(color=legend_title),**text_args)
        
        # add ellipse
        if ellipse:
            if ellipse_type == "convex":
                index = list(sorted(coord[legend_title].unique()))
                def convex_full(data):
                    if data.shape[0] < 3:
                        return data
                    hull = ConvexHull(data.values)
                    return data.iloc[hull.vertices]
                hulls = DataFrame().astype(float)
                for k in index:
                    data = coord[coord[legend_title]==k].iloc[:,list(axis)]
                    hull = convex_full(data)
                    hull.insert(0,legend_title,k)
                    hulls = concat_empty(hulls,hull,axis=0)
                # convert to category
                hulls[legend_title] = hulls[legend_title].astype("category")
                # add to plot
                p = (
                    p 
                    + geom_polygon(
                        data = hulls,
                        mapping = aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",color=legend_title,fill=legend_title,group=legend_title), 
                        alpha = ellipse_alpha,
                        inherit_aes=False
                    )
                )
            else:
                p = p + stat_ellipse(aes(color=legend_title,fill=legend_title),geom=geom_ellipse,type = ellipse_type,alpha = ellipse_alpha,level=ellipse_level)

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
                    points_args = point_args_sup,
                    text_args = text_args_sup
                )

    # add cluster centers
    if show_clust_cent:
        cluster_center = obj.cluster_.coord.rename_axis(legend_title).reset_index()
        cluster_center[legend_title] = cluster_center[legend_title].astype("category")

        if "point" in geom:
            p = (
                p
                + geom_point(
                    data = cluster_center,
                    mapping = aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",shape=legend_title,color=legend_title),
                    size = center_marker_size,
                    inherit_aes=False
                )
            )
        if "arrow" in geom:
            p = (
                p 
                + geom_segment(
                    data = cluster_center,
                    mapping = aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}",color=legend_title), 
                    arrow = arrow(angle=10,length=0.1,type="closed"),
                    size = center_arrow_size,
                    inherit_aes = False
                )
            )

    p = p + scale_color_brewer(type="qual",palette=palette)

    #set color manual and fill manual
    if ellipse and (obj.__class__.__name__ not in ("VARHCPC","VARKMeansPC")):
        p = p + scale_fill_brewer(type="qual",palette=palette)

    #add others elements
    if title is None:
        title = f"Factor map ({obj.call_.obj.__class__.__name__} - {obj.__class__.__name__})"
    p = set_axis(
        p=p,
        self = obj.call_.obj,
        axis = axis,
        x_lim = x_lim,
        y_lim = y_lim,
        x_label = x_label,
        y_label = y_label,
        title = title,
        add_hline = add_hline,
        add_vline = add_vline,
        add_grid = add_grid,
        ggtheme = ggtheme
    )
    return p