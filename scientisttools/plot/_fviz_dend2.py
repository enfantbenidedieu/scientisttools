# -*- coding: utf-8 -*-
from numpy import arange
from pandas import Series, DataFrame, Categorical
from scipy.cluster.hierarchy import dendrogram, cut_tree, fcluster
from mizani.palettes import brewer_pal
from plotnine import (
    aes,
    coord_cartesian,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_segment,
    geom_text,
    geom_rect,
    ggplot,
    guides,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal
)

def fviz_dend2(Z,
               ncl = None,
               h = None,
               criterion = "distance",
               labels = None,
               color_labels_by_cluster = False,
               add_rect = False,
               rect_border = "gray",
               rect_type = "dashed",
               rect_size = 1,
               rect_fill = False,
               rect_alpha = 0.1,
               lower_rect = 0.5,
               upper_rect = 0.9,
               palette = "Set2",
               text_size = 11,
               line_size = 1,
               family = "sans-serif",
               horiz = False,
               x_label = None,
               y_label = None,
               title = None,
               subtitle = None,
               ggtheme = theme_minimal()):
    """
    Visualization of Dendrogram

    Draws easily beautiful dendrogram using plotnine.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded with the matrix returned by the `https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage` function.

    ncl : int, default = None
        The number of groups for cutting the tree

    h : float, default = None
        Cut the dendrogram by cutting at height h. (k overrides h)

    criterion : str, default = "maxclust"
        The criterion to use in forming flat clusters

    color_labels_by_cluster : bool, default = False
        if True, then labels are colored by cluster.

    add_rect : boo, default = False
        If True, then add a rectangle around groups.

    rect_border : str, default = "gray"
        Border color for rectangles.

    rect_type : str, default = "dashed"
        Border line type for rectangles.

    rect_size : int, default = 1
        Border line size for rectangles.

    rect_fill : bool, default = False
        If True, fill the rectangle.

    rect_alpha : float, default = 0.1
        The transparency of rectange fill.

    lower_rect : float, default = 0.5
        A value of how low should the lower part of the rectange around clusters. Ignored when rect = False.

    upper_rect : float, default = 0.9
        A value of how high should the higher part of the rectange around clusters. Ignored when rect = False.

    palette : str, list, tuple, default = "Set2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    text_size : int, dafeult = 11
        The size for labels.

    line_size : int, default = 1
        The size for segments.

    family : str, default = "sans-serif"
        The font face.

    horiz : bool, default = False
        If True, an horizontal dendrogram is drawn.

    x_label : str, default = None
        The label text of x. If None, then x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then y_label is chosen.

    title : str, default = None
        The title of the graph you draw. If None, then a title is chosen.

    subtitle : str, default = None
        The subtitle of the graph you draw. If None, then a title is chosen.

    ggtheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    p : ggplot
        a plotnine

    See also
    --------
    :class:`~scientisttools.CatVARHCPC`
        Categorical Variables Hierachical Clustering on Principal Components (CatVARHCPC).
    :class:`~scientisttools.HCPC`
        Hierarchical Clustering on Principal Components (HCPC).
    :class:`~scientisttools.VARHCPC`
        Variables Agglomerative Hierachical Clustering on Principal Components (VARHCPC).
    :class:`~scientisttools.fviz_cluster`
        Visualize Clustering Analysis.
    :class:`~scientisttools.fviz_dend`
        Visualization of Dendrogram.

    Reference
    ---------
    https://anyplot.ai/dendrogram-basic/python/pygal
    https://anyplot.ai/dendrogram-basic

    Examples
    --------
    >>> from scientisttools.datasets import usarrests
    >>> from scientisttools import PCA, HCPC, fviz_dend2
    >>> clf = PCA(ncp=3)
    >>> clf.fit(usarrests)
    >>> clf2 = HCPC(ncl=4,consol=False,order=False)
    >>> clf2.fit(clf)
    >>> p = fviz_dend2(obj=clf2.call_.tree.Z,ncl=4,color_labels_by_cluster=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # graph data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # number of observations and number of clusters
    n = Z.shape[0] + 1 
    # labels
    if labels is None:
        labels = [f"row{x+1}" for x in range(n)]
    # dendrogram
    d = dendrogram(Z,labels=labels,no_plot=True)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # get cluster series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if all(x is None for x in (ncl,h)) or (ncl == 1):
        cluster = Series([1]*n,index = labels,name = "cluster",dtype="category")
    elif h is not None and ncl is None:
        cluster = Series(fcluster(Z,t=h,criterion=criterion)+1,index=labels,name="cluster",dtype="category")
    elif all(x is not None for x in (ncl,h)) or ncl > 1:
        cluster = Series((cut_tree(Z,n_clusters=ncl)+1).reshape(-1, ), index=labels,name="cluster",dtype="category")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set palette
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    uk = list(sorted(list(cluster.unique())))
    ncl = len(uk)
    if palette is None:
        palette = "Set2"
    if isinstance(palette,str):
        colors = brewer_pal(type="qual", palette=palette)(len(uk))
    elif isinstance(palette,(list,tuple)):
        if len(palette) != len(uk):
            raise TypeError("Not convenient palette definition")
        if "black" in palette:
            raise ValueError("Change black color with another color")
        colors = palette
    else:
        raise TypeError("palette should be one of str, list of tuple")
    # set color mapping
    colors_mapping = dict(zip(uk,colors))
    colors_mapping["all"] = "black"
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # leafs
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    leafs = {x : y for x,y in zip(cluster.index,cluster.values)}
    nodes = {}
    for i, l in enumerate(labels):
        nodes[i] = {leafs[l]}
    for i, r in enumerate(Z):
        left, right = int(r[0]), int(r[1])
        nodes[n + i] = nodes[left] | nodes[right]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # branch type label for each merge
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    branch_type_labels = {x : y  for x, y in zip(uk,uk)}
    merge_branch_types = []
    for i in range(len(Z)):
        sp = nodes[n + i]
        if len(sp) == 1:
            merge_branch_types.append(branch_type_labels[next(iter(sp))])
        else:
            merge_branch_types.append("all")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # map dendrogram order to linkage order via merge heights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    height_to_merge = {}
    for i, h in enumerate(Z[:, 2]):
        height_to_merge.setdefault(round(h, 10), []).append(i)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # build segment dataframe
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    segments = []
    for xs, ys in zip(d["icoord"], d["dcoord"], strict=True):
        h = round(max(ys), 10)
        if h in height_to_merge and height_to_merge[h]:
            merge_idx = height_to_merge[h].pop(0)
            btype = merge_branch_types[merge_idx]
        else:
            btype = "all"
        segments.append({"x": xs[0], "xend": xs[1], "y": ys[0], "yend": ys[1], "cluster": btype})
        segments.append({"x": xs[1], "xend": xs[2], "y": ys[1], "yend": ys[2], "cluster": btype})
        segments.append({"x": xs[2], "xend": xs[3], "y": ys[2], "yend": ys[3], "cluster": btype})
    segments_df = DataFrame(segments)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Leaf labels with based coloring
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    n_leaves = len(d["ivl"])
    leaf_positions = [(i + 1) * 10 - 5 for i in range(n_leaves)]
    leaf_labels = d["ivl"]
    leaf_btypes = [branch_type_labels[leafs[l]] for l in leaf_labels]
    label_df = DataFrame({"x": leaf_positions, "label": leaf_labels, "y": [0.0] * n_leaves, "cluster": leaf_btypes})

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # append all to list of cluster
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    category_order = uk
    category_order.append("all")
    # convert to ordered categorical 
    segments_df["cluster"] = Categorical(segments_df["cluster"], categories=category_order, ordered=True)
    label_df["cluster"] = Categorical(label_df["cluster"], categories=category_order, ordered=True)

    # Merge node points - highlight where clusters join (plotnine geom_point layer)
    merge_nodes = []
    for xs, ys, btype in zip(d["icoord"], d["dcoord"], merge_branch_types, strict=True):
        cx = (xs[1] + xs[2]) / 2
        cy = max(ys)
        merge_nodes.append({"x": cx, "y": cy, "cluster": btype})
    merge_df = DataFrame(merge_nodes)
    # convert to ordered categorical 
    merge_df["cluster"] = Categorical(merge_df["cluster"], categories=category_order, ordered=True)

    # Key merge threshold: where Setosa separates from the rest
    if ncl > 1:
        height = (Z[n-ncl-1, 2] + Z[n-ncl, 2]) / 2
    else:
        height = max(Z[:, 2])
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # range of axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x range
    x_min = min(segments_df["x"].min(), segments_df["xend"].min())
    x_max = max(segments_df["x"].max(), segments_df["xend"].max())
    x_pad = (x_max - x_min) * 0.06
    # set y range
    y_min, y_max = -lower_rect*height, max(Z[:, 2])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # rectange data frame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    padding = 5
    leaf_df = DataFrame({"obs": d["leaves"],"x": [padding + 10*i for i in range(len(d["leaves"]))]})
    leaf_df["labels"] = leaf_df["obs"].map(dict(enumerate(labels)))
    leaf_df["cluster"] = leaf_df["labels"].map(leafs).astype("category")
    rects_df = leaf_df.groupby("cluster").agg(xmin=("x", "min"),xmax=("x", "max")).reset_index()
    rects_df["xmin"] -= padding - 1
    rects_df["xmax"] += padding - 1
    rects_df["ymin"] = y_min
    rects_df["ymax"] = upper_rect * height

    # set color segments_by_cluster
    if ncl > 1:
        color_segments_by_cluster = True
    else:
        color_segments_by_cluster = False
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # cluster dendrogram
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # initialization
    p = ggplot()

    if horiz is False:
        # add segments
        if color_segments_by_cluster:
            p = p + geom_segment(aes(x="x", xend="xend", y="y", yend="yend", color="cluster"), data=segments_df, size=line_size)
        else:
            p = p + geom_segment(aes(x="x", xend="xend", y="y", yend="yend"), data=segments_df, color="black",size=line_size)

        # add labels
        if color_labels_by_cluster and ncl > 1:
            p = ( 
                p 
                + geom_text(mapping=aes(x="x", y="y", label="label", color="cluster"),data=label_df,angle=90,
                            ha="center",va="top",size=text_size,nudge_y=0,show_legend=False)
            )
        else:
            p = (
                p 
                + geom_text(mapping=aes(x="x", y="y", label="label"),data=label_df,angle=90,
                            ha="center",va="top",size=text_size,nudge_y=0,show_legend=False)
            )

        # color and remove 
        if (color_segments_by_cluster or color_labels_by_cluster) and ncl > 1:
            if add_rect:
                if rect_fill:
                    p = (
                        p 
                        + geom_rect(mapping=aes(xmin="xmin",xmax="xmax",ymin="ymin",ymax="ymax",fill="cluster"),data=rects_df, alpha = rect_alpha)
                        + scale_fill_manual(values=colors_mapping) 
                        + guides(fill=False)
                    )
                else:
                    p = (
                        p 
                        + geom_rect(mapping=aes(xmin="xmin",xmax="xmax",ymin="ymin",ymax="ymax"),data=rects_df, alpha = rect_alpha,
                                    color = rect_border, fill = "white",linetype = rect_type,size = rect_size)
                    )
            # set color manual and remove legend
            p = (
                p 
                + scale_color_manual(values=colors_mapping) 
                + guides(color=False)
            )
        # 
        p = (
            p 
            + scale_x_continuous(breaks=[], expand=(0.04, 0))
            + scale_y_continuous(breaks=list(arange(0, y_max+1, 2)), expand=(0.10, 0))
            + coord_cartesian(xlim=(x_min - x_pad, x_max + x_pad), ylim=(y_min, y_max))
        )

        # set x_label, y_label and title
        if x_label is None:
            x_label = ""
        if y_label is None:
            y_label = "Height"
        if title is None:
            title = "Cluster Dendrogram"
        if subtitle is None:
            subtitle = ""
        p = p + labs(x=x_label,y=y_label,title=title,subtitle=subtitle)
            
        # add theme
        p = (
            p 
            + ggtheme
            + theme(
                text = element_text(size=text_size, family=family),
                axis_title_x = element_blank(),
                axis_text_x = element_blank(),
                axis_ticks_major_x = element_blank(),
                plot_background = element_rect(fill="#FAFAFA", color="none"),
                panel_background = element_rect(fill="#FAFAFA", color="none"),
                panel_grid_major_x = element_blank(),
                panel_grid_minor_x = element_blank(),
                panel_grid_minor_y = element_blank(),
                panel_grid_major_y = element_line(alpha=0.2, size=0.5, color="#CCCCCC")
            )
        )
    else:
        # add segments
        if color_segments_by_cluster:
            p = p + geom_segment(aes(y="x", yend="xend", x="y", xend="yend", color="cluster"), data=segments_df, size=line_size)
        else:
            p = p + geom_segment(aes(y="x", yend="xend", x="y", xend="yend"), data=segments_df, color="black",size=line_size)

        # add labels
        if color_labels_by_cluster and ncl > 1:
            p = ( 
                p 
                + geom_text(mapping=aes(y="x", x="y", label="label", color="cluster"),data=label_df,angle=0,
                            ha="right",va="center",size=text_size,show_legend=False)
            )
        else:
            p = (
                p 
                + geom_text(mapping=aes(y="x", x="y", label="label"),data=label_df,angle=0,
                            ha="right",va="center",size=text_size,show_legend=False)
            )

        # color and remove 
        if color_segments_by_cluster or color_labels_by_cluster:
            if add_rect:
                if rect_fill:
                    p = (
                        p 
                        + geom_rect(mapping=aes(ymin="xmin",ymax="xmax",xmin="ymin",xmax="ymax",fill="cluster"),data=rects_df, alpha = rect_alpha)
                        + scale_fill_manual(values=colors_mapping) 
                        + guides(fill=False)
                    )
                else:
                    p = (
                        p 
                        + geom_rect(mapping=aes(ymin="xmin",ymax="xmax",xmin="ymin",xmax="ymax"),data=rects_df, alpha = rect_alpha,
                                    color = rect_border, fill = "white",linetype = rect_type,size = rect_size)
                    )
            # set color manual and remove legend
            p = (
                p 
                + scale_color_manual(values=colors_mapping) 
                + guides(color=False)
            )
        # 
        p = (
            p 
            + scale_y_continuous(breaks=[], expand=(0.04, 0))
            + scale_x_continuous(breaks=list(arange(0, y_max+1, 2)), expand=(0.10, 0))
            + coord_cartesian(ylim=(x_min - x_pad, x_max + x_pad), xlim=(y_min, y_max))
        )

        # set x_label, y_label and title
        if x_label is None:
            x_label = ""
        if y_label is None:
            y_label = "Height"
        if title is None:
            title = "Cluster Dendrogram"
        if subtitle is None:
            subtitle = ""
        p = p + labs(y=x_label,x=y_label,title=title,subtitle=subtitle)

        # add theme
        p = (
            p 
            + ggtheme
            + theme(
                text = element_text(size=text_size, family=family),
                axis_title_y = element_blank(),
                axis_text_y = element_blank(),
                axis_ticks_major_x = element_blank(),
                plot_background = element_rect(fill="#FAFAFA", color="none"),
                panel_background = element_rect(fill="#FAFAFA", color="none"),
                panel_grid_major_x = element_blank(),
                panel_grid_minor_x = element_blank(),
                panel_grid_minor_y = element_blank(),
                panel_grid_major_y = element_line(alpha=0.2, size=0.5, color="#CCCCCC")
            )
        )
    return p