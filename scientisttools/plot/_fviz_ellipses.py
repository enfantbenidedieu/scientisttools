# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from sklearn.utils.validation import check_is_fitted
from mizani.palettes import brewer_pal
from plotnine import (
    aes,
    ggplot,
    geom_point, 
    geom_polygon,
    geom_text,
    facet_wrap,
    scale_color_manual,
    scale_fill_manual,
    stat_ellipse,
    theme_bw
)

# interns functions
from ..methods.others import data_ellipse
from ..methods.functions.get_sup_label import get_sup_label
from ._fviz import (
    check_is_valid_axis,
    check_is_valid_geom,
    set_axis
)

def fviz_ellipses(obj,
                  axis = [0,1],
                  geom = ("point","text"),
                  repel = False,
                  point_args = dict(size=1.5),
                  text_args = dict(size=8),
                  habillage = None,
                  palette = "Dark2",
                  add_ellipses = False, 
                  ellipse_type = "confidence",
                  level = 0.95,
                  alpha = 0.1,
                  x_lim = None,
                  y_lim = None,
                  x_label = None,
                  y_label = None,
                  title = None,
                  subtitle = None,
                  pntheme = theme_bw(),
                  **kwargs):
    """
    Draw ellipses around the categories

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.MCA`,:class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA` or :class:`~scientisttools.DMFA`.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both points and labels.

    repel : bool, default = False
        Whether to avoid overplotting individuals text labels or not.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for individuals points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for individuals texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    habillage : list, tuple, default = None 
        The indexes or names of the categorical variables.

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

    pntheme : function, default = theme_bw() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.

    Returns
    -------
    A plotnine object.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import FAMD, fviz_ellipses
    >>> clf = FAMD()
    >>> clf.fit(wine.data)
    >>> p = fviz_ellipses(clf,habillage=("Label", "Soil"),add_ellipses=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid obj
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not ( obj.__class__.__name__ in ("PCA","MCA","FAMD","PCAmix","MPCA","MFA","DMFA")):
        raise TypeError(f"{obj.__class__.__name__} class is not supported.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=0)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if habillage is not None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if habillage is None:
        raise ValueError("habillage must be assign.")
    elif not isinstance(habillage,(list,tuple)):
        raise TypeError("habilage must be either a list or tuple.")
    elif len(habillage) < 2:
        raise ValueError("habillage must have length greater than or equal to 2")
    # labels
    labels = get_sup_label(obj.call_.Xtot,indexes=habillage,axis=1)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # coordinates of individuals
    coord = concat((obj.ind_.coord.iloc[:,axis],obj.call_.Xtot),axis=1)
    # drop supplementary individuals
    if hasattr(obj,"ind_sup_"):
        coord = coord.drop(index=obj.call_.ind_sup)

    # set data
    data, df_ells = DataFrame().astype("float"), DataFrame().astype("float")
    for k in labels:
        df = coord.loc[:,[f"Dim{axis[0]+1}",f"Dim{axis[1]+1}",k]]
        # element for ellipse (convex or confidence)
        if add_ellipses and ellipse_type in ("confidence","convex"):
            df_ell = data_ellipse(X=df,ellipse_type=ellipse_type,axis=axis,level=level)
            df_ell["Variable"] = k
            df_ell = df_ell.rename(columns={k : "habillage"})
            df_ells = concat((df_ells,df_ell),axis=0,ignore_index=True)
        # 
        df["Variable"] = k
        df = df.rename(columns={k : "habillage"}).reset_index().rename(columns={"index" : "rownames"})
        data = concat((data,df),axis=0,ignore_index=True)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set palette
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # unique categories
    categories = data["habillage"].unique().tolist()
    if isinstance(palette,str):
        colors = brewer_pal(type="qual", palette=palette)(len(categories))
    elif isinstance(palette,(list,tuple)):
        if len(palette) != len(categories):
            raise TypeError("Not convenient palette definition")
        colors = palette
    else:
        raise TypeError("palette should be one of str, list of tuple")
    # set color mapping
    colors_mapping = dict(zip(categories,colors))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set text arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and ("text" in geom):
        text_args["adjust_text"] = dict(arrowprops=dict(lw=1.0))

    # initialization
    p = ggplot(
        data=data,
        mapping=aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",label="rownames")
    )
    # show points
    if "point" in geom:
        p = p + geom_point(aes(color="habillage"),**point_args,show_legend=False)
    # show texts
    if "text" in geom:
        p = p + geom_text(aes(color="habillage"),**text_args,show_legend=False)

    # draw ellipses
    if ellipse:
        if ellipse_type in ("confidence","convex"):
            p = (
                p 
                + geom_polygon(
                    data = df_ells,
                    mapping = aes(
                        x = f"Dim{axis[0]+1}",
                        y = f"Dim{axis[1]+1}",
                        color = "habillage",
                        fill = "habillage",
                        group = "habillage"
                    ), 
                    alpha = alpha,
                    show_legend = False,
                    inherit_aes = False
                )
            )
        else:
            p = (
                p 
                + stat_ellipse(
                    mapping = aes(color="habillage",fill="habillage",group="habillage"),
                    geom = "polygon",
                    type = ellipse_type,
                    alpha = alpha,
                    level = level,
                    show_legend = False
                )
            )
        # scale fill manually
        p = p + scale_fill_manual(values=colors_mapping)

    # scale color mannually and split by categorical variables
    p = (
        p 
        + scale_color_manual(values=colors_mapping) 
        + facet_wrap("Variable")
    ) 

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = f"{obj.__class__.__name__} - Ellipse Plot"

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
        pntheme = pntheme,
        **kwargs
    )
    return p 