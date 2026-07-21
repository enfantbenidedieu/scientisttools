# -*- coding: utf-8 -*-
from numpy import linspace, sin, cos, pi, issubdtype,number,asarray,ndarray,sqrt,arctan2
from pandas import DataFrame, Series, concat
from mizani.palettes import brewer_pal
from sklearn.utils.validation import check_is_fitted
from plotnine import (
    ggplot,
    aes,
    geom_point, 
    geom_segment,
    geom_ribbon,
    scale_color_gradient2,
    guides,
    guide_legend,
    stat_ellipse,
    scale_color_manual,
    scale_fill_manual,
    geom_polygon,
    arrow,
    theme_minimal,
    labs, 
    xlim,ylim,
    geom_hline, 
    geom_vline,
    theme,
    geom_text
)

# interns functions
from ..methods.others import data_ellipse

#list of colors
list_colors = ["black", "red", "green", "blue", "cyan", "magenta","darkgray", "darkgoldenrod", "darkgreen", "violet",
                "turquoise", "orange", "lightpink", "lavender", "yellow","lightgreen", "lightgrey", "lightblue", "darkkhaki",
                "darkmagenta", "darkolivegreen", "lightcyan", "darkorange","darkorchid", "darkred", "darksalmon", 
                "darkseagreen","darkslateblue", "darkslategray", "darkslategrey","darkturquoise", 
                "darkviolet", "lightgray", "lightsalmon","lightyellow", "maroon"]

# list of lines
list_lines = ['solid','dashed','dotted','dashdot','longdash','twodash']

def check_is_valid_axis(obj,
                        axis=[0,1]):
    """
    check is valid axis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.CA`, :class:`~scientisttools.MCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`
    
    axis : list, default = [0,1]
        The dimensions to be plotted.

    Returns
    -------
    None
    """
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")
    elif ((len(axis) != 2) or 
        (axis[0] < 0) or 
        (axis[1] > obj.call_.ncp-1)  or 
        (axis[0] > axis[1])):
        raise ValueError("You must pass a valid 'axis'.")
    
def check_is_valid_geom(geom,
                        axis=0):
    """
    check is valid geom

    Parameters
    ----------
    geom : str, list, tuple
        The geometry to be used for the graph.

    axis : {0,1,"index","column"}, default = 0
        which axis to aggregate.

        * None or 0 or "index" indicates aggregating along rows
        * 1 or "columns" indicates aggregating along columns

    Returns
    -------
    None    
    """
    if axis in (0,"index"):
        if isinstance(geom,str):
            if geom not in ("point","text"):
                raise ValueError("The specified value for the argument geom are not allowed")
        elif isinstance(geom,(list,tuple)):
            intersect = [x for x in geom if x in ("point","text")]
            if len(intersect) == 0:
                raise ValueError("The specified value(s) for the argument geom are not allowed")
    elif axis in (1,"columns"):
        if isinstance(geom,str):
            if geom not in ("arrow","point","text"):
                raise ValueError("The specified value for the argument geom are not allowed")
        elif isinstance(geom,(list,tuple)):
            intersect = [x for x in geom if x in ("arrow","point","text")]
            if len(intersect) == 0:
                raise ValueError("The specified value(s) for the argument geom are not allowed")

def coord_adjust(coord,
                 cos2,
                 contrib,
                 color = "black",
                 legend_title = None,
                 habillage = None,
                 palette = "Dark2",
                 lim_cos2 = None,
                 lim_contrib = None):
    """
    Coordinates Adjustment
   
    Parameters
    ----------
    coord : DataFrame of shape (n_samples, n_components)
        Factor coordinates.

    cos2 : DataFrame of shape (n_samples, n_components)
        Squared cosinus.

    contrib : DataFrame of shape (n_samples, n_components)
        Contributions contributions.

    color : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for individuals and variables, respectively. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for individuals/variables are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 
        To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    legend_title : str, defaut = None
        The title of the legend.
    
    habillage : str, default = None
        an optional factor variable for coloring the observations by groups. 

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    lim_cos2 : float, default = None
        The square cosinus limit.

    lim_contrib : float, default = None
        The relative contribution limit.

    Returns
    -------
    coord : DataFrame of shape (n_samples, n_components)
        Factor coordinates.

    legend_title : str
        Legend title.

    color_mapping : dict
        Mapping data values to colors.
    """
    # initialization of index
    index, colors_mapping = None, None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set color
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(color,str):
        if color == "cos2":
            if legend_title is None:
                legend_title = "Cos2"
            coord.insert(coord.shape[1],legend_title,cos2.sum(axis=1))
        elif color == "contrib":
            if legend_title is None:
                legend_title = "Contrib"
            coord.insert(coord.shape[1],legend_title,contrib.sum(axis=1))
        elif color == "coord":
            if legend_title is None:
                legend_title = "Coord"
            coord.insert(coord.shape[1],legend_title,coord.iloc[:,:2].pow(2).sum(axis=1))
        elif color in coord.columns:
            if not issubdtype(coord[color].dtype,number):
                raise TypeError("'color' must be a numeric variable.")
            legend_title = color
    elif (isinstance(color,ndarray) and 
          (len(color) == coord.shape[0]) and 
          all(isinstance(x,(int,float)) for x in color)):
        if legend_title is None:
            legend_title = "Cont_Var"
        coord.insert(coord.shape[1],legend_title,asarray(color))
    elif hasattr(color,"labels_"):
        if legend_title is None:
            legend_title = "Cluster"
        coord.insert(coord.shape[1],legend_title,[str(x+1) for x in color.labels_])
        coord[legend_title] = coord[legend_title].astype("category")
        index = coord[legend_title].unique().tolist()
    elif (isinstance(color,(list,tuple,Series)) and 
          (len(color) == coord.shape[0]) and 
          all(isinstance(x,str) for x in color)):
        if legend_title is None:
            legend_title = "Group"
        coord.insert(coord.shape[1],legend_title,color)
        coord[legend_title] = coord[legend_title].astype("category")
        index = coord[legend_title].unique().tolist()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set palette
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ((habillage is not None) or 
        hasattr(color,"labels_") or 
        (isinstance(color,(list,tuple,Series)) and 
         (len(color) == coord.shape[0]) and 
         all(isinstance(x,str) for x in color))):  
        # index for habillages  
        if habillage is not None:
            index = coord[habillage].unique().tolist()
        
        # set colors
        if isinstance(palette,str):
            colors = brewer_pal(type="qual",palette=palette)(len(index))
        elif isinstance(palette,(list,tuple)):
            if len(palette) != len(index):
                raise TypeError("Not convenient palette definition")
            colors = palette
        else:
            raise TypeError("palette should be one of str, list of tuple")
        # set color mapping
        colors_mapping = dict(zip(index,colors))
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # using lim cos2
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = (
                cos2
                .sum(axis=1)
                .to_frame("cosinus")
                .sort_values(by="cosinus",ascending=False)
                .query("cosinus > @lim_cos2")
            )
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # using lim contrib
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = (
                contrib
                .sum(axis=1)
                .to_frame("contrib")
                .sort_values(by="contrib",ascending=False)
                .query("contrib > @lim_contrib")
            )
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    return coord, legend_title, colors_mapping

def fviz_scatter(obj,
                 choice = "ind",
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 color = "black",
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
                 lim_cos2 = None,
                 lim_contrib = None):
    """
    Create a scatter plot with text
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.MCA`, :class:`~scientisttools.MFA`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.PCA`, :class:`~scientisttools.PCAmix`

    choice : str, default = "ind"
        Name of the active choices.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text"). 

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both types.
    
    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    color : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for individuals. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for individuals are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 
        To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

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

    lim_cos2 : float, default = None
        The cos2 limit.

    lim_contrib : float, default = None
        The relative contribution limit.

    Returns
    -------
    A plotnine object.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_scatter
    >>> # instanciation
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> # graph of individuals
    >>> p = fviz_scatter(clf,choice="ind",repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid obj
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not ( obj.__class__.__name__ in ("PCA","CA","MCA","FAMD","PCAmix","MPCA","MFA","DMFA")):
        raise TypeError(f"{obj.__class__.__name__} class is not supported.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if geom is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=0)

    # remove point if arrow
    if (isinstance(geom,(list,tuple)) and  
        all(x in geom for x in ("arrow","point"))):
        geom = [x for x in geom if x != "arrow"]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # extract model attribute
    mattr = getattr(obj,f"{choice}_")
    # split : coordinates, square cosinus, contributions
    coord, cos2, contrib = mattr.coord.iloc[:,axis], mattr.cos2.iloc[:,axis], mattr.contrib.iloc[:,axis]
    if choice in ("ind","row"):
        coord = concat((coord,obj.call_.Xtot),axis=1)
        if hasattr(obj,"ind_sup_"):
            coord = coord.drop(index=obj.call_.ind_sup)
        if hasattr(obj,"row_sup_"):
            coord = coord.drop(index=obj.call_.row_sup)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set text arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and ("text" in geom):
        text_args["adjust_text"] = dict(arrowprops=dict(lw=1.0))
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # update coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    coord, legend_title, color_mapping = coord_adjust(
        coord=coord,
        cos2=cos2,
        contrib=contrib,
        color=color,
        legend_title=legend_title,
        habillage=habillage,
        palette=palette,
        lim_cos2=lim_cos2,
        lim_contrib=lim_contrib
    )
    
    # initialization
    p = ggplot(
        data = coord,
        mapping = aes(x = f"Dim{axis[0]+1}",y=f"Dim{axis[1]+1}",label=coord.index)
    )

    if habillage is None : 
        if ((color in [*["cos2","contrib","coord"],*coord.columns.tolist()]) or 
            (isinstance(color,ndarray) and 
             (len(color) == coord.shape[0]) and 
             all(isinstance(x,(int,float)) for x in color)) or 
            hasattr(color,"labels_") or 
            (isinstance(color,(list,tuple,Series)) and 
             (len(color) == coord.shape[0]) and 
             all(isinstance(x,str) for x in color))):
            # show points
            if "point" in geom:
                p = p + geom_point(aes(color=legend_title),**point_args)
            # show labels
            if "text" in geom:
                p = p + geom_text(aes(color=legend_title),**text_args)
            
            # scale color gradient
            if ((color in [*["cos2","contrib","coord"],*coord.columns.tolist()]) or 
                (isinstance(color,ndarray) and 
                 (len(color) == coord.shape[0]) and 
                 all(isinstance(x,(int,float)) for x in color))):
                # scale color gradient
                p = (
                    p 
                    + scale_color_gradient2(
                        low = gradient_cols[0],
                        high = gradient_cols[2],
                        mid = gradient_cols[1],
                        name = legend_title
                    )
                )    
        else:
            # show points
            if "point" in geom:
                p = p + geom_point(color=color,**point_args)
            # show labels
            if "text" in geom:
                p = p + geom_text(color=color,**text_args)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in geom:
            p = p + geom_point(aes(color=habillage),**point_args)
        if "text" in geom:
            p = p + geom_text(aes(color=habillage),**text_args)
        
    # add ellipse and set color and fill
    if ((habillage is not None) or 
        hasattr(color,"labels_") or 
        (isinstance(color,(list,tuple,Series)) and 
         (len(color) == coord.shape[0]) and 
         all(isinstance(x,str) for x in color))):
        # set column name
        col_name = habillage if habillage is not None else legend_title
        # add ellipse
        if add_ellipses:
            if ellipse_type in ("convex","confidence"):
                # construct ellipse data
                data = data_ellipse(
                    X = concat((coord.iloc[:,axis],coord[col_name]),axis=1),
                    ellipse_type = ellipse_type,
                    axis = axis,
                    level = level
                )

                # add to plot
                p = (
                    p 
                    + geom_polygon(
                        data = data,
                        mapping = aes(
                            x = f"Dim{axis[0]+1}",
                            y = f"Dim{axis[1]+1}",
                            color = col_name,
                            fill = col_name,
                            group = col_name
                        ), 
                        alpha = alpha,
                        inherit_aes = False
                    )
                )
            else:
                p = (
                    p 
                    + stat_ellipse(
                        mapping = aes(color=col_name,fill=col_name),
                        geom = "polygon",
                        type = ellipse_type,
                        alpha = alpha,
                        level = level
                    )
                )
            # scale fill manual
            p = p + scale_fill_manual(values=color_mapping)
        # scale color manual
        p = (
            p 
            + scale_color_manual(values=color_mapping)
            + guides(color=guide_legend(title=col_name))
        )
        
    return p

def overlap_coord(coord,
                  axis = [0,1],
                  repel = False,
                  a = 0.03,
                  b = 0.07):
    """
    Overlapping coordinates

    Parameters
    ----------
    coord : DataFrame of shape (n_samples, n_components)
        Factor coordinates.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    repel : bool, default = True
        Whether to avoid overplotting text labels or not.

    Returns
    -------
    coord : DataFrame of shape (n_samples, n_components + 2)
        Data with overlapped coordinates.

    References
    ----------
    [1] https://stackoverflow.com/questions/64935396/how-do-i-shift-the-geom-text-labels-to-after-a-geom-segment-arrow-in-ggplot2
    """
    def rshift(r,
               theta,
               a=0.03,
               b=0.07):
        """
        Radial shift

        Radial shift function to move the labels away from the arrows.

        Parameters
        ----------
        r : float
            Radius

        theta : float
            Angle
        
        a : float, default = 0.03
            Offset

        b : float, default = 0.07
            Amplitude (scaling factor).

        Returns
        --------
        value : float
            Radial shift
        """
        return r + a + b*abs(cos(theta))
    if repel:
        coord = (coord
                    .assign(
                        r = lambda x : sqrt(x[f"Dim{axis[0]+1}"]**2+x[f"Dim{axis[1]+1}"]**2),
                        theta = lambda x : arctan2(x[f"Dim{axis[1]+1}"],x[f"Dim{axis[0]+1}"]),
                        rnew = lambda x : rshift(r=x["r"],theta=x["theta"],a=a,b=b),
                        xnew = lambda x : x["rnew"]*cos(x["theta"]),
                        ynew = lambda x : x["rnew"]*sin(x["theta"])
                        )
                    .drop(columns=["r","theta","rnew"]))
    else:
        coord = coord.assign(xnew = lambda x : x[f"Dim{axis[0]+1}"], ynew = lambda x : x[f"Dim{axis[1]+1}"])
    return coord 

# draw (add) circle to plot
def fviz_circle(p,
                r = 1.0,
                x0 = 0.0, 
                y0 = 0.0,
                color = "gray"):
    """
    Draw a circle

    Draw a circle with plotnine based on center and radius.

    Parameters
    ----------
    p : class
        A plotnine object.

    r : float, default = 1.0
        The radius.

    x0 : float, default = 0.0
        The `x` center.

    y0 : float, default = 0.0
        The `y` center.

    color : str, default = "gray"
        The color of the circle.

    Returns
    -------
    A plotnine object.

    Reference
    ---------
    [1] `Draw a circle with ggplot2 <https://stackoverflow.com/questions/6862742/draw-a-circle-with-ggplot2>`

    Examples
    --------
    >>> from plotnine import ggplot
    >>> from scientisttools import fviz_circle
    >>> p = fviz_circle(p=ggplot())
    >>> print(p.show())
    """
    # create data
    data = DataFrame({
        "x" : x0 + r*cos(linspace(0,pi,num=100)),
        "ymin" : y0 + r*sin(linspace(0,-pi,num=100)),
        "ymax" : y0 + r*sin(linspace(0,pi,num=100))
    })
    # create circle
    p = (
        p  
        + 
        geom_ribbon(
            data = data,
            mapping = aes(x="x",ymin="ymin",ymax="ymax"),
            inherit_aes = False,
            color = color,
            fill = None
        )
    )
    return p 

def fviz_arrow(obj,
               choice = "quanti_var",
               axis = [0,1],
               geom = ("arrow","text"),
               repel = False,
               color = "black",
               segment_args = dict(size=0.5,alpha=1),
               point_args = dict(size=1.5),
               text_args = dict(size=8),
               gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
               legend_title = None,
               palette = "Dark2",
               scale = 1,
               lim_cos2 = None,
               lim_contrib = None,
               circle = True,
               col_circle = "gray"):
    """
    Create an arrow plot with text
    
    Parameters
    ----------
    obj : class 
        an object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`, :class:`~scientisttools.MFA`

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("arrow","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point,"text"). 

        * "arrow" to show only arrows.
        * "point" to show only points
        * "text" to show only labels.
        * ("arrow","text") to show both arrows and texts.
        * ("point","text") to show both points and texts.
    
    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    color : str, 1darray, km class, list, tuple, Series, default = "black"
        Color for variables. Can be a continuous variable or a factor variable. 
        Possible values include also : "cos2", "contrib", "coord", "x" or "y". 
        In this case, the colors for variables are automatically controlled by their 
        qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). 

    segment_args : dict, default = dict(size = 0.5)
        A dictionary containing parameters (except color) for segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high correlation values.

    legend_title : str, defaut = None
        The title of the legend.

    palette : str, list, tuple, default = "Dark2"
        If string, the color palette to be used for coloring or filling by groups. If list or tuple, the colors for labels.

    scale : int, default = 1
        The scale of factor coordinates.

    lim_cos2 : float, default = None
        The cos2 limit.

    lim_contrib : float, default = None
        The relative contribution limit.

    circle : bool, default = True
        If True, draw a circle.

    col_circle : str, default = "gray"
        Color for the circle.
    
    Returns
    -------
    A plotnine object.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_arrow
    >>> # instanciation
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> # graph of variables
    >>> p = fviz_arrow(clf,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj is a valid class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("PCA","FAMD","PCAmix","MPCA","MFA","DMFA")):
        raise TypeError(f"{obj.__class__.__name__} class is not supported.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if geom is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=1)

    # remove point if arrow
    if (isinstance(geom,(list,tuple)) and  
        all(x in geom for x in ("arrow","point"))):
        geom = [x for x in geom if x != "point"]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # extract model attribute
    mattr = getattr(obj,f"{choice}_")
    # split : coordinates, square cosinus, contributions
    coord, cos2, contrib = mattr.coord.iloc[:,axis].mul(scale), mattr.cos2.iloc[:,axis], mattr.contrib.iloc[:,axis]
    
    # define text coordinates
    coord = overlap_coord(coord=coord,axis=axis,repel=repel)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # update coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    coord, legend_title, color_mapping = coord_adjust(
        coord=coord,
        cos2=cos2,
        contrib=contrib,
        color=color,
        legend_title=legend_title,
        habillage=None,
        palette=palette,
        lim_cos2=lim_cos2,
        lim_contrib=lim_contrib
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x, y for texts
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel:
        x_text, y_text = "xnew", "ynew"
    else:
        x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"
    
    # initialization
    p = ggplot(
        data = coord,
        mapping = aes(
            x = f"Dim{axis[0]+1}",
            y = f"Dim{axis[1]+1}",
            label = coord.index
        )
    )

    if ((isinstance(color,str) and 
         (color in [*["cos2","contrib","coord"],*coord.columns])) or 
         (isinstance(color,ndarray) and 
          (len(color) == coord.shape[0]) and 
          all(isinstance(x,(int,float)) for x in color))):
        # show segments
        if "arrow" in geom:
            p = (
                p 
                + geom_segment(
                    mapping = aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}",color=legend_title),
                    arrow = arrow(angle=30,length=0.2/2.54),
                    **segment_args
                )
            )
        # show points
        if "point" in geom:
            p = p + geom_point(aes(color=legend_title),**point_args)
        # scale color gradient2
        if any(x in geom for x in ("arrow","point")):
            p = (
                p 
                + scale_color_gradient2(
                    low = gradient_cols[0],
                    high = gradient_cols[2],
                    mid = gradient_cols[1]
                )
            )
        # show texts
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text,color=legend_title),**text_args)

    elif (hasattr(color,"labels_") or 
          (isinstance(color,(list,tuple,Series)) and 
           (len(color) == coord.shape[0]) and 
           all(isinstance(x,str) for x in color))):
        # show segments
        if "arrow" in geom:
            p = (
                p 
                + geom_segment(
                    mapping = aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}",color=legend_title),
                    arrow = arrow(angle=30,length=0.2/2.54),
                    **segment_args
                ) 
            )
        # show points
        if "point" in geom:
            p = p + geom_point(aes(color=legend_title),**point_args)
        # add legend title
        if any(x in geom for x in ("arrow","point")):
            p = p + guides(color=guide_legend(title=legend_title))
        # show texts
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text,color=legend_title),**text_args,show_legend = False)
    else:
        # show segments
        if "arrow" in geom:
            p = p + geom_segment(
                mapping = aes(x=0,y=0,xend=f"Dim{axis[0]+1}",yend=f"Dim{axis[1]+1}"),
                arrow = arrow(angle=30,length=0.2/2.54),
                color = color,
                **segment_args
            )
        # show points
        if "point" in geom:
            p = p + geom_point(color=color,**point_args)
        # show texts
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text),color=color,**text_args)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set color manual
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (hasattr(color,"labels_") or 
        (isinstance(color,(list,tuple,Series)) and 
         (len(color) == coord.shape[0]) and 
         all(isinstance(x,str) for x in color))):
        p = p + scale_color_manual(values=color_mapping)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show correlation circle
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if circle:
        p = fviz_circle(p=p,color=col_circle)
    
    return p

# add choices (point & text) to plotnine graph
def add_scatter(p,
                data,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                color = "blue",
                point_args = dict(shape="^",size=1.5),
                text_args = dict(size=8)):
    """
    Add points/texts to plotnine graph

    Parameters
    ----------
    p : class
        A plotnine object.

    data : DataFrame of shape (n_samples, n_components)
        Factor coordinates.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("point","text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("point","text").

        * "point" to show only points.
        * "text" to show only labels.
        * ("point","text") to show both points and labels.

    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    color : str, default = "blue"
        Color for the points and/or texts.

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `geom_text <https://plotnine.org/reference/geom_text.html>`).

    Returns
    -------
    A plotnine abject.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_scatter, add_scatter
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> # show active individuals points
    >>> p = fviz_scatter(clf,choice="ind",repel=True)
    >>> # show supplementary individuals points
    >>> p = add_scatter(p=p,data=clf.ind_sup_.coord,repel=True)
    >>> print(p)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if data is an object class pandas.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(data,DataFrame):
        raise TypeError(f"{type(data)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if axis is an instance of list
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if geom is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=0)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set text arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and ("text" in geom):
        text_args["adjust_text"] = dict(arrowprops=dict(lw=1.0))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "point" in geom:
        p = (
            p 
            + geom_point(
                data = data,
                mapping = aes(
                    x = f"Dim{axis[0]+1}", 
                    y = f"Dim{axis[1]+1}"
                ),
                inherit_aes = False,
                color = color,
                **point_args
            )
        )
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show texts
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "text" in geom:
        p = (
            p 
            + geom_text(
                data = data,
                mapping = aes(
                    x = f"Dim{axis[0]+1}",
                    y = f"Dim{axis[1]+1}",
                    label = data.index
                ),
                inherit_aes = False,
                color = color,
                **text_args
            )
        )
    return p

# add choices (arrow & text) to plotnine graph
def add_arrow(p,
              data,
              axis = [0,1],
              geom = ("arrow","text"),
              repel = False,
              color = "blue",
              segment_args = dict(linetype="dashed",size=0.5),
              point_args = dict(size=1.5),
              text_args = dict(size=8)):
    """
    Add arrows/points/texts to plotnine object

    Parameters
    ----------
    p : class
        A plotnine object.

    data : DataFrame of shape (n_samples, n_components)
        Factor coordinates.

    axis : list, default = [0,1]
        The dimensions to be plotted.

    geom : str, list, tuple, default = ("arrow","point,"text")
        The geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point,"text"). 

        * "arrow" to show only segments.
        * "points" to show only points.
        * "text" to show only labels.
        * ("arrow","text") to show both arrows and texts.
        * ("point","text") to show both points and texts.

    repel : bool, default = False
        Whether to avoid overplotting text labels or not.

    color : str, default = "blue"
        Color for the segments, points and/or texts.

    segment_args : dict, default = dict(linetype="dashed",size = 0.5)
        A dictionary containing parameters (except color) for segments (see `plotnine.geom_segment <https://plotnine.org/reference/geom_segment.html>`).

    point_args : dict, default = dict(size = 1.5)
        A dictionary containing parameters (except color) for points (see `plotnine.geom_point <https://plotnine.org/reference/geom_point.html>`).

    text_args : dict, default = dict(size = 8)
        A dictionary containing parameters (except color) for texts (see `plotnine.geom_text <https://plotnine.org/reference/geom_text.html>`).

    Returns
    -------
    A plotnine abject.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_arrow, add_arrow
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> # show active variables segments
    >>> p = fviz_arrow(clf,repel=True)
    >>> # show supplementary continuous variables segments
    >>> p = add_arrow(p=p,data=clf.quanti_var_sup_.coord,repel=True)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if data is an instance of class pandas.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(data,DataFrame):
        raise TypeError(f"{type(data)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if axis is an instance of list
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if geom is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom=geom,axis=1)

    # remove point if arrow
    if (isinstance(geom,(list,tuple)) and  
        all(x in geom for x in ("arrow","point"))):
        geom = [x for x in geom if x != "point"]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # define text coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    data = overlap_coord(coord=data,axis=axis,repel=repel)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x, y for geom_text
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and ("text" in geom):
        x_text, y_text = "xnew", "ynew"
    else:
        x_text, y_text = f"Dim{axis[0]+1}", f"Dim{axis[1]+1}"
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show segments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "arrow" in geom:
        p = (
            p 
            + geom_segment(
                data = data,
                mapping = aes(
                    x = 0,
                    y = 0,
                    xend = f"Dim{axis[0]+1}",
                    yend = f"Dim{axis[1]+1}"
                ),
                arrow = arrow(angle=30,length=0.2/2.54),
                inherit_aes = False,
                color = color,
                **segment_args
            )
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "point" in geom:
        p = (
            p 
            + geom_point(
                data = data,
                mapping = aes(
                    x = f"Dim{axis[0]+1}", 
                    y = f"Dim{axis[1]+1}"
                ),
                inherit_aes = False,
                color = color,
                **point_args
            )
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show texts
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "text" in geom:
        p = (
            p 
            + geom_text(
                data = data,
                mapping=aes(
                    x = x_text,
                    y = y_text,
                    label = data.index
                ),
                inherit_aes = False,
                color = color,
                **text_args
            )
        )
    return p

def set_axis(p,
             obj,
             axis = [0,1],
             x_lim = None,
             y_lim = None,
             x_label = None,
             y_label = None,
             title = None,
             subtitle = None,
             pntheme = theme_minimal(),
             **kwargs):
    """
    Add elements to plotnine object

    Parameters
    ----------
    p : class
        A plotnine object.

    obj : class
        An instance of class 

    axis : list, default = [0,1]
        The dimensions to be plotted.

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
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj has eig_ attribute
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not hasattr(obj, "eig_"):
        raise ValueError(f"{obj.__class__.__name__} class is not supported.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj=obj,axis=axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x label, y label and title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x label
    if x_label is None:
        x_label = f"Dim{axis[0]+1} ({round(obj.eig_.iloc[axis[0],2],0)}%)"
    # set y label
    if y_label is None:
        y_label = f"Dim{axis[1]+1} ({round(obj.eig_.iloc[axis[1],2],0)}%)"
    # set title
    if title is None:
        title = f"{obj.__class__.__name__} - Factor map"
    # set subtitle
    if subtitle is None:
        subtitle = ""
    # show labs
    p = p + labs(
        x = x_label,
        y = y_label,
        title = title,
        subtitle = subtitle
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x range and y range
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x limits
    if x_lim is not None:
        p = p + xlim(x_lim)
    # set y limits
    if y_lim is not None:
        p = p + ylim(y_lim)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # show horizontal and vertical lines
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = (
        p 
        + geom_hline(yintercept=0,linetype="dashed")
        + geom_vline(xintercept=0,linetype="dashed")
    )

    # add plotnine theme
    p = p + pntheme
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # customize theme
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if kwargs is not None:
        p = p + theme(**kwargs)
    return p