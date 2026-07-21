# -*- coding: utf-8 -*-
from pandas import Categorical
from plotnine import (
    ggplot, 
    aes,
    geom_bar, 
    theme, 
    theme_minimal, 
    labs, 
    element_text
)

def fviz_cos2(obj,
              choice = "ind",
              axis = None,
              y_label = None,
              top = None,
              col_bar = "steelblue",
              bar_args = dict(fill="steelblue",width=0.8),
              sort = "desc",
              angle = 45,
              pntheme = theme_minimal(),
              **kwargs):
    
    """
    Visualize the quality of representation of row/columns elements
    
    This function can be used to visualize the quality of representation of rows/columns from the results of Principal Component Analysis (:class:`~scientisttools.PCA`), 
    Correspondence Analysis (:class:`~scientistools.CA`), Multiple Correspondence Analysis (:class:`~scientisttools.MCA`), Factor Analysis of Mixed Data (:class:`~scientisttools.FAMD`), 
    Principal Component Analysis of Mixed Data (:class:`~scientisttools.PCAmix`), Mixed Principal Component Analysis (:class:`~scientisttools.MPCA`), and Multiple Factor Analysis (:class:`~scientisttools.MFA`) functions. 
        
    Parameters
    ----------
    obj : class
        An object of class :class:`~ scientisttools.PCA`, :class:`~ scientisttools.CA`, :class:`~ scientisttools.MCA`, :class:`~ scientisttools.FAMD`, :class:`~ scientisttools.PCAmix`, :class:`~ scientisttools.MPCA`, :class:`~ scientisttools.MFA`

    choice : {"row","col","ind","quanti_var","levels","quali_var","freq","group","partial_axes"}, default = "ind"
        The element to subset. Allowed values are :

        * "row" for rows
        * "col" for columns
        * 'ind' for individuals
        * "quanti_var" for continuous variables
        * "levels" for variable categories
        * 'quali_var' for categorical variables
        * 'freq' for frequencies
        * 'group' for groupes
        
    axis : int, default = None. 
        Select the axis for which the row/col cos2 are plotted. If None, axis = 0.
        
    y_label : str, default = None.
        The label text of y. If None, then a y_label is chosen.
        
    top : int, default = None
        The number of top elements to be shown. If top is None, then all labels are plotted.

    col_bar : str, default = "steelblue"
        Outline color for the bar plot.

    bar_args : dict, default = dict(fill="steelblue",width=0.8)
        A dictionary containing parameters (except color) for bar plot (see `plotnine.geom_bar <https://plotnine.org/reference/geom_bar.html>`).
            
    sort : {None, "asc","desc"}, default = "desc"
        None or a string specifying whether the value should be sorted. Allowed values are:

        * None for no sorting
        * "asc" for ascending
        * "desc" for descending

    angle : int, default = 45
        Rotation angle for x axis tick labels. Default is 45 degrees.
    
    grid : bool, default = True
        A boolean to either add or not a grid customization.

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).
    
    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.
    
    Returns
    -------
    A plotnine object.

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, fviz_cos2
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison.data)
    >>> # cos2 of individuals
    >>> p = fviz_cos2(clf,choice="ind",axis=0)
    >>> print(p.show())
    >>> # cos2 of variable categories
    >>> p = fviz_cos2(clf,choice="levels",axis=0)
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("PCA","CA","MCA","FAMD","PCAmix","MPCA","MFA","DMFA")):
        raise TypeError("'obj' must be an object of class PCA, CA, MCA, FAMD, PCAmix, MPCA, MFA, DMFA")    
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid element
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (choice in ("row","col","ind","var","quanti_var","levels","quali_var","freq","group")):
        raise ValueError("'element' should be one of 'row', 'col', 'ind', 'var', 'quanti_var', levels, 'quali_var', 'freq', 'group'.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ncp = obj.call_.ncp
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise TypeError("'axis' must be an integer.")
    elif axis not in list(range(0,ncp)):
        raise TypeError(f"'axis' must be an integer between 0 and {ncp-1}.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # extract cos2
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    data = getattr(obj,f"{choice}_",False).cos2.iloc[:,axis].to_frame().reset_index()
    data.columns = ["name","cos2"]
    data["name"] = Categorical(data["name"],categories=data["name"],ordered=True)

    if top is not None:
        data = data.sort_values(by="cos2",ascending=False).head(top)
            
    # set name
    def getname(elt):
        match elt:
            case "freq":
                return "frequencies"
            case "ind":
                return "individuals"
            case "var":
                return "variables"
            case "levels":
                return "variable categories"
            case "quali_var":
                return "categorical variables"
            case "quanti_var":
                return "continuous variables"
            case "row":
                return "rows"
            case "col":
                return "columns"
            case "group":
                return "groups"
    
    # reorder label by cos2
    def rcos2(sort):
        if sort == "desc":
            return "reorder(name,-cos2)"
        elif sort == "asc":
            return"reorder(name,cos2)"
        elif sort is None:
            return "name"
        else:
            raise ValueError("'sort' must be one of 'desc','asc', None")

    # initialization 
    p = (
         ggplot(data=data,mapping=aes(x=rcos2(sort),y="cos2",group = 1)) 
         + geom_bar(stat="identity",color=col_bar,**bar_args)
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set labs
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if y_label is None:
        y_label = "Cos2 - Quality of representation"
    p = (
        p + 
        labs(
            title=f"Cos2 of {getname(elt=choice)} to Dim-{axis+1}",
            y=y_label,
            x=""
        )
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add plotnine theme
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = p + pntheme

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x tick label angle
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if angle > 5:
        ha = "right"
    if angle == 90:
        ha = "center"
    p = p + theme(axis_text_x=element_text(rotation=angle,ha=ha),**kwargs)
    return p 