# -*- coding: utf-8 -*-
from plotnine import ggplot, theme_minimal

#intern functions
from .fviz import fviz_scatter, add_scatter, set_axis

def fviz_mca_ind(self,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 col_ind ="black",
                 point_args_ind = dict(size=1.5),
                 text_args_ind = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 habillage = None,
                 palette = None,
                 add_ellipses = False,  
                 ellipse_level = 0.95,
                 ellipse_type = "norm",
                 ellipse_alpha = 0.1,
                 ind_sup = False,
                 col_ind_sup = "blue",
                 point_args_ind_sup = dict(shape = "^",size=1.5),
                 text_args_ind_sup = dict(size=8),
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 add_hline = True,
                 add_vline=True,
                 add_grid = True,
                 ggtheme = theme_minimal()):
    
    """
    Visualize Multiple Correspondence Analysis (MCA) - Graph of individuals
    -----------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_ind() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_mca_ind(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `**kwargs`: an additional informations (see `fviz_scatter`).

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import poison, MCA, fviz_mca_ind
    >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
    >>> res_mca.fit(poison)
    >>> #graph of individuals
    >>> p = fviz_mca_ind(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise ValueError("'self' must be an object of class MCA")
    
    #active individuals
    p = fviz_scatter(self,
                     element = "ind",
                     axis = axis,
                     geom = geom,
                     repel = repel,
                     lim_cos2 = lim_cos2,
                     lim_contrib = lim_contrib,
                     color = col_ind,
                     point_args = point_args_ind,
                     text_args = text_args_ind,
                     gradient_cols = gradient_cols,
                     legend_title = legend_title,
                     habillage = habillage,
                     palette = palette,
                     add_ellipses = add_ellipses, 
                     ellipse_level = ellipse_level,
                     ellipse_type = ellipse_type,
                     ellipse_alpha = ellipse_alpha)
    
    #add supplementary individuals
    if ind_sup:
        if hasattr(self,"ind_sup_"):
            p = add_scatter(p=p,data=self.ind_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_ind_sup,points_args=point_args_ind_sup,text_args=text_args_ind_sup)
    
    #add others elements
    if title is None:
        title = "Individuals - MCA"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

# Graph for categories
def fviz_mca_var(self,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 col_var ="black",
                 point_args_var = dict(size=1.5),
                 text_args_var = dict(size = 8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = None,
                 quali_sup = False,
                 col_quali_sup = "blue",
                 point_args_quali_sup = dict(size = 1.5),
                 text_args_quali_sup = dict(size = 8),
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 add_hline = True,
                 add_vline = True,
                 add_grid =True,
                 ggtheme = theme_minimal()):
    
    """
    Visualize Multiple Correspondence Analysis (MCA) - Graph of categories
    ----------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_mod() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for categories.

    Usage
    -----
    ```python
    >>> fviz_mca_var(self, **kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class MCA or SpecificMCA

    `**kwargs`: additionals parameters (see `fviz_scatter`).

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import poison MCA, fviz_mca_var
    >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
    >>> res_mca.fit(poison)
    >>> #graph of categories
    >>> p = fviz_mca_var(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    
    #active individuals
    p = fviz_scatter(self,
                     element = "var",
                     axis = axis,
                     geom = geom,
                     repel = repel,
                     lim_cos2 = lim_cos2,
                     lim_contrib = lim_contrib,
                     color = col_var,
                     point_args = point_args_var,
                     text_args = text_args_var,
                     gradient_cols = gradient_cols,
                     legend_title = legend_title,
                     palette = palette)
    
    #add supplementary variables
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)
    
    #add others elements
    if title is None:
        title = "Variable categories - MCA"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_mca_quali_var(self,
                       axis = [0,1],
                       geom = ("point","text"),
                       repel = False,
                       col_quali_var = "black",
                       point_args_quali_var = dict(size=1.5),
                       text_args_quali_var = dict(size=8),
                       quali_sup = False,
                       col_quali_sup = "blue",
                       point_args_quali_sup = dict(shape="^",size=1.5),
                       text_args_quali_sup = dict(size=8),
                       quanti_sup = False,
                       col_quanti_sup = "red",
                       point_args_quanti_sup = dict(shape="^",size=1.5),
                       text_args_quanti_sup = dict(size=8),
                       x_lim = (0,1.1),
                       y_lim = (0,1.1),
                       x_label = None,
                       y_label = None,
                       title = None,
                       add_grid =True,
                       add_hline = True,
                       add_vline = True, 
                       ggtheme = theme_minimal()):
    """
    Visualize Multiple Correspondence Analysis (MCA) - Graph of variables
    ---------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_var() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for variables.

    Usage
    -----
    ```python
    >>> fviz_mca_quali_var(self, **kwargs) 
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `**kwargs`: additionals parameters (see `add_scatter`)

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import poison, MCA, fviz_mca_var
    >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
    >>> res_mca.fit(poison)
    >>> #graph of variables
    >>> p = fviz_mca_quali_var(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    p = add_scatter(p=ggplot(),data=self.var_.eta2,axis=axis,geom=geom,repel=repel,color=col_quali_var,points_args=point_args_quali_var,text_args=text_args_quali_var)

    #add supplementary variables
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.eta2,axis=axis,geom=geom,repel=repel,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

    #add supplementary variables
    if quanti_sup:
        if hasattr(self,"quanti_sup_"):
            p = add_scatter(p=p,data=self.quanti_sup_.cos2,axis=axis,geom=geom,repel=repel,color=col_quanti_sup,points_args=point_args_quanti_sup,text_args=text_args_quanti_sup)
    
    #add others elements
    if title is None:
        title = "Variables - MCA"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_mca_biplot(self,
                    axis=[0,1],
                    geom_ind = ("point","text"),
                    geom_var = ("point","text"),
                    repel_ind = False,
                    repel_var = False,
                    col_ind = "black",
                    point_args_ind = dict(size=1.5),
                    text_args_ind = dict(size=8),
                    col_var = "steelblue",
                    point_args_var = dict(shape="^",size=1.5),
                    text_args_var = dict(size=8),
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    legend_title = None,
                    habillage = None,
                    palette = None,
                    add_ellipses = False, 
                    ellipse_level = 0.95,
                    ellipse_type = "norm",
                    ellipse_alpha = 0.1,
                    ind_sup = False,
                    col_ind_sup = "red",
                    point_args_ind_sup = dict(shape="^",size=1.5),
                    text_args_ind_sup = dict(size=8),
                    quali_sup = False,
                    col_quali_sup = "blue",
                    point_args_quali_sup = dict(shape=">",size=1.5),
                    text_args_quali_sup = dict(size=8),
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = None,
                    add_hline = True,
                    add_vline = True,
                    add_grid = True,
                    ggtheme = theme_minimal()):
    """
    Visualize Multiple Correspondence Analysis (MCA) - Biplot of individuals and categories
    ---------------------------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_biplot() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for individuals and categories.

    Usage
    -----
    ```python
    >>> fviz_mca_biplot(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `**kwargs`: additionals parameters (see `fviz_scatter`)    

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import poison, MCA, fviz_mca_biplot
    >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
    >>> res_mca.fit(poison)
    >>> #biplot of individuals and categories
    >>> p = fviz_mca_biplot(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    
    #active individuals
    p = fviz_scatter(self,
                     element = "ind",
                     axis = axis,
                     geom = geom_ind,
                     repel = repel_ind,
                     color = col_ind,
                     point_args = point_args_ind,
                     text_args = text_args_ind,
                     gradient_cols = gradient_cols,
                     legend_title = legend_title,
                     habillage = habillage,
                     palette = palette,
                     add_ellipses = add_ellipses, 
                     ellipse_level = ellipse_level,
                     ellipse_type = ellipse_type,
                     ellipse_alpha = ellipse_alpha)
    
    #add supplementary individuals
    if ind_sup:
        if hasattr(self,"ind_sup_"):
            p = add_scatter(p=p,data=self.ind_sup_.coord,axis=axis,geom=geom_ind,repel=repel_ind,color=col_ind_sup,points_args=point_args_ind_sup,text_args=text_args_ind_sup)
    
    #add active categories
    p = add_scatter(p=p,data=self.var_.coord,axis=axis,geom=geom_var,repel=repel_var,color=col_var,points_args=point_args_var,text_args=text_args_var)

    #add supplementary categories
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom_ind,repel=repel_ind,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

    #set title
    if title is None:
        title = "MCA - Biplot of individuals and categories"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)
    
    return p
    
def fviz_mca(self, element = "biplot",**kwargs):
    """
    Visualize Multiple Correspondence Analysis (MCA)
    ------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca() provides plotnine-based elegant visualization of MCA/SpecificMCA outputs.

        * `fviz_mca_ind()`: Graph of individuals
        * `fviz_mca_var()`: Graph of categories
        * `fviz_mca_quali_var()`: Graph of variables
        * `fviz_mca_biplot()`: Biplot of individuals and categories
    
    Usage
    -----
    ```python
    >>> fviz_mca(self, element = "ind", **kwargs)
    >>> fviz_mca(self, element = "var", **kwargs)
    >>> fviz_mca(self, element = "quali_var", **kwargs)
    >>> fviz_mca(self, element = "biplot", **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `element`: The element to subset. Possible values are:
        * "ind" for the individuals graphs
        * "var" for the categories graphs
        * "quali_var" for the variables graphs
        * "biplot" for both individuals and categories graphs

    `**kwargs`: further arguments passed to or from other methods

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """    
    if element == "ind":
        return fviz_mca_ind(self,**kwargs)
    elif element == "var":
        return fviz_mca_var(self,**kwargs)
    elif element == "quali_var":
        return fviz_mca_var(self,**kwargs)
    elif element == "biplot":
        return fviz_mca_biplot(self,**kwargs)
    else:
        raise ValueError("'choice' should be one of 'ind', 'var', 'quali_var','biplot'")