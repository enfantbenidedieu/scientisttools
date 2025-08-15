# -*- coding: utf-8 -*-
from plotnine import theme_minimal

#intern functions
from .fviz import fviz_scatter, add_scatter, set_axis

def fviz_ca_row(self,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                lim_cos2 = None,
                lim_contrib = None,
                col_row = "black",
                point_args_row = dict(size=1.5),
                text_args_row = dict(size=8),
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                habillage = None,
                palette = None,
                add_ellipses = False,  
                ellipse_level = 0.95,
                ellipse_type = "norm",
                ellipse_alpha = 0.1,
                row_sup = False,
                col_row_sup = "blue",
                point_args_row_sup = dict(shape="^",size=1.5),
                text_args_row_sup = dict(size=8),
                quali_sup = False,
                col_quali_sup = "red",
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
    Visualize Correspondence Analysis (CA) - Graph of row variables
    ---------------------------------------------------------------

    Description
    -----------
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. `fviz_ca_row` provides plotnine based elegant visualization of CA outputs from Python functions.

    Usage
    -----
    ```python
    >>> fviz_ca_row(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class CA

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
    >>> from scientisttools import children, CA, fviz_ca_row
    >>> res_ca = CA(n_components=None,row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children)
    >>> #graph of row variables
    >>> p = fviz_ca_row(res_ca, repel=True)
    >>> print(p)
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    #active row points
    p = fviz_scatter(self,
                     element = "row",
                     axis = axis,
                     geom = geom,
                     repel = repel,
                     lim_cos2 = lim_cos2,
                     lim_contrib = lim_contrib,
                     color = col_row,
                     point_args = point_args_row,
                     text_args = text_args_row,
                     gradient_cols = gradient_cols,
                     legend_title = legend_title,
                     habillage = habillage,
                     palette = palette,
                     add_ellipses = add_ellipses, 
                     ellipse_level = ellipse_level,
                     ellipse_type = ellipse_type,
                     ellipse_alpha = ellipse_alpha)
    
    #add supplementary rows
    if row_sup:
        if hasattr(self,"row_sup_"):
            p = add_scatter(p=p,data=self.row_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_row_sup,points_args=point_args_row_sup,text_args=text_args_row_sup)
        
    #add supplementary categories
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

    #add others elements
    if title is None:
        title = "CA graph of rows"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_ca_col(self,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                lim_cos2 = None,
                lim_contrib = None,
                col_col = "black",
                point_args_col = dict(size=1.5),
                text_args_col = dict(size=8),
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                palette = None,
                col_sup = True,
                col_col_sup = "blue",
                point_args_col_sup = dict(shape=">",size=1.5),
                text_args_col_sup = dict(size=8),
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
    Visualize Correspondence Analysis (CA) - Graph of column variables
    ------------------------------------------------------------------

    Description
    -----------
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. `fviz_ca_col` provides plotnine based elegant visualization of CA outputs from Python functions.

    Usage
    -----
    ```python
    >>> fviz_ca_col(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `**kwargs`: additionals parameters (see `fviz_scatter`).

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import children, CA, fviz_ca_col
    >>> res_ca = CA(n_components=None,row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Columns factor map
    >>> p = fviz_ca_col(res_ca,repel=True)
    >>> print(p)
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    #active individuals
    p = fviz_scatter(self,
                     element = "col",
                     axis = axis,
                     geom = geom,
                     repel = repel,
                     lim_cos2 = lim_cos2,
                     lim_contrib = lim_contrib,
                     color = col_col,
                     point_args = point_args_col,
                     text_args = text_args_col,
                     gradient_cols = gradient_cols,
                     legend_title = legend_title,
                     palette = palette)
    
    #add supplementary columns points
    if col_sup:
        if hasattr(self,"col_sup_"):
            p = add_scatter(p=p,data=self.col_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_col_sup,points_args=point_args_col_sup,text_args=text_args_col_sup)

    #add others elements
    if title is None:
        title = "CA graph of columns"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_ca_biplot(self,
                   axis = [0,1],
                   geom_row = ("point","text"),
                   geom_col = ("point","text"),
                   repel_row = True,
                   repel_col = True,
                   col_row = "black",
                   point_args_row = dict(size=1.5),
                   text_args_row = dict(size=8),
                   col_col = "steelblue",
                   point_args_col = dict(size=1.5),
                   text_args_col = dict(size=8),
                   gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                   legend_title = None,
                   habillage = None,
                   palette = None,
                   add_ellipses = False,  
                   ellipse_level = 0.95,
                   ellipse_type = "norm",
                   ellipse_alpha = 0.1,
                   row_sup = True,
                   col_row_sup = "red",
                   point_args_row_sup = dict(shape="^",size=1.5),
                   text_args_row_sup = dict(size=8),
                   quali_sup = True,
                   col_quali_sup = "darkred",
                   point_args_quali_sup = dict(shape="v",size=1.5),
                   text_args_quali_sup = dict(size=8),
                   col_sup = True,
                   col_col_sup = "darkblue",
                   point_args_col_sup = dict(shape="x",size=1.5),
                   text_args_col_sup = dict(size=8),
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
    Visualize Correspondence Analysis (CA) - Biplot of row and columns variables
    ----------------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_ca_biplot(self,**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class CA

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
    >>> from scientisttools import children, CA, fviz_ca_biplot
    >>> res_ca = CA(n_components=None,row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Biplot
    >>> p = fviz_ca_biplot(res_ca,repel_row=True,repel_col=True)
    >>> print(p)
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise ValueError("'self' must be an object of class CA")
    
    #active row points
    p = fviz_scatter(self,
                     element = "row",
                     axis = axis,
                     geom = geom_row,
                     repel = repel_row,
                     color = col_row,
                     point_args = point_args_row,
                     text_args = text_args_row,
                     gradient_cols = gradient_cols,
                     legend_title = legend_title,
                     habillage = habillage,
                     palette = palette,
                     add_ellipses = add_ellipses, 
                     ellipse_level = ellipse_level,
                     ellipse_type = ellipse_type,
                     ellipse_alpha = ellipse_alpha)
    
    #add supplementary rows
    if row_sup:
        if hasattr(self,"row_sup_"):
            p = add_scatter(p=p,data=self.row_sup_.coord,axis=axis,geom=geom_row,repel=repel_row,color=col_row_sup,points_args=point_args_row_sup,text_args=text_args_row_sup)
        
    #add supplementary categories
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom_row,repel=repel_row,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

    #add active columns points
    p = add_scatter(p=p,data=self.col_.coord,axis=axis,geom=geom_col,repel=repel_col,color=col_col,points_args=point_args_col,text_args=text_args_col)

    #add supplementary columns points
    if col_sup:
        if hasattr(self,"col_sup_"):
            p = add_scatter(p=p,data=self.col_sup_.coord,axis=axis,geom=geom_col,repel=repel_col,color=col_col_sup,points_args=point_args_col_sup,text_args=text_args_col_sup)

    #add others elements
    if title is None:
        title = "CA graph of rows and columns"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p
    
def fviz_ca(self, element="biplot",**kwargs):
    """
    Draw the Correspondence Analysis (CA) graphs
    --------------------------------------------

    Description
    -----------
    Draw the Correspondence Analysis (CA) graphs.

    Usage
    -----
    ```python
    >>> fviz_ca(self, element = "row", **kwargs)
    >>> fviz_ca(self, element = "col", **kwargs)
    >>> fviz_ca(self, element = "biplot", **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `element`: the graph to plot. Allowed values are:
        * 'row' for the row points factor map
        * 'col' for the columns points factor map
        * 'biplot' for biplot of row and columns factor map

    `**kwargs`: further arguments passed to or from other methods

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import children, CA, fviz_ca
    >>> res_ca = CA(n_components=None,row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children)
    >>> #rows factor map
    >>> p1 = fviz_ca(res_ca, element = "row")
    >>> print(p1)
    >>> #columns factor map
    >>> p2 = fviz_ca(res_ca, element = "col")
    >>> print(p2)
    >>> #biplot
    >>> p3 = fviz_ca(res_ca, element = "biplot")
    >>> print(p3)
    ```
    """
    if element == "row":
        return fviz_ca_row(self,**kwargs)
    elif element == "col":
        return fviz_ca_col(self,**kwargs)
    elif element == "biplot":
        return fviz_ca_biplot(self,**kwargs)
    else:
        raise ValueError("'element' should be one of 'row', 'col', 'biplot'")