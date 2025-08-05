# -*- coding: utf-8 -*-
from .fviz_add import fviz_point, fviz_add_point

def fviz_ca_row(self,
                axis = [0,1],
                geom = ["point","text"],
                repel = False,
                lim_cos2 = None,
                lim_contrib = None,
                x_lim = None,
                y_lim = None,
                x_label = None,
                y_label = None,
                title = None,
                point_alpha_row = 1,
                point_col_row = "black",
                fill_row = None,
                shape_row = "o",
                point_size_row = 1.5,
                stroke_row = 0.5,
                text_type_row = "text",
                text_alpha_row = 1,
                angle_row = 0,
                text_col_row = "black",
                family_row = None,
                fontstyle_row = "normal",
                fontweight_row = "normal",
                ha_row = "center",
                lineheight_row = 1.2,
                text_size_row = 8,            
                va_row = "center",
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                habillage = None,
                palette = None,
                add_ellipses = False, 
                geom_ellipse = "polygon",
                type = "t",
                level = 0.95,
                alpha = 0.25,
                row_sup=True,
                point_alpha_row_sup = 1,
                point_col_row_sup = "steelblue",
                fill_row_sup = None,
                shape_row_sup = "^",
                point_size_row_sup = 1.5,
                stroke_row_sup = 0.5,
                text_type_row_sup = "text",
                text_alpha_row_sup = 1,
                angle_row_sup = 0,
                text_col_row_sup = "steelblue",
                family_row_sup = None,
                fontstyle_row_sup = "normal",
                fontweight_row_sup = "normal",
                ha_row_sup = "center",
                lineheight_row_sup = 1.2,
                text_size_row_sup = 8,            
                va_row_sup = "center",
                quali_sup = True,
                point_alpha_quali_sup = 1,
                point_col_quali_sup = "red",
                fill_quali_sup = None,
                shape_quali_sup = ">",
                point_size_quali_sup = 1.5,
                stroke_quali_sup = 0.5,
                text_type_quali_sup = "text",
                text_alpha_quali_sup = 1,
                angle_quali_sup = 0,
                text_col_quali_sup = "red",
                family_quali_sup = None,
                fontstyle_quali_sup = "normal",
                fontweight_quali_sup = "normal",
                ha_quali_sup = "center",
                lineheight_quali_sup = 1.2,
                text_size_quali_sup = 8,
                va_quali_sup = "center",
                add_grid = True,
                add_hline = True,
                alpha_hline = 0.5,
                col_hline = "black",
                size_hline = 0.5,
                linetype_hline="dashed",
                add_vline = True,
                alpha_vline = 0.5,
                col_vline = "black",
                size_vline = 0.5,
                linetype_vline ="dashed",
                ggtheme = None):
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

    `**kwargs`: additionals parameters. For more see `fviz_point`.
    
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
    #set title if None
    if title is None:
        title = "Row points - CA"
    
    p = fviz_point(self,
                    element = "row",
                    axis = axis,
                    geom = geom,
                    repel = repel,
                    lim_cos2 = lim_cos2,
                    lim_contrib = lim_contrib,
                    x_lim = x_lim,
                    y_lim = y_lim,
                    x_label = x_label,
                    y_label = y_label,
                    title = title,
                    point_alpha = point_alpha_row,
                    point_col = point_col_row,
                    fill = fill_row,
                    shape = shape_row,
                    point_size = point_size_row,
                    stroke = stroke_row,
                    text_type = text_type_row,
                    text_alpha = text_alpha_row,
                    angle =  angle_row,
                    text_col = text_col_row,
                    family = family_row,
                    fontstyle = fontstyle_row,
                    fontweight = fontweight_row,
                    ha = ha_row,
                    lineheight = lineheight_row,
                    text_size = text_size_row,            
                    va = va_row,
                    gradient_cols = gradient_cols,
                    legend_title = legend_title,
                    habillage = habillage,
                    palette = palette,
                    add_ellipses = add_ellipses, 
                    geom_ellipse = geom_ellipse,
                    type = type,
                    level = level,
                    alpha = alpha,
                    sup = row_sup,
                    name_sup = "row_sup_",
                    point_alpha_sup = point_alpha_row_sup,
                    point_col_sup = point_col_row_sup,
                    fill_sup = fill_row_sup,
                    shape_sup = shape_row_sup,
                    point_size_sup = point_size_row_sup,
                    stroke_sup = stroke_row_sup,
                    text_type_sup = text_type_row_sup,
                    text_alpha_sup = text_alpha_row_sup,
                    angle_sup = angle_row_sup,
                    text_col_sup = text_col_row_sup,
                    family_sup = family_row_sup,
                    fontstyle_sup = fontstyle_row_sup,
                    fontweight_sup = fontweight_row_sup,
                    ha_sup = ha_row_sup,
                    lineheight_sup = lineheight_row_sup,
                    text_size_sup = text_size_row_sup,
                    va_sup = va_row_sup,
                    quali_sup = quali_sup,
                    name_quali_sup = "quali_sup_",
                    point_alpha_quali_sup = point_alpha_quali_sup,
                    point_col_quali_sup = point_col_quali_sup,
                    fill_quali_sup = fill_quali_sup,
                    shape_quali_sup = shape_quali_sup,
                    point_size_quali_sup = point_size_quali_sup,
                    stroke_quali_sup = stroke_quali_sup,
                    text_type_quali_sup = text_type_quali_sup,
                    text_alpha_quali_sup = text_alpha_quali_sup,
                    angle_quali_sup = angle_quali_sup,
                    text_col_quali_sup = text_col_quali_sup,
                    family_quali_sup = family_quali_sup,
                    fontstyle_quali_sup = fontstyle_quali_sup,
                    fontweight_quali_sup = fontweight_quali_sup,
                    ha_quali_sup = ha_quali_sup,
                    lineheight_quali_sup = lineheight_quali_sup,
                    text_size_quali_sup = text_size_quali_sup,
                    va_quali_sup = va_quali_sup,
                    add_grid = add_grid,
                    add_hline = add_hline,
                    alpha_hline =alpha_hline,
                    col_hline = col_hline,
                    size_hline = size_hline,
                    linetype_hline = linetype_hline,
                    add_vline = add_vline,
                    alpha_vline = alpha_vline,
                    col_vline = col_vline,
                    size_vline = size_vline,
                    linetype_vline = linetype_vline,
                    ggtheme = ggtheme)

    return p

def fviz_ca_col(self,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                lim_cos2 = None,
                lim_contrib = None,
                x_lim = None,
                y_lim = None,
                x_label = None,
                y_label = None,
                title = None,
                point_alpha_col = 1,
                point_col_col = "black",
                fill_col = None,
                shape_col = "o",
                point_size_col = 1.5,
                stroke_col = 0.5,
                text_type_col = "text",
                text_alpha_col = 1,
                angle_col = 0,
                text_col_col = "black",
                family_col = None,
                fontstyle_col = "normal",
                fontweight_col = "normal",
                ha_col = "center",
                lineheight_col = 1.2,
                text_size_col = 8,
                va_col = "center",
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                palette = None,
                col_sup = True,
                point_alpha_col_sup = 1,
                point_col_col_sup = "steelblue",
                fill_col_sup = None,
                shape_col_sup = "^",
                point_size_col_sup = 1.5,
                stroke_col_sup = 0.5,
                text_type_col_sup = "text",
                text_alpha_col_sup = 1,
                angle_col_sup = 0,
                text_col_col_sup = "steelblue",
                family_col_sup = None,
                fontstyle_col_sup = "normal",
                fontweight_col_sup = "normal",
                ha_col_sup = "center",
                lineheight_col_sup = 1.2,
                text_size_col_sup = 8,
                va_col_sup = "center",
                add_grid = True,
                add_hline = True,
                alpha_hline = 0.5,
                col_hline = "black",
                linetype_hline="dashed",
                size_hline = 0.5,
                add_vline = True,
                alpha_vline = 0.5,
                col_vline = "black",
                linetype_vline ="dashed",
                size_vline = 0.5,
                ggtheme = None):
    
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

    `**kwargs`: additionals parameters. For more see `fviz_point`

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
    #set title if None
    if title is None:
        title = "Columns points - CA"
    
    p = fviz_point(self,
                    element = "col",
                    axis = axis,
                    geom = geom,
                    repel = repel,
                    lim_cos2 = lim_cos2,
                    lim_contrib = lim_contrib,
                    x_lim = x_lim,
                    y_lim = y_lim,
                    x_label = x_label,
                    y_label = y_label,
                    title = title,
                    point_alpha = point_alpha_col,
                    point_col = point_col_col,
                    fill = fill_col,
                    shape = shape_col,
                    point_size = point_size_col,
                    stroke = stroke_col,
                    text_type = text_type_col,
                    text_alpha = text_alpha_col,
                    angle =  angle_col,
                    text_col = text_col_col,
                    family = family_col,
                    fontstyle = fontstyle_col,
                    fontweight = fontweight_col,
                    ha = ha_col,
                    lineheight = lineheight_col,
                    text_size = text_size_col,            
                    va = va_col,
                    gradient_cols = gradient_cols,
                    legend_title = legend_title,
                    habillage = None,
                    palette = palette,
                    sup = col_sup,
                    name_sup = "col_sup_",
                    point_alpha_sup = point_alpha_col_sup,
                    point_col_sup = point_col_col_sup,
                    fill_sup = fill_col_sup,
                    shape_sup = shape_col_sup,
                    point_size_sup = point_size_col_sup,
                    stroke_sup = stroke_col_sup,
                    text_type_sup = text_type_col_sup,
                    text_alpha_sup = text_alpha_col_sup,
                    angle_sup = angle_col_sup,
                    text_col_sup = text_col_col_sup,
                    family_sup = family_col_sup,
                    fontstyle_sup = fontstyle_col_sup,
                    fontweight_sup = fontweight_col_sup,
                    ha_sup = ha_col_sup,
                    lineheight_sup = lineheight_col_sup,
                    text_size_sup = text_size_col_sup,
                    va_sup = va_col_sup,
                    quali_sup = False,
                    add_grid = add_grid,
                    add_hline = add_hline,
                    alpha_hline = alpha_hline,
                    col_hline = col_hline,
                    size_hline = size_hline,
                    linetype_hline = linetype_hline,
                    add_vline = add_vline,
                    alpha_vline = alpha_vline,
                    col_vline = col_vline,
                    size_vline = size_vline,
                    linetype_vline = linetype_vline,
                    ggtheme = ggtheme)

    return p

def fviz_ca_biplot(self,
                   axis = [0,1],
                   geom_row = ("point","text"),
                   repel_row = True,
                   geom_col = ("point","text"),
                   repel_col = True,
                   x_lim = None,
                   y_lim = None,
                   x_label = None,
                   y_label = None,
                   title = None,
                   point_alpha_row = 1,
                   point_col_row = "black",
                   fill_row = None,
                   shape_row = "o",
                   point_size_row = 1.5,
                   stroke_row = 0.5,
                   text_type_row = "text",
                   text_alpha_row = 1,
                   angle_row = 0,
                   text_col_row = "black",
                   family_row = None,
                   fontstyle_row = "normal",
                   fontweight_row = "normal",
                   ha_row = "center",
                   lineheight_row = 1.2,
                   text_size_row = 8,
                   va_row = "center",
                   point_alpha_col = 1,
                   point_col_col = "blue",
                   fill_col = None,
                   shape_col = "^", 
                   point_size_col = 1.5,
                   stroke_col = 0.5,
                   text_type_col = "text",
                   text_alpha_col = 1,
                   angle_col = 0,
                   text_col_col = "blue",
                   family_col = None,
                   fontstyle_col = "normal",
                   fontweight_col = "normal",
                   ha_col = "center",
                   lineheight_col = 1.2,
                   text_size_col = 8,
                   va_col = "center",
                   gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                   legend_title = None,
                   habillage = None,
                   palette = None,
                   add_ellipses = False, 
                   geom_ellipse = "polygon",
                   type = "t",
                   level = 0.95,
                   alpha = 0.25,
                   row_sup = True,
                   point_alpha_row_sup = 1,
                   point_col_row_sup = "red",
                   fill_row_sup = None,
                   shape_row_sup = ">",
                   point_size_row_sup = 1.5,
                   stroke_row_sup = 0.5,
                   text_type_row_sup = "text",
                   text_alpha_row_sup = 1,
                   angle_row_sup = 0,
                   text_col_row_sup = "red",
                   family_row_sup = None,
                   fontstyle_row_sup = "normal",
                   fontweight_row_sup = "normal",
                   ha_row_sup = "center",
                   lineheight_row_sup = 1.2,
                   text_size_row_sup = 8,
                   va_row_sup = "center",
                   col_sup = True,
                   point_alpha_col_sup = 1,
                   point_col_col_sup = "darkblue",
                   fill_col_sup = None,
                   shape_col_sup = "v", 
                   point_size_col_sup = 1.5,
                   stroke_col_sup = 0.5,
                   text_type_col_sup = "text",
                   text_alpha_col_sup = 1,
                   angle_col_sup = 0,
                   text_col_col_sup = "darkblue",
                   family_col_sup = None,
                   fontstyle_col_sup = "normal",
                   fontweight_col_sup = "normal",
                   ha_col_sup = "center",
                   lineheight_col_sup = 1.2,
                   text_size_col_sup = 8,
                   va_col_sup = "center",
                   quali_sup = True,
                   point_alpha_quali_sup = 1,
                   point_col_quali_sup = "darkblue",
                   fill_quali_sup = None,
                   shape_quali_sup = "v", 
                   point_size_quali_sup = 1.5,
                   stroke_quali_sup = 0.5,
                   text_type_quali_sup = "text",
                   text_alpha_quali_sup = 1,
                   angle_quali_sup = 0,
                   text_col_quali_sup = "darkblue",
                   family_quali_sup = None,
                   fontstyle_quali_sup = "normal",
                   fontweight_quali_sup = "normal",
                   ha_quali_sup = "center",
                   lineheight_quali_sup = 1.2,
                   text_size_quali_sup = 8,
                   va_quali_sup = "center",
                   add_grid = True,
                   add_hline = True,
                   alpha_hline = 0.5,
                   col_hline = "black",
                   size_hline = 0.5,
                   linetype_hline = "dashed",
                   add_vline = True,
                   alpha_vline = 0.5,
                   col_vline = "black",
                   size_vline = 0.5,
                   linetype_vline = "dashed",
                   ggtheme = None):
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

    `**kwargs`: additionals parameters. For more see `fviz_point`
    
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
    #set title if None
    if title is None:
        title = "CA - Biplot"

    p = fviz_point(self,
                    element = "row",
                    axis = axis,
                    geom = geom_row,
                    repel = repel_row,
                    x_lim = x_lim,
                    y_lim = y_lim,
                    x_label = x_label,
                    y_label = y_label,
                    title = title,
                    point_alpha = point_alpha_row,
                    point_col = point_col_row,
                    fill = fill_row,
                    shape = shape_row,
                    point_size = point_size_row,
                    stroke = stroke_row,
                    text_type = text_type_row,
                    text_alpha = text_alpha_row,
                    angle =  angle_row,
                    text_col = text_col_row,
                    family = family_row,
                    fontstyle = fontstyle_row,
                    fontweight = fontweight_row,
                    ha = ha_row,
                    lineheight = lineheight_row,
                    text_size = text_size_row,            
                    va = va_row,
                    gradient_cols = gradient_cols,
                    legend_title = legend_title,
                    habillage = habillage,
                    palette = palette,
                    add_ellipses = add_ellipses, 
                    geom_ellipse = geom_ellipse,
                    type = type,
                    level = level,
                    alpha = alpha,
                    sup = row_sup,
                    name_sup = "row_sup_",
                    point_alpha_sup = point_alpha_row_sup,
                    point_col_sup = point_col_row_sup,
                    fill_sup = fill_row_sup,
                    shape_sup = shape_row_sup,
                    point_size_sup = point_size_row_sup,
                    stroke_sup = stroke_row_sup,
                    text_type_sup = text_type_row_sup,
                    text_alpha_sup = text_alpha_row_sup,
                    angle_sup = angle_row_sup,
                    text_col_sup = text_col_row_sup,
                    family_sup = family_row_sup,
                    fontstyle_sup = fontstyle_row_sup,
                    fontweight_sup = fontweight_row_sup,
                    ha_sup = ha_row_sup,
                    lineheight_sup = lineheight_row_sup,
                    text_size_sup = text_size_row_sup,
                    va_sup = va_row_sup,
                    quali_sup = quali_sup,
                    name_quali_sup = "quali_sup_",
                    point_alpha_quali_sup = point_alpha_quali_sup,
                    point_col_quali_sup = point_col_quali_sup,
                    fill_quali_sup = fill_quali_sup,
                    shape_quali_sup = shape_quali_sup,
                    point_size_quali_sup = point_size_quali_sup,
                    stroke_quali_sup = stroke_quali_sup,
                    text_type_quali_sup = text_type_quali_sup,
                    text_alpha_quali_sup = text_alpha_quali_sup,
                    angle_quali_sup = angle_quali_sup,
                    text_col_quali_sup = text_col_quali_sup,
                    family_quali_sup = family_quali_sup,
                    fontstyle_quali_sup = fontstyle_quali_sup,
                    fontweight_quali_sup = fontweight_quali_sup,
                    ha_quali_sup = ha_quali_sup,
                    lineheight_quali_sup = lineheight_quali_sup,
                    text_size_quali_sup = text_size_quali_sup,
                    va_quali_sup = va_quali_sup,
                    add_grid = add_grid,
                    add_hline = add_hline,
                    alpha_hline =alpha_hline,
                    col_hline = col_hline,
                    size_hline = size_hline,
                    linetype_hline = linetype_hline,
                    add_vline = add_vline,
                    alpha_vline = alpha_vline,
                    col_vline = col_vline,
                    size_vline = size_vline,
                    linetype_vline = linetype_vline,
                    ggtheme = ggtheme)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add columns points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = fviz_add_point(p = p,
                       data = self.col_.coord,
                        axis = axis,
                        geom = geom_col,
                        repel = repel_col,
                        point_alpha = point_alpha_col,
                        point_col = point_col_col,
                        fill = fill_col,
                        shape = shape_col,
                        point_size = point_size_col,
                        stroke = stroke_col,
                        text_type = text_type_col,
                        text_alpha = text_alpha_col,
                        angle = angle_col,
                        text_col = text_col_col,
                        family = family_col,
                        fontstyle = fontstyle_col,
                        fontweight = fontweight_col,
                        ha = ha_col,
                        lineheight = lineheight_col,
                        text_size = text_size_col,
                        va = va_col)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary columns points
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if col_sup:
        if not hasattr(self, "col_sup_"):
            raise ValueError("No supplumentary columns")
        p = fviz_add_point(p = p,
                           data = self.col_sup_.coord,
                           axis = axis,
                           geom = geom_col,
                           repel = repel_col,
                           point_alpha = point_alpha_col_sup,
                           point_col = point_col_col_sup,
                           fill = fill_col_sup,
                           shape = shape_col_sup,
                           point_size = point_size_col_sup,
                           stroke = stroke_col_sup,
                           text_type = text_type_col_sup,
                           text_alpha = text_alpha_col_sup,
                           angle = angle_col_sup,
                           text_col = text_col_col_sup,
                           family = family_col_sup,
                           fontstyle = fontstyle_col_sup,
                           fontweight = fontweight_col_sup,
                           ha = ha_col_sup,
                           lineheight = lineheight_col_sup,
                           text_size = text_size_col_sup,
                           va = va_col_sup)

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