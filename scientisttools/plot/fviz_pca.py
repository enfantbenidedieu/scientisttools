# -*- coding: utf-8 -*-
from plotnine import arrow, theme_minimal

#intern functions
from .fviz import fviz_scatter, fviz_arrow, add_scatter, add_arrow, fviz_circle, set_axis

def fviz_pca_ind(self,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 col_ind = "black",
                 point_args_ind = dict(shape = "o", size = 1.5),
                 text_args_ind = dict(size = 8),
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
                 point_args_ind_sup = dict(shape = "^", size = 1.5),
                 text_args_ind_sup = dict(size=8),
                 quali_sup = False,
                 col_quali_sup = "red",
                 point_args_quali_sup = dict(shape = ">",size=1.5),
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
    Visualize Principal Component Analysis (PCA) - Graph of individuals
    -------------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. `fviz_pca_ind` provides plotnine based elegant visualization of PCA outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_pca_ind(self,**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `**kwargs`: additionals informations. For more see `fviz_scatter`.

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_pca_ind
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #hraph of individuals
    >>> print(fviz_pca_ind(res_pca,repel=True))
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")

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
        
    #add supplementary categories
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

    #add others elements
    if title is None:
        title = "PCA graph of individuals"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

# Variables Factor Map
def fviz_pca_var(self,
                 axis = [0,1],
                 geom = ("arrow","text"),
                 repel = False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 col_var = "black",
                 segment_args_var = dict(linetype = "solid",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                 text_args_var = dict(size = 8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = None,
                 quanti_sup = False,
                 col_quanti_sup = "blue",
                 segment_args_quanti_sup = dict(linetype="dashed",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                 text_args_quanti_sup = dict(size=8),
                 scale = 1,
                 add_circle = True,
                 col_circle = "gray",
                 x_lim = (-1.1,1.1),
                 y_lim = (-1.1,1.1),
                 x_label = None,
                 y_label = None,
                 title = None,
                 add_hline = True,
                 add_vline = True,
                 add_grid = True,
                 ggtheme = theme_minimal()):
    """
    Visualize Principal Component Analysis (PCA) - Graph of variables
    -----------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. `fviz_pca_var` provides plotnine based elegant visualization of PCA outputs for variables.

    Usage
    -----
    ```python
    >>> fviz_pca_var(self,**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `**kwargs`: additionals informations. For more see `fviz_arrow`.

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_pca_var
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #graph of variables
    >>> print(fviz_pca_var(res_pca))
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    #active variables
    p = fviz_arrow(self = self,
                    element = "var",
                    axis = axis,
                    geom = geom,
                    repel = repel,
                    lim_cos2 = lim_cos2,
                    lim_contrib = lim_contrib,
                    color = col_var,
                    segment_args = segment_args_var,
                    text_args = text_args_var,
                    gradient_cols = gradient_cols,
                    legend_title = legend_title,
                    palette = palette,
                    scale = scale)
    
    #add supplementary quantitative variables
    if quanti_sup:
        if hasattr(self,"quanti_sup_"):
            p = add_arrow(p=p,data=self.quanti_sup_.coord.mul(scale),axis=axis,geom=geom,repel=repel,color=col_quanti_sup,segment_args=segment_args_quanti_sup,text_args=text_args_quanti_sup)

    #add correlation circle
    if add_circle:
        p = fviz_circle(p=p,color=col_circle)

    #add others elements
    if title is None:
        title = "PCA graph of variables"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p
    
def fviz_pca_biplot(self,
                    axis=[0,1],
                    geom_ind = ("point","text"),
                    geom_var = ("arrow","text"),
                    repel_ind = False,
                    repel_var = False,
                    col_ind = "black",
                    point_args_ind = dict(size=1.5),
                    text_args_ind = dict(size=8),
                    col_var = "steelblue",
                    segment_args_var = dict(size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
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
                    col_ind_sup = "blue",
                    point_args_ind_sup = dict(shape="^",size=1.5),
                    text_args_ind_sup = dict(size=8),
                    quanti_sup = False,
                    col_quanti_sup = "darkblue",
                    segment_args_quanti_sup = dict(linetype="dashed",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                    text_args_quanti_sup = dict(size=8),
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
    Visualize Principal Component Analysis (PCA) - Biplot of individuals and variables
    ----------------------------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. `fviz_pca_biplot` provides plotnine based elegant visualization of PCA outputs for individuals and variables.

    Usage
    -----
    ```python
    >>> fviz_pca_biplot(self,**kwargs)
    ```

    Parameters
    ----------
    see `fviz_scatter`, `fviz_arrow`.

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_pca_biplot
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #biplot of individuals and variables
    >>> print(fviz_pca_biplot(res_pca))
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
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
        
    #add supplementary categories
    if quali_sup:
        if hasattr(self,"quali_sup_"):
            p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom_ind,repel=repel_ind,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)
    
    #rescale variables coordinates
    xscale = (max(self.ind_.coord.iloc[:,axis[0]])-min(self.ind_.coord.iloc[:,axis[0]]))/(max(self.var_.coord.iloc[:,axis[0]])-min(self.var_.coord.iloc[:,axis[0]]))
    yscale = (max(self.ind_.coord.iloc[:,axis[1]])-min(self.ind_.coord.iloc[:,axis[1]]))/(max(self.var_.coord.iloc[:,axis[1]])-min(self.var_.coord.iloc[:,axis[1]]))
    scale = min(xscale, yscale)
    
    #add variables informations
    p = add_arrow(p=p,data=self.var_.coord.mul(scale),axis=axis,geom=geom_var,repel=repel_var,color=col_var,segment_args=segment_args_var,text_args=text_args_var)
    
    #add supplementary quantitative variables
    if quanti_sup:
        if hasattr(self,"quanti_sup_"):
            p = add_arrow(p=p,data=self.quanti_sup_.coord.mul(scale),axis=axis,geom=geom_var,repel=repel_var,color=col_quanti_sup,segment_args=segment_args_quanti_sup,text_args=text_args_quanti_sup)

    #set title
    if title is None:
        title = "PCA - Biplot of individuals and variables"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)
    
    return p
    
def fviz_pca(self, element="biplot",**kwargs):
    """
    Visualize Principal Component Analysis (PCA)
    --------------------------------------------

    Description
    -----------
    Plot the graphs for a Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

        * `fviz_pca_ind()`: Graph of individuals
        * `fviz_pca_var()`: Graph of variables (Correlation circle)
        * `fviz_pca_biplot()`: Biplot of individuals and variables

    Usage
    -----
    ```python
    >>> fviz_pca(self, element = "ind",**kwargs)
    >>> fviz_pca(self, element = "var",**kwargs)
    >>> fviz_pca(self, element = "biplot",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCA

    `element`: the element to plot from the output. Allowed values are: 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs (= Correlation circle)
        * 'biplot' for biplot of individuals and variables
    
    `**kwargs`: further arguments passed to or from other methods

    Returns
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    see `fviz_pca_ind`, `fviz_pca_var`, `fviz_pca_biplot`
    """
    if element == "ind":
        return fviz_pca_ind(self,**kwargs)
    elif element == "var":
        return fviz_pca_var(self,**kwargs)
    elif element == "biplot":
        return fviz_pca_biplot(self,**kwargs)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'biplot'")