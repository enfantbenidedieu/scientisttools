# -*- coding: utf-8 -*-
from plotnine import theme_minimal

#intern functions
from .fviz import fviz_scatter, fviz_arrow, add_scatter, add_arrow, fviz_circle, set_axis

def fviz_famd_ind(self,
                  axis = [0,1],
                  geom = ("point","text"),
                  repel = False,
                  lim_cos2 = None,
                  lim_contrib = None,
                  col_ind ="blue",
                  point_args_ind = dict(size=1.5),
                  text_args_ind = dict(size=8),
                  col_quali_var = "black",
                  point_args_quali_var = dict(shape="^",size=1.5),
                  text_args_quali_var = dict(size=8),
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
                  point_args_ind_sup = dict(shape="v",size = 1.5),
                  text_args_ind_sup = dict(size=8),
                  quali_sup = False,
                  col_quali_sup = "darkred",
                  point_args_quali_sup = dict(shape="<",size= 1.5),
                  text_args_quali_sup = dict(size=8),
                  add_grid =True,
                  add_hline = True,
                  add_vline = True,
                  x_lim = None,
                  y_lim = None,
                  x_label = None,
                  y_label = None,
                  title = None,                
                  ggtheme = theme_minimal()):
    
    """
    Visualize Factor Analysis of Mixed Data (FAMD) - Graph of individuals
    ---------------------------------------------------------------------

    Description
    -----------
    Factor analysis of mixed data (FAMD) is used to analyze a data set containing both quantitative and qualitative variables. fviz_famd_ind() provides plotnine-based elegant visualization of FAMD outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_famd_ind(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `**kwargs`: additionals informations (see `fviz_scatter`).

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import autos2005, FAMD, fviz_famd_ind
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),quanti_sup=(12,13,14),quali_sup=15)
    >>> res_famd.fit(autos2005)
    >>> #graph of individuals
    >>> p = fviz_famd_ind(res_famd)
    >>> print(p)
    ```
    """
    # Check if self is an object of class FAMD  
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")
    
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
    
    #add active categories
    p = add_scatter(p=p,data=self.quali_var_.coord,axis=axis,geom=geom,repel=repel,color=col_quali_var,points_args=point_args_quali_var,text_args=text_args_quali_var)
    
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
        title = "Individuals - FAMD"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p
    
def fviz_famd_var(self,
                  element = "var",
                  axis = [0,1],
                  geom = ("arrow","point","text"),
                  repel = False,
                  lim_cos2 = None,
                  lim_contrib = None,
                  col_var ="red",
                  point_args_var = dict(shape="^",size=1.5),
                  segment_args_var = dict(linetype="solid",size=0.5,alpha=1),
                  text_args_var = dict(size=8),
                  gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),      
                  legend_title = None,
                  palette = None,
                  quanti_sup = False,
                  col_quanti_sup = "black",
                  point_args_quanti_sup = dict(shape="v",size=1.5),
                  segment_args_quanti_sup = dict(linetype="dashed",size=0.5,alpha=1),
                  text_args_quanti_sup = dict(size=8),
                  quali_sup = False,
                  col_quali_sup = "black",
                  point_args_quali_sup = dict(shape="<",size=1.5),
                  text_args_quali_sup = dict(size=8),
                  scale = 1,
                  add_circle = True,
                  col_circle = "gray",
                  add_grid =True,
                  add_hline = True,
                  add_vline =True,
                  x_lim = None,
                  y_lim = None,
                  x_label = None,
                  y_label = None,
                  title = None,
                  ggtheme=theme_minimal()):
    """
    Visualize Factor Analysis of Mixed Data (FAMD) - Graph of variables
    -------------------------------------------------------------------

    Description
    -----------
    Factor analysis of mixed data (FAMD) is used to analyze a data set containing both quantitative and qualitative variables. fviz_famd_var() provides plotnine-based elegant visualization of FAMD outputs for both quantitatives and qualitatives variables.

    Usage
    -----
    ```python
    >>> fviz_famd_var(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `**kwargs`: additionals informations (see `fviz_scatter`).

    Returns
    -------
    a plotnine

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import autos2005, FAMD, fviz_famd_var
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),quanti_sup=(12,13,14),quali_sup=15)
    >>> res_famd.fit(autos2005)
    >>> #graph of quantitative variables
    >>> p = fviz_famd_var(res_famd, element = "quanti_var")
    >>> print(p)
    >>> #graph of variables categories
    >>> p = fviz_famd_var(res_famd, element = "quali_var")
    >>> print(p)
    >>> #graph of variables
    >>> p = fviz_famd_var(res_famd, element = "var")
    >>> print(p)
    ```
    """
    #check if self is an object of class FAMD
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")

    if element == "quanti_var":
        #check if quantitative variables in active variables
        if not hasattr(self,"quanti_var_"):
            raise ValueError("No quantitative variables in active variables")
        
        #active variables
        p = fviz_arrow(self = self,
                       element = "quanti_var",
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

        #set title
        if title is None:
            title = "Quantitative variables - FAMD"
    elif element == "quali_var":
        #check if qualitative variables in active variables
        if not hasattr(self,"quali_var_"):
            raise ValueError("No qualitative variables in active variables")
        
        #active categories
        p = fviz_scatter(self,
                         element = "quali_var",
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
                         habillage = None,
                         palette = palette,
                         add_ellipses = False)
            
        #add supplementary categories
        if quali_sup:
            if hasattr(self,"quali_sup_"):
                p = add_scatter(p=p,data=self.quali_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

        #set title
        if title is None:
            title = "Qualitative variable categories - FAMD"
    elif element == "var":
        #check if mixed of variables in active variables
        if not hasattr(self,"var_"):
            raise ValueError("No mixed of variables in active variables")
        
        #active categories
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
                         habillage = None,
                         palette = palette,
                         add_ellipses = False)

        #add supplementary qualitative variables
        if quali_sup:
            if hasattr(self,"quali_sup_"):
                p = add_scatter(p=p,data=self.quali_sup_.eta2,axis=axis,geom=geom,repel=repel,color=col_quali_sup,points_args=point_args_quali_sup,text_args=text_args_quali_sup)

        #add supplementary quantitative variables
        if quanti_sup:
            if hasattr(self,"quanti_sup_"):
                p = add_scatter(p=p,data=self.quanti_sup_.cos2,axis=axis,geom=geom,repel=repel,color=col_quanti_sup,points_args=point_args_quanti_sup,text_args=text_args_quanti_sup)
        
        #set title
        if title is None:
            title = "Variables - FAMD"
    else:
        raise ValueError("'element' must be one of : 'quanti_var', 'quali_var' or 'var'.")

    #add others informations
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p
    
def fviz_famd(self, element="ind", **kwargs):
    """
    Visualize Factor Analysis of Mixed Data
    ---------------------------------------
    
    Description
    -----------
    Plot the graphs for Factor Analysis of Mixed Data (FAMD) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

        * `fviz_famd_ind()`: Graph of individuals
        * `fviz_famd_var()`: Graph of variables 

    Usage
    -----
    ```python
    >>> fviz_famd(self, element="ind", **kwargs)
    >>> fviz_famd(self, element="quanti_var", **kwargs)
    >>> fviz_famd(self, element="quali_var", **kwargs)
    >>> fviz_famd(self, element="var", **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `element`: the element to plot from the output. Possible values are :
        * "ind" for the individual graphs
        * "quanti_var" for the quantitative variables (=correlation circle)
        * "quali_var" for the categorical variables graphs
        * "var" for all the variables (quantitative and qualitative)
    
    `**kwargs`: further arguments passed to or from other methods

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    see `fviz_famd_ind`, `fviz_famd_var`, `fviz_famd_quali_var`, `fviz_famd_var`
    """
    if element == "ind":
        return fviz_famd_ind(self,**kwargs)
    elif element in ("var","quali_var","quanti_var"):
        return fviz_famd_var(self,**kwargs)
    else:
        raise ValueError("'element' should be one of 'ind', 'quanti_var', 'quali_var', 'var'.")