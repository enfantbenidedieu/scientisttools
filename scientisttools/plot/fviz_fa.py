# -*- coding: utf-8 -*-
from plotnine import ggplot, arrow, theme_minimal

#intern functions
from .fviz import fviz_circle, add_scatter, add_arrow, set_axis

def fviz_fa_ind(self,
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 col_ind = "black",
                 point_args_ind = dict(size=1.5),
                 text_args_ind = dict(size=8),
                 ind_sup = True,
                 col_ind_sup = "blue",
                 point_args_ind_sup = dict(shape="^",size=1.5),
                 text_args_ind_sup = dict(size=8),
                 add_grid = True,
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 add_hline = True,
                 add_vline = True,
                 ggtheme = theme_minimal()):
    """
    Visualize Factor Analysis (FactorAnalysis) - Graph of individuals
    -----------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_fa_ind(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FactorAnalysis

    `**kwargs`: additionals parameters (see `add_scatter`).

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, fviz_fa_ind
    >>> res_fa = FactorAnalysis(n_components=None,rotate=None,max_iter=1)
    >>> res_fa.fit(beer)
    >>> #graph of individuals
    >>> p = fviz_fa_ind(res_fa,repel=True)
    >>> print(p)
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis.")
    
    #active individuals
    p = add_scatter(p=ggplot(),data=self.ind_.coord,axis=axis,geom=geom,repel=repel,color=col_ind,points_args=point_args_ind,text_args=text_args_ind)
    
    #add supplementary individuals
    if ind_sup:
        if hasattr(self,"ind_sup_"):
            p = add_scatter(p=p,data=self.ind_sup_.coord,axis=axis,geom=geom,repel=repel,color=col_ind_sup,points_args=point_args_ind_sup,text_args=text_args_ind_sup)
    
    #add others elements
    if title is None:
        title = "FactorAnalysis - graph of individuals"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

#graph of variables
def fviz_fa_var(self,
                 axis = [0,1],
                 geom = ("arrow","text"),
                 repel = False,
                 col_var ="black",
                 segment_args_var = dict(linetype = "solid",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                 text_args_var = dict(size = 8),
                 scale = 1,
                 add_circle = True,
                 col_circle = "gray",
                 x_lim = (-1.1,1.1),
                 y_lim = (-1.1,1.1),
                 x_label = None,
                 y_label = None,
                 title =None,
                 add_hline = True,
                 add_vline = True,
                 add_grid = True,
                 ggtheme = theme_minimal()):
    
    """
    Visualize Factor Analysis (FactorAnalysis) - Graph of variables
    ----------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_fa_var(self, **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FactorAnalysis

    `**kwargs`: additionals parameters (see `add_arrow`).

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, fviz_fa_var
    >>> res_fa = FactorAnalysis(n_components=None,rotate=None,max_iter=1)
    >>> res_fa.fit(beer)
    >>> #graph of variables
    >>> p = fviz_fa_var(res_fa,repel=True)
    >>> print(p)
    ```
    """
    #check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")

    #active variables
    p = add_arrow(p=ggplot(),data=self.var_.coord.mul(scale),axis=axis,geom=geom,repel=repel,color=col_var,segment_args=segment_args_var,text_args=text_args_var)
    
    #add correlation circle
    if add_circle:
        p = fviz_circle(p=p,color=col_circle)

    #add others elements
    if title is None:
        title = "PFA graph of variables"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p
    
#biplot of individuals and variables
def fviz_fa_biplot(self,
                    axis = [0,1],
                    geom_ind = ("point","text"),
                    geom_var = ("arrow","text"),
                    repel_ind = False,
                    repel_var = True,
                    col_ind = "black",
                    point_args_ind = dict(size=1.5),
                    text_args_ind = dict(size=8),
                    col_var = "steelblue",
                    segment_args_var = dict(linetype = "solid",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                    text_args_var = dict(size = 8),
                    ind_sup = True,
                    col_ind_sup = "blue",
                    point_args_ind_sup = dict(shape="^",size=1.5),
                    text_args_ind_sup = dict(size=8),
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
    Visualize Factor Analysis (FactorAnalysis) - Biplot of individuals and variables
    --------------------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_fa_biplot(self, **kwargs) 
    ```

    Parameters
    ----------
    see `fviz_fa_ind`, `fviz_fa_var`

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, fviz_fa_biplot
    >>> res_fa = FactorAnalysis(n_components=None,rotate=None,max_iter=1)
    >>> res_fa.fit(beer)
    >>> #biplot - graph of individuals and variables
    >>> p = fviz_fa_biplot(res_fa,repel_ind=True,repel_var=True)
    >>> print(p)
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")
    
    #active individuals
    p = add_scatter(p=ggplot(),data=self.ind_.coord,axis=axis,geom=geom_ind,repel=repel_ind,color=col_ind,points_args=point_args_ind,text_args=text_args_ind)

    #add supplementary individuals
    if ind_sup:
        if hasattr(self,"ind_sup_"):
            p = add_scatter(p=p,data=self.ind_sup_.coord,axis=axis,geom=geom_ind,repel=repel_ind,color=col_ind_sup,points_args=point_args_ind_sup,text_args=text_args_ind_sup)
    
    #rescale variables coordinates
    xscale = (max(self.ind_.coord.iloc[:,axis[0]])-min(self.ind_.coord.iloc[:,axis[0]]))/(max(self.var_.coord.iloc[:,axis[0]])-min(self.var_.coord.iloc[:,axis[0]]))
    yscale = (max(self.ind_.coord.iloc[:,axis[1]])-min(self.ind_.coord.iloc[:,axis[1]]))/(max(self.var_.coord.iloc[:,axis[1]])-min(self.var_.coord.iloc[:,axis[1]]))
    scale = min(xscale, yscale)
    
    #add variables informations
    p = add_arrow(p=p,data=self.var_.coord.mul(scale),axis=axis,geom=geom_var,repel=repel_var,color=col_var,segment_args=segment_args_var,text_args=text_args_var)
    
    #set title
    if title is None:
        title = "FactorAnalysis - Biplot of individuals and variables"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)
    
    return p

def fviz_fa(self, element="biplot", **kwargs):
    """
    Visualize Factor Analysis (FactorAnalysis)
    ------------------------------------------

    Description
    -----------
    factor analysis is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. It is used to identify the structure of the relationship between the variable and the respondent. fviz_efa() provides plotnine-based elegant visualization of EFA outputs
    
        * `fviz_fa_ind()`: Graph of individuals
        * `fviz_fa_var()`: Graph of variables
        * `fviz_fa_biplot()`: Biplot of individuals and variables

    Usage
    -----
    ```python
    >>> fviz_fa(self, element = "ind", **kwargs)
    >>> fviz_fa(self, element = "var", **kwargs)
    >>> fviz_fa(self, element = "biplot", **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FactorAnalysis

    `element`: the element to subset. Allowed values are : 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs (correlation circle)
        * 'biplot' for biplot of individuals and variables
    
    `**kwargs`: further arguments passed to or from other methods

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, fviz_fa
    >>> res_fa = FactorAnalysis(n_components=None,rotate=None,max_iter=1)
    >>> res_fa.fit(beer)
    >>> #graph of individuals
    >>> p = fviz_fa(res_fa, element = "ind", repel=True)
    >>> print(p)
    >>> #graph of variables
    >>> p = fviz_fa(res_fa, element = "var", repel=True)
    >>> print(p)
    >>> #biplot - graph of individuals and variables
    >>> p = fviz_fa(res_fa, element = "biplot", repel_ind=True, repel_var=True)
    >>> print(p)
    ```
    """
    if element == "ind":
        return fviz_fa_ind(self,**kwargs)
    elif element == "var":
        return fviz_fa_var(self,**kwargs)
    elif element == "biplot":
        return fviz_fa_biplot(self,**kwargs)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'biplot'")