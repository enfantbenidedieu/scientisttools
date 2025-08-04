# -*- coding: utf-8 -*-
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_segment,
    scale_color_gradient2,
    guides,
    guide_legend,
    stat_ellipse,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    arrow,
    annotate,
    theme_minimal)
from pandas import Categorical, concat
from numpy import issubdtype,number,asarray,ndarray

#intern functions
from .fviz_add import fviz_add, gg_circle,text_label,list_colors

def fviz_fa_ind(self,
                 axis = [0,1],
                 geom = ["point","text"],
                 repel = False,
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label=None,
                 title =None,
                 alpha_ind = 1,
                 col_ind = "black",
                 fill_ind = None,
                 shape_ind = "o",
                 point_size_ind = 1.5,
                 text_size_ind = 8,
                 stroke_ind = 0.5,
                 text_type_ind = "text",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = None,
                 ind_sup = True,
                 alpha_ind_sup = 1,
                 col_ind_sup = "blue",
                 fill_ind_sup = None,
                 shape_ind_sup = "^",
                 point_size_ind_sup = 1.5,
                 text_size_ind_sup = 8,
                 stroke_ind_sup = 0.5,
                 text_type_ind_sup = "text",
                 add_grid = True,
                 ha = "center",
                 va = "center",
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
                 ggtheme=theme_minimal()):
    
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
    `self` : an object of class FactorAnalysis

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `palette` :  a list or tuple specifying the color palette to be used for coloring or filling by groups.
    
    `color` : a color for the active individuals (by default = "black").

    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `ind_sup` : a boolean to either add or not supplementary individuals (by default = True).

    `color_sup` : a color for the supplementary individuals points (by default = "blue").

    `marker_sup` :  a marker style for the supplementary individuals points (by default = "^").

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `repel` : a boolean, whether to avoid overplotting text labels or not (by default == False)

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis.")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or(axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    coord = concat((self.ind_.coord,self.call_.Xtot),axis=1)

    if isinstance(col_ind,str):
        if col_ind in coord.columns:
            if not issubdtype(coord[col_ind].dtype, number):
                raise TypeError("'color' must me a numeric variable")
            c = coord[col_ind].values
            if legend_title is None:
                legend_title = col_ind
    elif isinstance(col_ind,ndarray):
        c = asarray(col_ind)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(col_ind, "labels_"):
        if legend_title is None:
            legend_title = "Cluster"
        coord.loc[:,legend_title] = [str(x+1) for x in col_ind.labels_]
        index = coord[legend_title].unique().tolist()
        coord[legend_title] = Categorical(coord[legend_title],categories=sorted(index),ordered=True)

        if palette is None:
            palette = [x for x in list_colors if x != col_ind][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    # Initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(col_ind,str) and col_ind in [*["cos2","contrib"],*coord.columns]) or (isinstance(col_ind,ndarray)):
        if "point" in geom:
            p = (p + geom_point(aes(color=c),alpha=alpha_ind,fill=fill_ind,shape=shape_ind,size=point_size_ind,stroke=stroke_ind,show_legend=False) 
                    + scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type_ind,repel,mapping=aes(color=c),size=text_size_ind,ha=ha,va=va)
    elif hasattr(col_ind, "labels_"):
        if "point" in geom:
            p = (p + geom_point(aes(color=legend_title,fill=legend_title,shape=legend_title),alpha=alpha_ind,size=point_size_ind,stroke=stroke_ind,show_legend=True)
                    + guides(color=guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type_ind,repel,mapping=aes(color=legend_title),size=text_size_ind,ha=ha,va=va)
        p = p + scale_color_manual(values=palette)
    else:
        if "point" in geom:
            p = p + geom_point(alpha=alpha_ind,color=col_ind,fill=fill_ind,size=point_size_ind,stroke=stroke_ind,show_legend=False)
        if "text" in geom:
            p = p + text_label(text_type_ind,repel,color=col_ind,size=text_size_ind,va=va,ha=ha)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##Add supplementary individuals coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_.coord
            if "point" in geom:
                p = p + geom_point(ind_sup_coord,aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),alpha=alpha_ind_sup,color = col_ind_sup,fill=fill_ind_sup,shape = shape_ind_sup,size=point_size_ind_sup,stroke=stroke_ind_sup,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_ind_sup,repel,data=ind_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),color = col_ind_sup,size=text_size_ind_sup,ha=ha,va=va)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Individuals Factor Map - FA"
    p = fviz_add(p,self,axis,x_label,y_label,title,x_lim,y_lim,add_hline,alpha_hline,col_hline,linetype_hline,size_hline,add_vline,alpha_vline,col_vline,linetype_vline,size_vline,add_grid,ggtheme)     
    
    return p

def fviz_fa_var(self,
                 axis = [0,1],
                 geom = ["arrow", "text"],
                 lim_contrib = None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 alpha_var = 1,
                 col_var ="black",
                 linetype_var = "solid",
                 line_size_var = 0.5,
                 text_size_var = 8,
                 text_type_var = "text",
                 arrow_angle_var = 10,
                 arrow_length_var = 0.1,
                 arrow_type_var = "closed",
                 scale = 1,
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 palette = None,
                 add_circle = True,
                 col_circle = "gray",
                 add_grid =True,
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
                 ha = "center",
                 va = "center",
                 ggtheme=theme_minimal()):
    
    """
    Visualize Exploratory Factor Analysis (EFA) - Graph of variables
    ----------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_efa_var(self, **kwargs)
    ```

    Parameters
    ----------
    ``self` : an object of class FactorAnalysis

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `color` : a color for the active variables (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `palette` :  a list or tuple specifying the color palette to be used for coloring or filling by groups.
    
    `scale` : a numeric specifying scale the variables coordinates (by default 1)

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `text_size` : a numeric value specifying the label size (by default = 8).

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

    `add_circle` : a boolean, whether to add or not a circle to plot.

    `color_circle` : a string specifying the color for the correlation circle (by default = "gray")

    `arrow_angle` : a numeric specifying the angle in degrees between the tail a single edge (by default = 10)

    `arrow_length` : a numeric specifying the length of the edge in "inches" (by default = 0.1)

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or(axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    coord = self.var_.coord.mul(scale)

    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    if isinstance(col_var,str):
        if col_var == "contrib":
            c = self.var_.contrib.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(col_var,ndarray):
        c = asarray(col_var)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(col_var, "labels_"):
        if legend_title is None:
            legend_title = "Cluster"
        coord.loc[:,legend_title] = [str(x+1) for x in col_var.labels_]
        index = coord[legend_title].unique().tolist()
        coord[legend_title] = Categorical(coord[legend_title],categories=sorted(index),ordered=True)

        if palette is None:
            palette = [x for x in list_colors if x != col_var][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    # Initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(col_var,str) and col_var in ["cos2","contrib"]) or (isinstance(col_var,ndarray)):
        # Add gradients colors
        if "arrow" in geom:
            p = (p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c),alpha=alpha_var,linetype=linetype_var,size=line_size_var,arrow = arrow(angle=arrow_angle_var,length=arrow_length_var,type=arrow_type_var))
                   + scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type_var,False,mapping=aes(color=c),size=text_size_var,va=va,ha=ha)
    elif hasattr(col_var, "labels_"):
        if "arrow" in geom:
            p = (p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=legend_title),alpha=alpha_var,linetype=linetype_var,size=line_size_var,arrow = arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
                   + guides(color=guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type_var,False,mapping=aes(color=legend_title),size=text_size_var,ha=ha,va=va)
        p = p + scale_color_manual(values=palette)
    else:
        if "arrow" in geom:
            p = p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"),alpha=alpha_var,color=col_var,linetype=linetype_var,size=line_size_var,arrow = arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
        if "text" in geom:
            p = p + text_label(text_type_var,False,color=col_var,size=text_size_var,ha=ha,va=va)
        
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=col_circle, fill=None)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Variables Factor Map - FA"
    p = fviz_add(p,self,axis,x_label,y_label,title,(-1,1),(-1,1),add_hline,alpha_hline,col_hline,linetype_hline,size_hline,add_vline,alpha_vline,col_vline,linetype_vline,size_vline,add_grid,ggtheme)     
    
    return p

def fviz_fa_biplot(self,
                    axis = [0,1],
                    geom_ind = ["point","text"],
                    geom_var = ["arrow","text"],
                    repel_ind = False,
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = "FA - Biplot",
                    alpha_ind = 1,
                    col_ind = "black",
                    fill_ind = None,
                    shape_ind = "o",
                    point_size_ind = 1.5,
                    stroke_ind = 0.5,
                    text_type_ind = "text",
                    text_size_ind = 8,
                    alpha_var = 1,
                    col_var = "steelblue",
                    linetype_var = "solid",
                    line_size_var = 0.5,
                    arrow_angle_var = 10,
                    arrow_length_var = 0.1,
                    arrow_type_var = "closed",
                    text_type_var = "text",
                    text_size_var = 8,
                    add_circle = False,
                    col_circle = "gray",
                    ind_sup = True,
                    alpha_ind_sup = 1,
                    col_ind_sup = "blue",
                    fill_ind_sup = None,
                    shape_ind_sup = "^",
                    point_size_ind_sup = 1.5,
                    stroke_ind_sup = 0.5,
                    text_type_ind_sup = "text",
                    text_size_ind_sup = 8,
                    add_grid =True,
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
                    ha_ind = "center",
                    va_ind = "center",
                    ha_var = "center",
                    va_var = "center",
                    ggtheme=theme_minimal()):
    """
    Visualize Factor Analysis (FA) - Biplot of individuals and variables
    --------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> fviz_efa_biplot(self, **kwargs) 
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
    >>> 
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or(axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    #Individuals and variables coordinates
    ind, var = self.ind_.coord.iloc[:,axis], self.var_.coord.iloc[:,axis]
    ind.columns, var.columns = ["x","y"], ["x","y"]

    # Rescale variables coordinates
    xscale, yscale = (max(ind["x"]) - min(ind["x"]))/(max(var["x"]) - min(var["x"])), (max(ind["y"]) - min(ind["y"]))/(max(var["y"]) - min(var["y"]))
    scale = min(xscale, yscale)

    #Extract individuals coordinates
    ind_coord, var_coord = self.ind_.coord, self.var_.coord.mul(scale)

    # Initialize
    p = ggplot(data=ind_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_coord.index))

   
    if "point" in geom_ind:
        p = p + geom_point(alpha=alpha_ind,color=col_ind,fill=fill_ind,shape=shape_ind,size=point_size_ind,stroke=stroke_ind,show_legend=False)
    if "text" in geom_ind:
        p = p + text_label(text_type_ind,repel_ind,color=col_ind,size=text_size_ind,ha=ha_ind,va=va_ind)
    
    #add supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_.coord
            if "point" in geom_ind:
                p = p + geom_point(ind_sup_coord,aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),alpha=alpha_ind_sup,color=col_ind_sup,fill=fill_ind_sup,shape=shape_ind_sup,size=point_size_ind_sup,stroke=stroke_ind_sup,show_legend=False)
            if "text" in geom_ind:
                p = p + text_label(text_type_ind_sup,repel_ind,data=ind_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),color=col_ind_sup,size=text_size_ind_sup,ha=ha_ind,va=va_ind)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add active variables
    if "arrow" in geom_var:
            p = p + annotate("segment",x=0,y=0,xend=asarray(var_coord.iloc[:,axis[0]]),yend=asarray(var_coord.iloc[:,axis[1]]),alpha=alpha_var,color=col_var,linetype=linetype_var,size=line_size_var,arrow=arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
    if "text" in geom_var:
        p = p + text_label(text_type_var,False,data=var_coord,mapping=aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_coord.index),color=col_var,size=text_size_var,va=va_var,ha=ha_var)
    
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=col_circle, fill=None)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "PCA - Biplot"
    p = fviz_add(p,self,axis,x_label,y_label,title,x_lim,y_lim,add_hline,alpha_hline,col_hline,linetype_hline,size_hline,add_vline,alpha_vline,col_vline,linetype_vline,size_vline,add_grid,ggtheme)     
    
    return p

def fviz_fa(self,element="biplot",**kwargs):
    """
    Visualize Factor Analysis (EFA)
    -------------------------------

    Description
    -----------
    factor analysis is a statistical technique that is used to reduce data to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena. It is used to identify the structure of the relationship between the variable and the respondent. fviz_efa() provides plotnine-based elegant visualization of EFA outputs
    
        * `fviz_fa_ind()`: Graph of individuals
        * `fviz_fa_var()`: Graph of variables
        * `fviz_fa_biplot()`: Biplot of individuals and variables

    Usage
    -----
    ```python
    >>> fviz_fa(self,element=("ind","var","biplot"))
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
    >>> 
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