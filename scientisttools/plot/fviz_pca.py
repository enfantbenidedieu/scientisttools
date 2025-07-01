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

def fviz_pca_ind(self,
                 axis = [0,1],
                 geom = ["point","text"],
                 repel_ind = False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 x_lim = None,
                 y_lim = None,
                 x_label = None,
                 y_label = None,
                 title = None,
                 alpha_ind = 1,
                 col_ind ="black",
                 fill_ind = None,
                 shape_ind = "o",
                 point_size_ind = 1.5,
                 text_size_ind = 8,
                 stroke_ind = 0.5,
                 text_type_ind = "text",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 habillage = None,
                 palette = None,
                 add_ellipses = False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 ind_sup = True,
                 alpha_ind_sup = 1,
                 col_ind_sup = "blue",
                 fill_ind_sup = None,
                 shape_ind_sup = "^",
                 point_size_ind_sup = 1.5,
                 text_size_ind_sup = 8,
                 stroke_ind_sup = 0.5,
                 text_type_ind_sup = "text",
                 quali_sup = True,
                 alpha_quali_sup = 1,
                 col_quali_sup = "red",
                 fill_quali_sup = None,
                 shape_quali_sup = ">",
                 point_size_quali_sup = 1.5,
                 text_size_quali_sup = 8,
                 stroke_quali_sup = 0.5,
                 text_type_quali_sup = "text",
                 add_grid =True,
                 ha_ind = "center",
                 va_ind = "center",
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
                 ggtheme = theme_minimal()):
    """
    Visualize Principal Component Analysis (PCA) - Graph of individuals
    -------------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. fviz_pca_ind provides plotnine based elegant visualization of PCA outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_pca_ind(self,
                    axis = [0,1],
                    geom = ["point","text"],
                    repel_ind = False,
                    lim_cos2 = None,
                    lim_contrib = None,
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = None,
                    alpha_ind = 1,
                    col_ind ="black",
                    fill_ind = None,
                    shape_ind = "o",
                    point_size_ind = 1.5,
                    text_size_ind = 8,
                    stroke_ind = 0.5,
                    text_type_ind = "text",
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    legend_title = None,
                    habillage = None,
                    palette = None,
                    add_ellipses = False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
                    ind_sup = True,
                    alpha_ind_sup = 1,
                    col_ind_sup = "blue",
                    fill_ind_sup = None,
                    shape_ind_sup = "^",
                    point_size_ind_sup = 1.5,
                    text_size_ind_sup = 8,
                    stroke_ind_sup = 0.5,
                    text_type_ind_sup = "text",
                    quali_sup = True,
                    alpha_quali_sup = 1,
                    col_quali_sup = "red",
                    fill_quali_sup = None,
                    shape_quali_sup = ">",
                    point_size_quali_sup = 1.5,
                    text_size_quali_sup = 8,
                    stroke_quali_sup = 0.5,
                    text_type_quali_sup = "text",
                    add_grid =True,
                    ha_ind = "center",
                    va_ind = "center",
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
                    ggtheme = theme_minimal())
        ```

    Parameters
    ----------
    `self` : an object of class PCA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active individuals points (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `palette` :  a list or tuple specifying the color palette to be used for coloring or filling by groups.
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").
    
    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `ind_sup` : a boolean to either add or not supplementary individuals (by default = True).

    `ind_sup_color` : a color for the supplementary individuals points (by default = "blue").

    `ind_sup_marker` :  a marker style for the supplementary individuals points (by default = "^").

    `ind_sup_point_size` : a numeric value specifying the supplementary individuals marker size (by default = 1.5).
    
    `ind_sup_text_size` : a numeric value specifying the supplementary individuals label size (by default = 8).

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `habillage` : a string or an integer specifying the variables of indexe for coloring the observations by groups. Default value is None.

    `add_ellipses` : a boolean to either add or not ellipses (by default = False). 

    `ellipse_type` : ellipse multivariate distribution (by default = "t" for t-distribution). However, you can set type = "norm" to assume a multivariate normal distribution or type = "euclid" for an euclidean ellipse.

    `confint_level` : ellipse confindence level (by default = 0.95).

    `geom_ellipse` : ellipse geometry (by default = "polygon").

    `quali_sup` : a boolean to either add or not supplementary categorical variable (by default = True).

    `quali_sup_color` : a color for the supplementary categorical variables points (by default = "red").

    `quali_sup_marker` :  a marker style for the supplementary categorical variables points (by default = ">").

    `quali_sup_point_size` : a numeric value specifying the supplementary categorical variables marker size (by default = 1.5).
    
    `quali_sup_text_size` : a numeric value specifying the supplementary categorical variables label size (by default = 8).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `lim_cos2` : a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

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
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_ind
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=[23,24,25,26],quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Graph of individuals
    >>> p = fviz_pca_ind(res_pca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    #individuals factor coordinates and active data
    coord = concat([self.ind_.coord,self.call_.X],axis=1)

    # Add supplementary quantitatives columns
    if self.quanti_sup is not None:
        X_quanti_sup = self.call_.Xtot.loc[:,self.call_.quanti_sup].astype("float")
        if self.ind_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=self.call_.ind_sup)
        coord = concat([coord,X_quanti_sup],axis=1)
    
    # Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_.Xtot.loc[:,self.call_.quali_sup]
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_.ind_sup)
        coord = concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_.cos2.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    if isinstance(col_ind,str):
        if col_ind == "cos2":
            c = self.ind_.cos2.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif col_ind == "contrib":
            c = self.ind_.contrib.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif col_ind in coord.columns.tolist():
            if not issubdtype(coord[col_ind].dtype,number):
                raise TypeError("'color' must me a numeric variable.")
            c = coord[col_ind].values
            if legend_title is None:
                legend_title = col_ind
    elif isinstance(col_ind,ndarray):
        c = asarray(col_ind)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(col_ind,"labels_"):
        if legend_title is None:
            legend_title = "Cluster"
        coord.loc[:,legend_title] = [str(x+1) for x in col_ind.labels_]
        index = coord[legend_title].unique().tolist()
        coord[legend_title] = Categorical(coord[legend_title],categories=sorted(index),ordered=True)

    #set palette and shape
    if habillage is not None or hasattr(col_ind,"labels_"):    
        if habillage is not None:
            index = coord[habillage].unique().tolist()
        if palette is None:
            palette = [x for x in list_colors if x not in [col_ind,col_ind_sup,col_quali_sup]][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
        
        if isinstance(shape_ind,str):
            shape_ind = [shape_ind]*len(index)
        elif isinstance(shape_ind,(list,tuple)):
            if len(shape_ind) != len(index):
                raise TypeError(f"'shape_ind' must be a list or tuple with length {len(index)}.")

    # Initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if habillage is None :  
        if (isinstance(col_ind,str) and col_ind in [*["cos2","contrib"],*coord.columns]) or (isinstance(col_ind,ndarray)):
            if "point" in geom:
                p = (p + geom_point(aes(color=c),alpha=alpha_ind,fill=fill_ind,shape=shape_ind,size=point_size_ind,stroke=stroke_ind,show_legend=False) 
                       + scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                p = p + text_label(text_type_ind,repel_ind,mapping=aes(color=c),size=text_size_ind,ha=ha_ind,va=va_ind)
        elif hasattr(col_ind, "labels_"):
            if "point" in geom:
                p = (p + geom_point(aes(color=legend_title,fill=legend_title,shape=legend_title),alpha=alpha_ind,size=point_size_ind,stroke=stroke_ind,show_legend=True)
                       + guides(color=guide_legend(title=legend_title)))
            if "text" in geom:
                p = p + text_label(text_type_ind,repel_ind,mapping=aes(color=legend_title),size=text_size_ind,ha=ha_ind,va=va_ind)
        else:
            if "point" in geom:
                p = p + geom_point(alpha=alpha_ind,color=col_ind,fill=fill_ind,size=point_size_ind,stroke=stroke_ind,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_ind,repel_ind,color=col_ind,size=text_size_ind,va=va_ind,ha=ha_ind)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in geom:
            p = p + geom_point(aes(color=habillage,fill=habillage,shape=habillage),alpha=alpha_ind,size=point_size_ind,stroke=stroke_ind,show_legend=True)
        if "text" in geom:
            p = p + text_label(text_type_ind,repel_ind,mapping=aes(color=habillage),size=text_size_ind,va=va_ind,ha=ha_ind)
        if add_ellipses:
            p = p + stat_ellipse(geom=geom_ellipse,mapping=aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)

    if habillage is not None or hasattr(col_ind,"labels_"):
        p = p + scale_color_manual(values=palette,labels=index)+scale_fill_manual(values=palette,labels=index)+scale_shape_manual(values=shape_ind,labels=index)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##Add supplementary individuals coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_.coord
            if "point" in geom:
                p = p + geom_point(ind_sup_coord,aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),alpha=alpha_ind_sup,color = col_ind_sup,fill=fill_ind_sup,shape = shape_ind_sup,size=point_size_ind_sup,stroke=stroke_ind_sup,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_ind_sup,repel_ind,data=ind_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),color = col_ind_sup,size=text_size_ind_sup,ha=ha_ind,va=va_ind)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary categorical variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_.coord
            if "point" in geom:
                p = p + geom_point(quali_sup_coord,aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),alpha=alpha_quali_sup,color = col_quali_sup,fill=fill_quali_sup,shape = shape_quali_sup,size=point_size_quali_sup,stroke=stroke_quali_sup,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_quali_sup,repel_ind,data=quali_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),color = col_quali_sup,size=text_size_quali_sup,ha=ha_ind,va=va_ind)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Individuals Factor Map - PCA"
    p = fviz_add(p,self,axis,x_label,y_label,title,x_lim,y_lim,add_hline,alpha_hline,col_hline,linetype_hline,size_hline,add_vline,alpha_vline,col_vline,linetype_vline,size_vline,add_grid,ggtheme)     
    
    return p

# Variables Factor Map
def fviz_pca_var(self,
                 axis=[0,1],
                 geom = ["arrow","text"],
                 lim_cos2 = None,
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
                 quanti_sup=True,
                 alpha_quanti_sup = 1,
                 col_quanti_sup = "blue",
                 linetype_quanti_sup = "dashed",
                 line_size_quanti_sup = 0.5,
                 text_size_quanti_sup = 8,
                 text_type_quanti_sup = "text",
                 arrow_angle_quanti_sup = 10,
                 arrow_length_quanti_sup = 0.1,
                 arrow_type_quanti_sup = "closed",
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
                 ha_var = "center",
                 va_var = "center",
                 ggtheme=theme_minimal()):
    """
    Visualize Principal Component Analysis (PCA) - Graph of variables
    -----------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. fviz_pca_var provides plotnine based elegant visualization of PCA outputs for variables.

    Usage
    -----
    ```python
    >>> fviz_pca_var(self,
                    axis = [0,1],
                    x_label = None,
                    y_label = None,
                    title = None,
                    color = "black",
                    geom = ["arrow","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    scale = 1,
                    legend_title = None,
                    text_type = "text",
                    text_size = 8,
                    add_grid = True,
                    quanti_sup=True,
                    quanti_sup_color = "blue",
                    quanti_sup_linestyle = "dashed",
                    quanti_sup_text_size = 8,
                    add_hline = True,
                    add_vline = True,
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style ="dashed",
                    ha = "center",
                    va = "center",
                    add_circle = True,
                    circle_color = "gray",
                    arrow_angle = 10,
                    arrow_length = 0.1,
                    lim_cos2 = None,
                    lim_contrib = None,
                    ggtheme=theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active variables (by default = "black").

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `palette` :  a list or tuple specifying the color palette to be used for coloring or filling by groups.
    
    `scale` : a numeric specifying scale the variables coordinates (by default 1)

    `legend_title` : a string corresponding to the title of the legend (by default = None).
    
    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `text_size` : a numeric value specifying the label size (by default = 8).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `quanti_sup` : a boolean to either add or not supplementary quantitatives variables (by default = True).

    `quanti_sup_color` : a color for the supplementary quantitatives quantitative variables (by default = "blue").

    `quanti_sup_linestyle` : a string specifying the supplementary quantitative variables line style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `quanti_sup_text_size` : a numeric value specifying the supplementary quantitative variables label size (by default = 8).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `add_circle` : a boolean, whether to add or not a circle to plot.

    `circle_color` : a string specifying the color for the correlation circle (by default = "gray")

    `arrow_angle` : a numeric specifying the angle in degrees between the tail a single edge (by default = 10)

    `arrow_length` : a numeric specifying the length of the edge in "inches" (by default = 0.1)

    `lim_cos2` : a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

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
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_var
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Graph of variables
    >>> p = fviz_pca_var(res_pca)
    >>> print(p)
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    #variables factor coordinates
    coord = self.var_.coord.mul(scale)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_.cos2.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")

    if isinstance(col_var,str):
        if col_var == "cos2":
            c = self.var_.cos2.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif col_var == "contrib":
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
            palette = [x for x in list_colors if x not in [col_var,col_quanti_sup]][:len(index)]
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
            p = p + text_label(text_type_var,False,mapping=aes(color=c),size=text_size_var,va=va_var,ha=ha_var)
    elif hasattr(col_var, "labels_"):
        if "arrow" in geom:
            p = (p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=legend_title),alpha=alpha_var,linetype=linetype_var,size=line_size_var,arrow = arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
                   + guides(color=guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type_var,False,mapping=aes(color=legend_title),size=text_size_var,ha=ha_var,va=va_var)
        p = p + scale_color_manual(values=palette)
    else:
        if "arrow" in geom:
            p = p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"),alpha=alpha_var,color=col_var,linetype=linetype_var,size=line_size_var,arrow = arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
        if "text" in geom:
            p = p + text_label(text_type_var,False,color=col_var,size=text_size_var,ha=ha_var,va=va_var)

    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=col_circle, fill=None)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##Add supplmentary continuous variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if quanti_sup:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_.coord.mul(scale)
            if "arrow" in geom:
                p  = p + annotate("segment",x=0,y=0,xend=asarray(sup_coord.iloc[:,axis[0]]),yend=asarray(sup_coord.iloc[:,axis[1]]),alpha=alpha_quanti_sup,color=col_quanti_sup,linetype=linetype_quanti_sup,size=line_size_quanti_sup,arrow = arrow(length=arrow_length_quanti_sup,angle=arrow_angle_quanti_sup,type=arrow_type_quanti_sup))
            if "text" in geom:
                p  = p + text_label(text_type_quanti_sup,False,data=sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),color=col_quanti_sup,size=text_size_quanti_sup,ha=ha_var,va=va_var)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Variables Factor Map - PCA"
    p = fviz_add(p,self,axis,x_label,y_label,title,(-1,1),(-1,1),add_hline,alpha_hline,col_hline,linetype_hline,size_hline,add_vline,alpha_vline,col_vline,linetype_vline,size_vline,add_grid,ggtheme)     
    
    return p
    
def fviz_pca_biplot(self,
                    axis=[0,1],
                    geom_ind = ["point","text"],
                    geom_var = ["arrow","text"],
                    repel_ind = False,
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = "PCA - Biplot",
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
                    habillage = None,
                    palette = None,
                    add_ellipses = False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
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
                    quali_sup = True,
                    alpha_quali_sup = 1,
                    col_quali_sup = "red",
                    fill_quali_sup = None,
                    shape_quali_sup = "v",
                    point_size_quali_sup = 1.5,
                    stroke_quali_sup = 0.5,
                    text_type_quali_sup = "text",
                    text_size_quali_sup = 8,
                    quanti_sup = True,
                    alpha_quanti_sup = 1,
                    col_quanti_sup = "darkred",
                    linetype_quanti_sup = "dashed",
                    line_size_quanti_sup = 0.5,
                    text_size_quanti_sup = 8,
                    text_type_quanti_sup = "text",
                    arrow_angle_quanti_sup = 10,
                    arrow_length_quanti_sup = 0.1,
                    arrow_type_quanti_sup = "closed",
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
    Visualize Principal Component Analysis (PCA) - Biplot of individuals and variables
    ----------------------------------------------------------------------------------

    Description
    -----------
    Principal components analysis (PCA) reduces the dimensionality of multivariate data, to two or three that can be visualized graphically with minimal loss of information. fviz_pca_biplot provides plotnine based elegant visualization of PCA outputs for individuals and variables.

    Parameters
    ----------
    see `fviz_pca_ind`, `fviz_pca_var`.

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, fviz_pca_biplot
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Biplot of individuals and variables
    >>> p = fviz_pca_biplot(res_pca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Individuals and variables coordinates
    ind, var = self.ind_.coord.iloc[:,axis], self.var_.coord.iloc[:,axis]
    ind.columns, var.columns = ["x","y"], ["x","y"]

    # Rescale variables coordinates
    xscale, yscale = (max(ind["x"])-min(ind["x"]))/(max(var["x"])-min(var["x"])), (max(ind["y"])-min(ind["y"]))/(max(var["y"])-min(var["y"]))
    scale = min(xscale, yscale)

    #Extract individuals and variables coordinates
    ind_coord, var_coord = self.ind_.coord, self.var_.coord.mul(scale)

    # Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_.Xtot.loc[:,self.call_.quali_sup]
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_.ind_sup)
        ind_coord = concat([ind_coord,X_quali_sup],axis=1)
    
   # Initialize
    p = ggplot(data=ind_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_coord.index))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #acctive individuals
    if habillage is None :  
        if "point" in geom_ind:
            p = p + geom_point(alpha=alpha_ind,color=col_ind,fill=fill_ind,shape=shape_ind,size=point_size_ind,stroke=stroke_ind,show_legend=False)
        if "text" in geom_ind:
            p = p + text_label(text_type_ind,repel_ind,color=col_ind,size=text_size_ind,ha=ha_ind,va=va_ind)
    else:
        if habillage not in ind_coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        index = ind_coord[habillage].unique().tolist()
        if palette is None:
            palette = [x for x in list_colors if x not in [col_ind,col_var,col_ind_sup,col_quali_sup,col_quanti_sup]][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
        if "point" in geom_ind:
            p = p + geom_point(aes(color=habillage,fill=habillage,shape=habillage),alpha=alpha_ind,size=point_size_ind,stroke=stroke_ind,show_legend=True)
        if "text" in geom_ind:
            p = p + text_label(text_type_ind,repel_ind,mapping=aes(color=habillage),size=text_size_ind,ha=ha_ind,va=va_ind)
        if add_ellipses:
            p = p + stat_ellipse(geom=geom_ellipse,mapping=aes(color = habillage,fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
        p = p + scale_color_manual(values=palette) + scale_fill_manual(values=palette)
    
    #add supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            ind_sup_coord = self.ind_sup_.coord
            if "point" in geom_ind:
                p = p + geom_point(ind_sup_coord,aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),alpha=alpha_ind_sup,color=col_ind_sup,fill=fill_ind_sup,shape=shape_ind_sup,size=point_size_ind_sup,stroke=stroke_ind_sup,show_legend=False)
            if "text" in geom_ind:
                p = p + text_label(text_type_ind_sup,repel_ind,data=ind_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_sup_coord.index),color=col_ind_sup,size=text_size_ind_sup,ha=ha_ind,va=va_ind)
    
    #add supplementary qualitatives coordinates
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_.coord
            if "point" in geom_ind:
                p = p + geom_point(quali_sup_coord,aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),alpha=alpha_quali_sup,color=col_quali_sup,fill=fill_quali_sup,shape=shape_quali_sup,size=point_size_quali_sup,stroke=stroke_quali_sup,show_legend=False)
            if "text" in geom_ind:
                p = p + text_label(text_type_quali_sup,repel_ind,data=quali_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),color=col_quali_sup,size=text_size_quali_sup,ha=ha_ind,va=va_ind)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add active variables
    if "arrow" in geom_var:
            p = p + annotate("segment",x=0,y=0,xend=asarray(var_coord.iloc[:,axis[0]]),yend=asarray(var_coord.iloc[:,axis[1]]),alpha=alpha_var,color=col_var,linetype=linetype_var,size=line_size_var,arrow=arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
    if "text" in geom_var:
        p = p + text_label(text_type_var,False,data=var_coord,mapping=aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_coord.index),color=col_var,size=text_size_var,va=va_var,ha=ha_var)
    
    #add supplmentary continuous variables
    if quanti_sup:
        if hasattr(self, "quanti_sup_"):
            quanti_sup_coord = self.quanti_sup_.coord.mul(scale)
            if "arrow" in geom_var:
                p  = p + annotate("segment",x=0,y=0,xend=asarray(quanti_sup_coord.iloc[:,axis[0]]),yend=asarray(quanti_sup_coord.iloc[:,axis[1]]),alpha=alpha_quanti_sup,color=col_quanti_sup,linetype=linetype_quanti_sup,size=line_size_quanti_sup,arrow=arrow(length=arrow_length_quanti_sup,angle=arrow_angle_quanti_sup,type=arrow_type_quanti_sup))
            if "text" in geom_var:
                p  = p + text_label(text_type_quanti_sup,False,data=quanti_sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_sup_coord.index),color=col_quanti_sup,size=text_size_quanti_sup,ha=ha_var,va=va_var)
    
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
    
def fviz_pca(self,choice="biplot",**kwargs):
    """
    Visualize Principal Component Analysis (PCA)
    --------------------------------------------

    Description
    -----------
    Plot the graphs for a Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

        * fviz_pca_ind() : Graph of individuals
        * fviz_pca_var() : Graph of variables (Correlation circle)
        * fviz_pca_biplot() : Biplot of individuals and variables

    Usage
    -----
    ```python
    >>> fviz_pca(self,choice=("ind","var","biplot"))
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    `choice` : the element to plot from the output. Possible value are : 
        * 'ind' for the individuals graphs
        * 'var' for the variables graphs (= Correlation circle)
        * 'biplot' for biplot of individuals and variables
    
    `**kwargs` : further arguments passed to or from other methods

    Returns
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    seff `fviz_pca_ind`, `fviz_pca_var`, `fviz_pca_biplot`
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    
    if choice not in ["ind","var","biplot"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'biplot'")

    if choice == "ind":
        return fviz_pca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_pca_var(self,**kwargs)
    else:
        return fviz_pca_biplot(self,**kwargs)