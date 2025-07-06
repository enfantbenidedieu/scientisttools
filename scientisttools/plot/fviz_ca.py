# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

#intern functions
from .fviz_add import fviz_add, text_label, list_colors

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
                alpha_row = 1,
                col_row = "black",
                fill_row = None,
                shape_row = "o",
                point_size_row = 1.5,
                text_size_row = 8,
                stroke_row = 0.5,
                text_type_row = "text",
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                habillage = None,
                palette = None,
                add_ellipses = False, 
                ellipse_type = "t",
                confint_level = 0.95,
                geom_ellipse = "polygon",
                row_sup=True,
                alpha_row_sup = 1,
                col_row_sup = "steelblue",
                fill_row_sup = None,
                shape_row_sup = "^",
                point_size_row_sup = 1.5,
                text_size_row_sup = 8,
                stroke_row_sup = 0.5,
                text_type_row_sup = "text",
                quali_sup = True,
                alpha_quali_sup = 1,
                col_quali_sup = "red",
                fill_quali_sup = None,
                shape_quali_sup = ">",
                point_size_quali_sup = 1.5,
                text_size_quali_sup = 8,
                stroke_quali_sup = 0.5,
                text_type_quali_sup = "text",
                add_grid = True,
                ha = "center",
                va = "center",
                add_hline = True,
                alpha_hline = 0.5,
                col_hline = "black",
                size_hline = 0.5,
                linestyle_hline="dashed",
                add_vline = True,
                alpha_vline = 0.5,
                col_vline="black",
                size_vline = 0.5,
                linestyle_vline ="dashed",
                ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize Correspondence Analysis - Graph of row variables
    ----------------------------------------------------------

    Description
    -----------
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. `fviz_ca_row` provides plotnine based elegant visualization of CA outputs from Python functions.

    Usage
    -----
    ```python
    >>> fviz_ca_row(self,
                    axis=[0,1],
                    x_lim=None,
                    y_lim=None,
                    x_label = None,
                    y_label = None,
                    title =None,
                    color ="black",
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    palette = None,
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
                    add_grid =True,
                    row_sup=True,
                    row_sup_color = "blue",
                    row_sup_marker = "^",
                    row_sup_point_size = 1.5,
                    row_sup_text_size = 8,
                    row_sup_text_type = "text",
                    quali_sup = True,
                    quali_sup_color = "red",
                    quali_sup_marker = ">",
                    quali_sup_point_size = 1.5,
                    quali_sup_text_size = 8,
                    quali_sup_text_type = "text",
                    add_hline = True,
                    add_vline=True,
                    legend_title = None,
                    habillage=None,
                    add_ellipses=False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    lim_cos2 = None,
                    lim_contrib = None,
                    repel=False,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `axis`: a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim`: a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim`: a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label`: a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label`: a string specifying the label text of y (by default = None and a x_label is chosen).

    `title`: a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color`: a color for the active rows points (by default = "black").

    `geom`: a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `palette` :  a list or tuple specifying the color palette to be used for coloring or filling by groups.
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").
    
    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `row_sup` : a boolean to either add or not supplementary row points (by default = True).

    `row_sup_color` : a color for the supplementary rows points (by default = "blue").

    `row_sup_marker` :  a marker style for the supplementary rows points (by default = "^").

    `row_sup_point_size` : a numeric value specifying the supplementary rows marker size (by default = 1.5).
    
    `row_sup_text_size` : a numeric value specifying the supplementary rows label size (by default = 8).

    `row_sup_text_type` :  a string specifying either `geom_text` or `geom_label` for supplementary rows (by default = "text"). Allowed values are : "text" or "label".

    `quali_sup` : a boolean to either add or not supplementary categorical variables (by default = True).

    `quali_sup_color` : a color for supplementary categorical vaiables (by default = "red").

    `quali_sup_marker` : the marker stype for the supplementary categorical variables (by default = ">").

    `quali_sup_point_size` : a numeric value specifying the supplementary categorical variables marker size (by default = 1.5).
    
    `quali_sup_text_size` : a numeric value specifying the supplementary categorical variables label size (by default = 8).

    `quali_sup_text_type` :  a string specifying either `geom_text` or `geom_label` for supplementary categorical variables (by default = "text"). Allowed values are : "text" or "label".

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `habillage` : color the rows points among a categorical variables, give the name of the supplementary categorical variable (by default = None).

    `add_ellipses` : a boolean to either add or not ellipses (by default = False). 

    `ellipse_type` : ellipse multivariate distribution (by default = "t" for t-distribution). However, you can set type = "norm" to assume a multivariate normal distribution or type = "euclid" for an euclidean ellipse.

    `confint_level` : ellipse confindence level (by default = 0.95).

    `geom_ellipse` : ellipse geometry (by default = "polygon").

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom", "baseline" or "center_baseline"

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

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
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2  = load_children2()
    >>> from scientisttools import CA, fviz_ca_row
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8)
    >>> res_ca.fit(children2)
    >>> #graph of row variables
    >>> p = fviz_ca_row(res_ca)
    >>> print(p)
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    #rows factor coordinates and active data
    coord = pd.concat((self.row_.coord,self.call_.X),axis=1)

    #Add supplementary columns
    if self.col_sup is not None:
        X_col_sup = self.call_.Xtot.loc[:,self.call_.col_sup].astype("float")
        if self.row_sup is not None:
            X_col_sup = X_col_sup.drop(index=self.call_.row_sup)
        coord = pd.concat([coord,X_col_sup],axis=1)

    #Add supplementary quantitatives columns
    if self.quanti_sup is not None:
        X_quanti_sup = self.call_.Xtot.loc[:,self.call_.quanti_sup].astype("float")
        if self.row_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=self.call_.row_sup)
        coord = pd.concat([coord,X_quanti_sup],axis=1)
    
    #Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_.Xtot.loc[:,self.call_.quali_sup].astype("object")
        if self.row_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_.row_sup)
        coord = pd.concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.row_.cos2.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.row_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")
    
    # Set color if cos2, contrib or continuous variables
    if isinstance(col_row,str):
        if col_row == "cos2":
            c = self.row_.cos2.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Cos2"
        elif col_row == "contrib":
            c = self.row_.contrib.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif col_row in coord.columns.tolist():
            if not np.issubdtype(coord[col_row].dtype, np.number):
                raise TypeError("'color' must me a numeric variable.")
            c = coord[col_row].values
            if legend_title is None:
                legend_title = col_row
    elif isinstance(col_row,np.ndarray):
        c = np.asarray(col_row)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(col_row,"labels_"):
        if legend_title is None:
            legend_title = "Cluster"
        coord.loc[:,legend_title] = [str(x+1) for x in col_row.labels_]
        coord[legend_title] = pd.Categorical(coord[legend_title],categories=sorted(coord[legend_title].unique().tolist()),ordered=True)

    #set palette
    if habillage is not None or hasattr(col_row,"labels_"):
        if hasattr(col_row,"labels_"):
            index = coord[legend_title].unique().tolist()
        if habillage is not None:
            index = coord[habillage].unique().tolist()
        if palette is None:
            palette = [x for x in list_colors if x not in [col_row,col_row_sup,col_quali_sup]][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if habillage is None:
        if (isinstance(col_row,str) and col_row in [*["cos2","contrib"],*coord.columns]) or (isinstance(col_row,np.ndarray)):
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(colour=c),alpha=alpha_row,fill=fill_row,shape=shape_row,size=point_size_row,stroke=stroke_row,show_legend=False)
                       + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                p = p + text_label(text_type_row,repel,mapping=pn.aes(color=c),size=text_size_row,va=va,ha=ha)
        elif hasattr(col_row, "labels_"):
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(color=legend_title),alpha=alpha_row,fill=fill_row,size=point_size_row,stroke=stroke_row,show_legend=False)
                       + pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom:
                p = p + text_label(text_type_row,repel,mapping=pn.aes(color=legend_title),size=text_size_row,va=va,ha=ha)
            p = p + pn.scale_color_manual(values=palette)
        else:
            if "point" in geom:
                p = p + pn.geom_point(alpha=alpha_row,color=col_row,fill=fill_row,shape=shape_row,size=point_size_row,stroke=stroke_row,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_row,repel,color=col_row,size=text_size_row,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame")
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color=habillage),alpha=alpha_row,shape=shape_row,size=point_size_row,stroke=stroke_row,show_legend=True)
        if "text" in geom:
            p = p + text_label(text_type_row,repel,mapping=pn.aes(color=habillage),size=text_size_row,va=va,ha=ha)
        if add_ellipses:
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
        p = p + pn.scale_color_manual(values=palette)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## add supplementary rows coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if row_sup:
        if hasattr(self, "row_sup_"):
            row_sup_coord = self.row_sup_.coord
            if "point" in geom:
                p = p + pn.geom_point(row_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),alpha=alpha_row_sup,color = col_row_sup,fill=fill_row_sup,shape = shape_row_sup,size=point_size_row_sup,stroke=stroke_row_sup,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_row_sup,repel,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),color = col_row_sup,size=text_size_row_sup,va=va,ha=ha)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary categorical variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_.coord
            if "point" in geom:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),alpha=alpha_quali_sup,color = col_quali_sup,fill=fill_quali_sup,shape = shape_quali_sup,size=point_size_quali_sup,stroke=stroke_quali_sup,show_legend=False)
            if "text" in geom:
                p = p + text_label(text_type_quali_sup,repel,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),color = col_quali_sup,size=text_size_quali_sup,va=va,ha=ha)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Row points - CA"
    p = fviz_add(p,self,axis,x_label,y_label,title,x_lim,y_lim,add_hline,alpha_hline,col_hline,linestyle_hline,size_hline,add_vline,alpha_vline,col_vline,linestyle_vline,size_vline,add_grid,ggtheme)     
    
    return p

def fviz_ca_col(self,
                axis=[0,1],
                geom = ["point","text"],
                repel=False,
                lim_cos2 = None,
                lim_contrib = None,
                x_lim= None,
                y_lim=None,
                x_label = None,
                y_label = None,
                title =None,
                alpha_col = 1,
                col_col = "black",
                fill_col = None,
                shape_col = "o",
                point_size_col = 1.5,
                text_size_col = 8,
                stroke_col = 0.5,
                text_type_col = "text",
                gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                legend_title = None,
                palette = None,
                col_sup = True,
                alpha_col_sup = 1,
                col_col_sup = "steelblue",
                fill_col_sup = None,
                shape_col_sup = "^",
                point_size_col_sup = 1.5,
                text_size_col_sup = 8,
                stroke_col_sup = 0.5,
                add_grid = True,
                ha = "center",
                va = "center",
                add_hline = True,
                alpha_hline = 0.5,
                col_hline = "black",
                linestyle_hline="dashed",
                size_hline = 0.5,
                add_vline = True,
                alpha_vline = 0.5,
                col_vline = "black",
                linestyle_vline ="dashed",
                size_vline = 0.5,
                ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Correspondence Analysis - Graph of column variables
    -------------------------------------------------------------

    Description
    -----------
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables. `fviz_ca_col` provides plotnine based elegant visualization of CA outputs from Python functions.

    Usage
    -----
    ```python
    >>> fviz_ca_col(self,
                 axis=[0,1],
                 x_lim= None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 palette = None,
                 text_type = "text",
                 marker = "o",
                 point_size = 1.5,
                 text_size = 8,
                 add_grid =True,
                 legend_title = None,
                 col_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_hline = True,
                 add_vline = True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 lim_cos2 = None,
                 lim_contrib = None,
                 repel=False,
                 ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class CA

    `axis` : a numeric list or vector of length 2 specifying the dimensions to be plotted, default = [0,1]

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a x_label is chosen).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `color` : a color for the active columns points (by default = "black")

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.
    
    `gradient_cols` :  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `palette` :  a list or tuple specifying the color palette to be used for coloring or filling by groups.
    
    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `marker` : the marker style (by default = "o").
    
    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `legend_title` : a string corresponding to the title of the legend (by default = None).

    `col_sup` : a boolean to either add or not supplementary columns points (by default = True).

    `color_sup` : a color for the supplementary columns points (by default = "blue").

    `marker_sup` :  a marker style for the supplementary columns points (by default = "^").

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `lim_cos2` : a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib` : a numeric specifying the relative contribution limit (by default = None),

    `repel` : a boolean, whether to avoid overplotting text labels or not (by default == False)

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load children dataset
    >>> from scientisttools import load_children
    >>> children  = load_children()
    >>> from scientisttools import CA, fviz_ca_col
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Columns factor map
    >>> p = fviz_ca_col(res_ca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize coordinates
    coord = self.col_.coord

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.col_.cos2.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = self.col_.contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")

    if isinstance(col_col,str):
        if col_col == "cos2":
            c = self.col_.cos2.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                 legend_title = "Cos2"
        elif col_col == "contrib":
            c = self.col_.contrib.iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(col_col,np.ndarray):
        c = np.asarray(col_col)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(col_col, "labels_"):
        c = [str(x+1) for x in col_col.labels_]
        if legend_title is None:
            legend_title = "Cluster"

        index = np.unique(c).tolist()
        if palette is None:
            palette = [x for x in list_colors if x not in [col_col,col_col_sup]][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(col_col,str) and col_col in ["cos2","contrib"]) or (isinstance(col_col,np.ndarray)):
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(colour=c),alpha=alpha_col,fill=fill_col,shape=shape_col,size=point_size_col,stroke=stroke_col,show_legend=False)
                   + pn.scale_color_gradient2(low = gradient_cols[0],mid = gradient_cols[1],high = gradient_cols[2],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type_col,repel,mapping=pn.aes(colour=c),size=text_size_col,va=va,ha=ha)
    elif hasattr(col_col, "labels_"):
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),alpha=alpha_col,fill=fill_col,shape=shape_col,size=point_size_col,stroke=stroke_col,show_legend=False)
                   + pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type_col,repel,mapping=pn.aes(color=c),size=text_size_col,va=va,ha=ha)
        p = p + pn.scale_color_manual(values=palette) 
    else:
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),alpha=alpha_col,color=col_col,fill=fill_col,shape=shape_col,size=point_size_col,stroke=stroke_col,show_legend=False)
        if "text" in geom:
            p = p + text_label(text_type_col,repel,color=col_col,size=text_size_col,va=va,ha=ha)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##Add supplementary columns coordinates
    #----------------------------------------------------------------------------------------------------------------------------------------
    if col_sup:
        if hasattr(self, "col_sup_"):
            sup_coord = self.col_sup_.coord
            if "point" in geom:
                p  = p + pn.geom_point(sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),alpha=alpha_col_sup,color=col_col_sup,fill=fill_col_sup,shape=shape_col_sup,size=point_size_col_sup,stroke=stroke_col_sup,show_legend=False)
            if "text" in geom:
                p  = p + text_label(text_type_col,repel,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),color=col_col_sup,size=text_size_col_sup,va=va,ha=ha)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##Add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------- 
    if title is None:
        title = "Columns points - CA"
    p = fviz_add(p,self,axis,x_label,y_label,title,x_lim,y_lim,add_hline,alpha_hline,col_hline,linestyle_hline,size_hline,add_vline,alpha_vline,col_vline,linestyle_vline,size_vline,add_grid,ggtheme)     
    
    return p

def fviz_ca_biplot(self,
                   axis = [0,1],
                   geom_row = ["point","text"],
                   repel_row = True,
                   geom_col = ["point","text"],
                   repel_col = True,
                   x_lim = None,
                   y_lim = None,
                   x_label = None,
                   y_label = None,
                   title = None,
                   alpha_row = 1,
                   col_row = "black",
                   fill_row = None,
                   shape_row = "o",
                   point_size_row = 1.5,
                   text_size_row = 8,
                   stroke_row = 0.5,
                   text_type_row = "text",
                   alpha_col = 1,
                   col_col = "blue",
                   fill_col = None,
                   shape_col = "^", 
                   point_size_col = 1.5,
                   text_size_col = 8,
                   stroke_col = 0.5,
                   text_type_col = "text",
                   habillage = None,
                   palette = None,
                   add_ellipses = False, 
                   ellipse_type = "t",
                   confint_level = 0.95,
                   geom_ellipse = "polygon",
                   row_sup = True,
                   alpha_row_sup = 1,
                   col_row_sup = "red",
                   fill_row_sup = None,
                   shape_row_sup = ">",
                   point_size_row_sup = 1.5,
                   text_size_row_sup = 8,
                   stroke_row_sup = 0.5,
                   text_type_row_sup = "text",
                   col_sup = True,
                   alpha_col_sup = 1,
                   col_col_sup = "darkblue",
                   fill_col_sup = None,
                   shape_col_sup = "v", 
                   point_size_col_sup = 1.5,
                   text_size_col_sup = 8,
                   stroke_col_sup = 0.5,
                   text_type_col_sup = "text",
                   quali_sup = True,
                   alpha_quali_sup = 1,
                   col_quali_sup = "darkblue",
                   fill_quali_sup = None,
                   shape_quali_sup = "v", 
                   point_size_quali_sup = 1.5,
                   text_size_quali_sup = 8,
                   stroke_quali_sup = 0.5,
                   text_type_quali_sup = "text",
                   add_grid = True,
                   ha_row = "center",
                   va_row = "center",
                   ha_col = "center",
                   va_col = "center",
                   add_hline = True,
                   alpha_hline = 0.5,
                   col_hline = "black",
                   size_hline = 0.5,
                   linestyle_hline = "dashed",
                   add_vline = True,
                   alpha_vline = 0.5,
                   col_vline = "black",
                   size_vline = 0.5,
                   linestyle_vline = "dashed",
                   ggtheme = pn.theme_minimal()) ->pn:
    """
    Visualize Correspondence Analysis - Biplot of row and columns variables
    -----------------------------------------------------------------------

    Parameters
    ----------
    see `fviz_ca_row`, `fviz_ca_col`.
    
    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load children dataset
    >>> from scientisttools import load_children
    >>> children  = load_children()
    >>> from scientisttools import CA, fviz_ca_biplot
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Biplot
    >>> p = fviz_ca_biplot(res_ca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class CA
    if self.model_ != "ca":
        raise ValueError("'self' must be an object of class CA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    #Initialize coordinates
    row_coord, col_coord = self.row_.coord, self.col_.coord

    #add supplementary categorical variables
    if self.quali_sup is not None:
        X_quali_sup = self.call_.Xtot.loc[:,self.call_.quali_sup]
        if self.row_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_.row_sup)
        row_coord = pd.concat([row_coord,X_quali_sup],axis=1)

    #Initialize plot
    p = pn.ggplot()
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Add rows factor coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if habillage is None:
        if "point" in geom_row:
            p = p + pn.geom_point(data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label = row_coord.index),alpha=alpha_row,color=col_row,fill=fill_row,shape=shape_row,size=point_size_row,stroke=stroke_row,show_legend=False)
        if "text" in geom_row:
            p = p + text_label(text_type_row,repel_row,data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_coord.index),color=col_row,size=text_size_row,ha=ha_row,va=va_row)
    else:
        if habillage not in row_coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        
        index = row_coord[habillage].unique().tolist()
        if palette is None:
            palette = [x for x in list_colors if x not in [col_row,col_row_sup,col_quali_sup,col_col,col_col_sup]][:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
        
        if "point" in geom_row:
            p = p + pn.geom_point(data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,fill=habillage),alpha=alpha_row,shape=shape_row,size=point_size_row,stroke=stroke_row)
        if "text" in geom_row:
            p = p + text_label(text_type_row,repel_row,data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,label=row_coord.index),size=text_size_row,ha=ha_row,va=va_row)
        
        if add_ellipses:
            p = p + pn.stat_ellipse(data=row_coord,geom=geom_ellipse,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
            
    #Add supplementary rows factor coordinates
    if row_sup:
        if hasattr(self, "row_sup_"):
            row_sup_coord = self.row_sup_.coord
            if "point" in geom_row:
                p = p + pn.geom_point(row_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),alpha=alpha_row_sup,color = col_row_sup,fill=fill_row_sup,shape=shape_row_sup,size=point_size_row_sup,stroke=stroke_row_sup,show_legend=False)
            if "text" in geom_row:
                p = p + text_label(text_type_row_sup,repel_row,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),color = col_row_sup,size=text_size_row_sup,ha=ha_row,va=va_row)
    
    # Add supplementary qualitatives coordinates
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_.coord
            if "point" in geom_row:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),alpha=alpha_quali_sup,color=col_quali_sup,fill=fill_quali_sup,shape=shape_quali_sup,size=point_size_quali_sup,stroke=stroke_quali_sup,show_legend=False)
            if "text" in geom_row:
                p = p + text_label(text_type_quali_sup,repel_row,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),color=col_quali_sup,size=text_size_quali_sup,ha=ha_row,va=va_row)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Add columns coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "point" in geom_col:
        p = p + pn.geom_point(data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label = col_coord.index),alpha=alpha_col,color=col_col,fill=fill_col,shape=shape_col,size=point_size_col,stroke=stroke_col,show_legend=False)
    if "text" in geom_col:
        p = p + text_label(text_type_col,repel_col,data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_coord.index),color=col_col,size=text_size_col,ha=ha_col,va=va_col)
    
    #add supplementary columns
    if col_sup:
        if hasattr(self, "col_sup_"):
            col_sup_coord = self.col_sup_.coord
            if "point" in geom_col:
                p  = p + pn.geom_point(col_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_coord.index),alpha=alpha_col_sup,color=col_col_sup,fill=fill_col_sup,shape=shape_col_sup,size=point_size_col_sup,stroke=stroke_col_sup,show_legend=False)
            if "text" in geom_col:
                p  = p + text_label(text_type_col_sup,repel_col,data=col_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_coord.index),color=col_col_sup,size=text_size_col_sup,ha=ha_col,va=va_col)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##Add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------        
    if title is None:
        title = "CA - Biplot"
    p = fviz_add(p,self,axis,x_label,y_label,title,x_lim,y_lim,add_hline,alpha_hline,col_hline,linestyle_hline,size_hline,add_vline,alpha_vline,col_vline,linestyle_vline,size_vline,add_grid,ggtheme)
    
    return p

def fviz_ca(self,choice="biplot",**kwargs)->pn:
    """
    Draw the Correspondence Analysis (CA) graphs
    --------------------------------------------

    Description
    -----------
    Draw the Correspondence Analysis (CA) graphs.

    Usage
    -----
    ```python
    >>> fviz_ca(self,choice=("row","col","biplot"),**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class CA

    `choice` : the graph to plot
        * 'row' for the row points factor map
        * 'col' for the columns points factor map
        * 'biplot' for biplot of row and columns factor map

    `**kwargs` : further arguments passed to or from other methods

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load children dataset
    >>> from scientisttools import load_children
    >>> children  = load_children()
    >>> from scientisttools import CA, fviz_ca
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Rows factor map
    >>> p1 = fviz_ca(res_ca,choice="row")
    >>> print(p1)
    >>> # Columns factor map
    >>> p2 = fviz_ca(res_ca,choice="col")
    >>> print(p2)
    >>> # Biplot
    >>> p3 = fviz_ca(res_ca,choice="biplot")
    >>> print(p3)
    ```
    """
    # Check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    if choice not in ["row","col","biplot"]:
        raise ValueError("'choice' should be one of 'row', 'col', 'biplot'")

    if choice == "row":
        return fviz_ca_row(self,**kwargs)
    elif choice == "col":
        return fviz_ca_col(self,**kwargs)
    elif choice == "biplot":
        return fviz_ca_biplot(self,**kwargs)