# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from scientisttools.plot.fviz_add import text_label, fviz_add,list_colors

def fviz_mca_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 title =None,
                 x_label = None,
                 y_label = None,
                 color ="black",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title=None,
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_ellipses=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 habillage = None,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 repel=False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize (specific) Multiple Correspondence Analysis (MCA/SpecificMCA) - Graph of individuals
    ----------------------------------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_ind() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for individuals.

    Usage
    -----
    ```python
    >>> fviz_mca_ind(self,
                    axis = [0,1],
                    x_lim = None,
                    y_lim = None,
                    title = None,
                    x_label = None,
                    y_label = None,
                    color = "black",
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
                    legend_title = None,
                    add_grid = True,
                    ind_sup = True,
                    color_sup = "blue",
                    marker_sup = "^",
                    add_ellipses = False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
                    habillage = None,
                    add_hline = True,
                    add_vline = True,
                    ha = "center",
                    va = "center",
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style = "dashed",
                    repel=False,
                    lim_cos2 = None,
                    lim_contrib = None,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class MCA or SpecificMCA

    see fviz_pca_ind

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA, fviz_mca_ind
    >>> res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> p = fviz_mca_ind(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA/SpecificMCA
    if self.model_ not in ["mca","specificmca"]:
        raise ValueError("'self' must be an object of class MCA or SpecificMCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize
    coord = self.ind_["coord"]

    # Add Categorical supplementary Variables
    coord = pd.concat([coord, self.call_["X"]],axis=1)

     ################ Add supplementary quantitatives columns
    if self.quanti_sup is not None:
        X_quanti_sup = self.call_["Xtot"].loc[:,self.call_["quanti_sup"]].astype("float")
        if self.ind_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat([coord,X_quanti_sup],axis=1)
    
    ################ Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query(f"contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")
    
    if isinstance(color,str):
        if color == "cos2":
            c = self.ind_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.ind_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns:
            if not np.issubdtype(coord[color].dtype, np.number):
                raise TypeError("'color' must me a numeric variable.")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        c= np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if habillage is None :
        # Using cosine and contributions
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns.tolist()]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
            c = [str(x+1) for x in color.labels_]
            if legend_title is None:
                legend_title = "Cluster"
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(color=c),size=point_size)+
                         pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom:
                p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipses:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(data=sup_coord,
                                    mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                    color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                    color=color_sup,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - "+self.model_.upper()
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

# Graph for categories
def fviz_mca_mod(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 legend_title=None,
                 marker = "o",
                 add_grid =True,
                 quali_sup = True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 corrected = False,
                 repel=False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize (Specific) Multiple Correspondence Analysis (MCA/SpecificMCA) - Graph of categories
    ---------------------------------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_mod() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for categories.

    Usage
    -----
    ```python
    >>> fviz_mca_mod(self,
                    axis=[0,1],
                    x_lim=None,
                    y_lim=None,
                    x_label = None,
                    y_label = None,
                    title =None,
                    color ="black",
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    legend_title=None,
                    marker = "o",
                    add_grid =True,
                    quali_sup = True,
                    color_sup = "blue",
                    marker_sup = "^",
                    add_hline = True,
                    add_vline=True,
                    ha="center",
                    va="center",
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style ="dashed",
                    corrected = False,
                    repel=False,
                    lim_cos2 = None,
                    lim_contrib = None,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class MCA or SpecificMCA

    see fviz_pca_ind()

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA, fviz_mca_mod
    >>> res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> p = fviz_mca_mod(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA/SpecificMCA
    if self.model_ not in ["mca","specificmca"]:
        raise TypeError("'self' must be an object of class MCA or SpecificMCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Corrected 
    if corrected:
        coord = self.var_["corrected_coord"]
    else:
        coord = self.var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

    if isinstance(color,str):
        if color == "cos2":
            c = self.var_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.var_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c= np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1], name = legend_title))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),size=point_size)+
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Add supplementary categories
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            var_sup_coord = self.quali_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(var_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index.tolist()),
                                      color=color_sup,size=point_size,shape=marker_sup)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=var_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index.tolist()),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,data=var_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Qualitatives variables categories - "+self.model_.upper()
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    p = p + pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p + pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mca_var(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title=None,
                 color="black",
                 color_sup = "blue",
                 color_quanti_sup = "red",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 marker="o",
                 marker_sup = "^",
                 marker_quanti_sup = ">",
                 legend_title = None,
                 text_type="text",
                 add_quali_sup=True,
                 add_quanti_sup = True,
                 add_grid =True,
                 add_hline = True,
                 add_vline =True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    """
    Visualize (specific) Multiple Correspondence Analysis (MCA/SpecificMCA) - Graph of variables
    --------------------------------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_var() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for variables.

    Usage
    -----
    ```python
    >>> fviz_mca_var(self,
                    axis=[0,1],
                    x_lim=None,
                    y_lim=None,
                    x_label = None,
                    y_label = None,
                    title=None,
                    color="black",
                    color_sup = "blue",
                    color_quanti_sup = "red",
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    point_size = 1.5,
                    text_size = 8,
                    marker="o",
                    marker_sup = "^",
                    marker_quanti_sup = ">",
                    legend_title = None,
                    text_type="text",
                    add_quali_sup=True,
                    add_quanti_sup = True,
                    add_grid =True,
                    add_hline = True,
                    add_vline =True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    repel=False,
                    ggtheme=pn.theme_minimal()) 
    ```

    Parameters
    ----------
    `self` : an object of class MCA or SpecificMCA

    see fviz_pca_ind

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA, fviz_mca_var
    >>> res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> p = fviz_mca_var(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA/SpecificMCA
    if self.model_ not in ["mca","specificmca"]:
        raise TypeError("'self' must be an object of class MCA or SpecificMCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize
    coord = self.var_["eta2"]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

    if isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    # Using cosine and contributions
    if isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low=gradient_cols[0],high=gradient_cols[2],mid=gradient_cols[1],name=legend_title))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if add_quanti_sup:
        if hasattr(self, "quanti_sup_"):
            quant_sup_cos2 = self.quanti_sup_["cos2"]
            if "point" in geom:
                p = p + pn.geom_point(quant_sup_cos2,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quant_sup_cos2.index),
                                      color = color_quanti_sup,shape = marker_quanti_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quant_sup_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quant_sup_cos2.index),
                                       color=color_quanti_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quant_sup_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quant_sup_cos2.index),
                                       color = color_quanti_sup,size=text_size,va=va,ha=ha)
    
    if add_quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_eta2 = self.quali_sup_["eta2"]
            if "point" in geom:
                p = p + pn.geom_point(quali_sup_eta2,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_eta2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_eta2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Graphe of variables - "+self.model_.upper()
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    p = p + pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p + pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mca_biplot(self,
                    axis=[0,1],
                    x_lim = None,
                    y_lim = None,
                    x_label = None,
                    y_label = None,
                    title = None,
                    ind_color ="black",
                    mod_color ="blue",
                    ind_geom = ["point","text"],
                    mod_geom = ["point","text"],
                    ind_point_size = 1.5,
                    mod_point_size = 1.5,
                    ind_text_size = 8,
                    mod_text_size = 8,
                    ind_text_type = "text",
                    mod_text_type = "text",
                    ind_marker = "o",
                    mod_marker = "o",
                    add_grid =True,
                    corrected = False,
                    ind_sup=True,
                    ind_sup_color = "red",
                    ind_sup_marker = "^",
                    add_ellipses=False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
                    quali_sup = True,
                    quali_sup_color = "green",
                    quali_sup_marker = "^",
                    habillage = None,
                    add_hline = True,
                    add_vline=True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    repel=False,
                    ggtheme=pn.theme_minimal())->pn:
    """
    Visualize (specific) Multiple Correspondence Analysis (MCA/SpecificMCA) - Biplot of individuals and categories
    --------------------------------------------------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca_biplot() provides plotnine based elegant visualization of MCA/SpecificMCA outputs for individuals and categories.

    Parameters
    ----------
    `self` : an object of class MCA or SpecificMCA

    see fviz_pca_biplot()

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA, fviz_mca_biplot
    >>> res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> p = fviz_mca_biplot(res_mca)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MCA/SpecificMCA
    if self.model_ not in ["mca","specificmca"]:
        raise TypeError("'self' must be an object of class MCA or SpecificMCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Initialize
    ind_coord = self.ind_["coord"]

    # Add Categorical supplementary Variables
    ind_coord = pd.concat([ind_coord, self.call_["X"]],axis=1)

    ################ Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
        ind_coord = pd.concat([ind_coord,X_quali_sup],axis=1)
    
    # Initialize
    p = pn.ggplot()
    
    ####################################################################################################################
    # Individuals
    ####################################################################################################################
    if habillage is None :
        if "point" in ind_geom:
            p = p + pn.geom_point(data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),
                                  color=ind_color,shape=ind_marker,size=ind_point_size,show_legend=False)
        if "text" in ind_geom:
            if repel :
                p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_coord.index),
                                   color=ind_color,size=ind_text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','color': ind_color,'lw':1.0}})
            else:
                p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=ind_coord.index),
                                   color=ind_color,size=ind_text_size,va=va,ha=ha)
    else:
        if habillage not in ind_coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in ind_geom:
            p = p + pn.geom_point(data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color = habillage,linetype = habillage),
                                  size=ind_point_size)
        if "text" in ind_geom:
            if repel:
                p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color=habillage,label=ind_coord.index),
                                   size=ind_text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(ind_text_type,data=ind_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",color=habillage,label=ind_coord.index),
                                   size=ind_text_size,va=va,ha=ha)
        
        if add_ellipses:
            p = p + pn.stat_ellipse(data=ind_coord,geom=geom_ellipse,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",fill=habillage),
                                    type = ellipse_type,alpha = 0.25,level=confint_level)
    # Adding supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            sup_coord = self.ind_sup_["coord"]
            if "point" in ind_geom:
                p = p + pn.geom_point(data=sup_coord,
                                    mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                    color = ind_sup_color,shape = ind_sup_marker,size=ind_point_size)
            if "text" in ind_geom:
                if repel:
                    p = p + text_label(ind_text_type,data=sup_coord,
                                    mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                    color=ind_sup_color,size=ind_text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': ind_sup_color,'lw':1.0}})
                else:
                    p = p + text_label(ind_text_type,data=sup_coord,
                                    mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color = ind_sup_color,size=ind_text_size,va=va,ha=ha)

    #########################################################################################################################################
    # Categories informations
    #########################################################################################################################################
    # Corrected 
    if corrected:
        mod_coord = self.var_["corrected_coord"]
    else:
        mod_coord = self.var_["coord"]
    
    if "point" in mod_geom:
        p = p + pn.geom_point(data=mod_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),
                              color=mod_color,shape=mod_marker,size=mod_point_size,show_legend=False)
    if "text" in mod_geom:
        if repel :
            p = p + text_label(mod_text_type,data=mod_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_coord.index),
                               color=mod_color,size=mod_text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(mod_text_type,data=mod_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_coord.index),
                               color=mod_color,size=mod_text_size,va=va,ha=ha)
        
    # Add supplementary categories
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            var_sup_coord = self.quali_sup_["coord"]
            if "point" in mod_geom:
                p = p + pn.geom_point(var_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index.tolist()),
                                      color=quali_sup_color,size=mod_point_size,shape=quali_sup_marker)
            if "text" in mod_geom:
                if repel:
                    p = p + text_label(mod_text_type,data=var_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index.tolist()),
                                       color=quali_sup_color,size=mod_text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(mod_text_type,data=var_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index),
                                       color=quali_sup_color,size=mod_text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = self.model_.upper()+" - Biplot"
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_mca(self,choice="biplot",**kwargs)->pn:
    """
    Visualize (specific) Multiple Correspondence Analysis (MCA/SpecificMCA)
    -----------------------------------------------------------------------

    Description
    -----------
    Multiple Correspondence Analysis (MCA) is an extension of simple CA to analyse a data table containing more than two categorical variables. fviz_mca() provides plotnine-based elegant visualization of MCA/SpecificMCA outputs.

        * fviz_mca_ind(): Graph of individuals
        * fviz_mca_var(): Graph of categories
        * fviz_mca_quali_var(): Graph of variables
        * fviz_mca_biplot(): Biplot of individuals and categories
    
    Usage
    -----
    ```
    >>> fviz_mca(self,choice = ("ind","var","quali_var","biplot"))
    ```

    Parameters
    ----------
    `self` : an object of class MCA, SpecificMCA

    `choice` : The element to subset. Possible values are :
        * "ind" for the individuals graphs
        * "var" for the categories graphs
        * "quali_var" for the variables graphs
        * "biplot" for both individuals and categories graphs

    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if self is an object of class MCA/SpecificMCA
    if self.model_ not in ["mca","specificmca"]:
        raise TypeError("'self' must be an object of class MCA or SpecificMCA")
    
    if choice not in ["ind","var","quali_var","biplot"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'quali_var','biplot'")
    
    if choice == "ind":
        return fviz_mca_ind(self,**kwargs)
    elif choice == "mod":
        return fviz_mca_mod(self,**kwargs)
    elif choice == "var":
        return fviz_mca_var(self,**kwargs)
    elif choice == "biplot":
        return fviz_mca_biplot(self,**kwargs)