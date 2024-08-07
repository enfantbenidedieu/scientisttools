# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from .text_label import text_label
from .gg_circle import gg_circle

def fviz_mfa_ind(self,
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
                 marker = "o",
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 legend_title=None,
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 habillage = None,
                 palette = None,
                 quali_sup = True,
                 color_quali_sup = "red",
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
    Visulize Multiple Factor Analysis (MFA) - Graph of individuals
    --------------------------------------------------------------

    Description
    -----------
    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (quantitative and/or qualitative) structured into groups. fviz_mfa_ind() provides plotnine-based elegant visualization of MFA individuals outputs.

    Usage
    -----
    ```python
    >>> fviz_mfa_ind(self,
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
                    marker = "o",
                    add_grid =True,
                    ind_sup=True,
                    color_sup = "blue",
                    marker_sup = "^",
                    legend_title=None,
                    add_ellipse=False, 
                    ellipse_type = "t",
                    confint_level = 0.95,
                    geom_ellipse = "polygon",
                    habillage = None,
                    palette = None,
                    quali_sup = True,
                    color_quali_sup = "red",
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
                    ggtheme=pn.theme_minimal()) 
    ```

    Parameters
    ----------
    `self` : an object of class MFA, MFAQUAL, MFAMIX, MFACT

    see fviz_pca_ind

    Returns
    -------
    a plotnine
    
    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierlab@gmail.com

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, fviz_mfa_ind
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> p = fviz_mfa_ind(res_mfa)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MFA, MFAQUAL, MFAMIX, MFACT
    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX, MFACT")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    # Initialize coordinates
    coord = self.ind_["coord"]

    # Concatenate with actives columns
    coord = pd.concat((coord,self.call_["X"]),axis=1)

    #### Add supplementary columns
    if self.num_group_sup is not None:
        not_in = [x for x in self.call_["Xtot"].columns if x not in self.call_["X"].columns]
        X_group_sup = self.call_["Xtot"][not_in]
        if hasattr(self,"ind_sup"):
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=self.call_["ind_sup"])
        coord = pd.concat((coord,X_group_sup),axis=1)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float) or isinstance(lim_cos2,int):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = self.ind_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.ind_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise TypeError("'color' must me a numeric variable")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    if habillage is None :        
        # Using cosine and contributions
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns.tolist()]) or isinstance(color,np.ndarray):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size)+ 
                        pn.scale_color_gradient2(midpoint=np.mean(c),low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
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
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom:
                p = p + pn.geom_point(color=color,shape=marker,size=point_size)
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if not hasattr(self, "quali_var_sup_"):
            raise ValueError(f"{habillage} not in DataFrame")
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        # Add ellipse
        if add_ellipse:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
        
        # Change color
        if palette is not None:
            p = p + pn.scale_color_manual(values=palette)
            
    # Add supplementary individuals
    if ind_sup:
        if not hasattr(self, "ind_sup_"):
            raise ValueError("No supplementary individuals")
        
        sup_coord = self.ind_sup_["coord"]
        if "point" in geom:
            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color = color_sup,shape = marker_sup,size=point_size)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color = color_sup,size=text_size,va=va,ha=ha)
    ### Add supplementary qualitatives
    if quali_sup:
        if not hasattr(self, "quali_var_sup_"):
            raise ValueError("No supplementary qualitatives")
        if habillage is None:
            quali_var_sup_coord = self.quali_var_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(quali_var_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index.tolist()),
                                      color=color_quali_sup,size=point_size)
                
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_var_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index.tolist()),
                                       color=color_quali_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,data=quali_var_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index.tolist()),
                                       color = color_quali_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Individuals factor map - MFA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mfa_var(self,
                 axis=[0,1],
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="group",
                 geom = ["arrow","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 marker = "^",
                 point_size=1.5,
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "red",
                 linestyle_sup="dashed",
                 legend_title = None,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 add_circle = True,
                 arrow_angle=10,
                 arrow_length =0.1,
                 lim_cos2 = None,
                 lim_contrib = None,
                 legend = "bottom",
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Multiple Factor Analysis (MFA) - Graph of variables
    -------------------------------------------------------------

    Description
    -----------
    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (quantitative and/or qualitative) structured into groups. fviz_mfa_var() provides plotnine-based elegant visualization of MFA quantitative variables outputs.

    Usage
    -----
    ```
    >>> fviz_mfa_var(self,
                    axis=[0,1],
                    x_label = None,
                    y_label = None,
                    title =None,
                    color ="group",
                    geom = ["arrow","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    marker = "^",
                    point_size=1.5,
                    text_type = "text",
                    text_size = 8,
                    add_grid =True,
                    quanti_sup=True,
                    color_sup = "red",
                    linestyle_sup="dashed",
                    legend_title = None,
                    add_hline = True,
                    add_vline=True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    add_circle = True,
                    arrow_angle=10,
                    arrow_length =0.1,
                    lim_cos2 = None,
                    lim_contrib = None,
                    legend = "bottom",
                    ggtheme=pn.theme_minimal()) 
    ```

    Parameters
    ----------
    `self` : an object of class MFA, MFAMIX

    see fviz_pca_var

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, fviz_mfa_var
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> p = fviz_mfa_var(res_mfa)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MFA, MFAMIX
    if self.model_ not in ["mfa","mfamix"]:
        raise TypeError("'self' must be an instance of class MFA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    coord = self.quanti_var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float) or isinstance(lim_cos2,int):
            lim_cos2 = float(lim_cos2)
            cos2 = self.quanti_var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.quanti_var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    if isinstance(color,str):
        if color == "cos2":
            c = self.quanti_var_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.quanti_var_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        if "point" in geom:
            p = p + pn.geom_point(color=c,shape=marker,size=point_size)
        # Add gradients colors
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                    arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+
                    pn.scale_color_gradient2(midpoint=np.mean(c),low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        # Point
        if "point" in geom:
            p = p + pn.geom_point(color=c,size=point_size)
        # Add arrow
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        # Text
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif color == "group":
        if legend_title is None:
            legend_title = "Groups"
        ###### Add group sup
        if quanti_sup:
            if hasattr(self, "quanti_var_sup_"):
                sup_coord = self.quanti_var_sup_["coord"]
                coord = pd.concat([coord,sup_coord],axis=0)

        # Reset and merge
        coord = coord.reset_index().rename(columns={"index" : "variable"})
        coord = pd.merge(coord,self.group_label_,on=["variable"]).set_index("variable")

        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color="group name"),size=point_size)
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color="group name"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color="group name"),size=text_size,va=va,ha=ha)
        # Set label position
        p = p + pn.theme(legend_position=legend)
    else:
        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size)
        if "arrow" in geom:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if "text" in geom:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Add supplementary continuous variables
    if quanti_sup:
        if not hasattr(self, "quanti_var_sup_"):
            raise ValueError("No supplementary quantitatives variables")
        
        sup_coord = self.quanti_var_sup_["coord"]
        if "arrow" in geom:
            p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
        if "text" in geom:
            p  = p + text_label(text_type,data=sup_coord,
                                mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                color=color_sup,size=text_size,va=va,ha=ha)
        
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color="lightgray", fill=None)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Quantitatives variables - MFA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.labs(title=title,x = x_label, y = y_label)
    
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_mfa_group(self,
                   axis=[0,1],
                   x_label = None,
                   y_label = None,
                   x_lim=None,
                    y_lim=None,
                    title =None,
                    color ="red",
                    geom = ["point","text"],
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
                    add_grid =True,
                    group_sup=True,
                    color_sup = "green",
                    marker_sup = "^",
                    legend_title=None,
                    add_hline = True,
                    add_vline=True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    repel=True,
                    lim_cos2 = None,
                    lim_contrib = None,
                    ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Multiple Factor Analysis (MFA) - Graph of variables groups
    --------------------------------------------------------------------

    Description
    -----------
    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (quantitative and/or qualitative) structured into groups. fviz_mfa_group() provides plotnine-based elegant visualization of MFA groups outputs.

    Usage
    -----
    ```python
    >>> fviz_mfa_group(self,
                        axis=[0,1],
                        x_label = None,
                        y_label = None,
                        x_lim=None,
                        y_lim=None,
                        title =None,
                        color ="red",
                        geom = ["point","text"],
                        gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                        point_size = 1.5,
                        text_size = 8,
                        text_type = "text",
                        marker = "o",
                        add_grid =True,
                        group_sup=True,
                        color_sup = "green",
                        marker_sup = "^",
                        legend_title=None,
                        add_hline = True,
                        add_vline=True,
                        ha="center",
                        va="center",
                        hline_color="black",
                        hline_style="dashed",
                        vline_color="black",
                        vline_style ="dashed",
                        repel=True,
                        lim_cos2 = None,
                        lim_contrib = None,
                        ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class MFA, MFAQUAL, MFAMIX, MFACT

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
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, fviz_mfa_group
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> p = fviz_mfa_group(res_mfa)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MFA, MFAQUAL, MFAMIX, MFACT
    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an instance of class MFA, MFAQUAL, MFAMIX, MFACT")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize coordinates
    coord = self.group_["coord"]
    
    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float) or isinstance(lim_cos2,int):
            lim_cos2 = float(lim_cos2)
            cos2 = self.group_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.group_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

    if isinstance(color,str):
        if color == "cos2":
            c = self.group_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.group_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size)+ 
                        pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
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
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
            
    # Add supplementary groups
    if group_sup:
       if self.num_group_sup is not None:
           sup_coord = self.group_["coord_sup"]
           if "point" in geom:
            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                    color = color_sup,shape = marker_sup,size=point_size)
           if "text" in geom:
               if repel:
                   p = p + text_label(text_type,data=sup_coord,
                                      mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
               else:
                   p = p + text_label(text_type,data=sup_coord,
                                      mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                       color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Variable groups - MFA"
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mfa_axes(self,
                 axis=[0,1],
                 x_label = None,
                 y_label = None,
                 title =None,
                 color="group",
                 color_circle = "lightgray",
                 geom = ["arrow","text"],
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 legend_title = None,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 add_circle = True,
                 arrow_angle=10,
                 arrow_length =0.1,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Multiple Factor Analysis (MFA) - Grpah of partial axes
    ----------------------------------------------------------------

    Description
    -----------
    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (quantitative and/or qualitative) structured into groups. fviz_mfa_axes() provides plotnine-based elegant visualization of MFA partial axes outputs.

    Usage
    -----
    ```python
    >>> fviz_mfa_axes(self,
                    axis=[0,1],
                    x_label = None,
                    y_label = None,
                    title =None,
                    color="group",
                    color_circle = "lightgray",
                    geom = ["arrow","text"],
                    text_type = "text",
                    text_size = 8,
                    add_grid =True,
                    legend_title = None,
                    add_hline = True,
                    add_vline=True,
                    ha="center",
                    va="center",
                    hline_color="black",
                    hline_style="dashed",
                    vline_color="black",
                    vline_style ="dashed",
                    add_circle = True,
                    arrow_angle=10,
                    arrow_length =0.1,
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class MFA, MFAQUAL, MFAMIX, MFACT

    see fviz_pca_ind

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, fviz_mfa_axes
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> p = fviz_mfa_axes(res_mfa)
    >>> print(p)
    ```
    """
    # Check if self is an object of class MFA, MFAQUAL, MFAMIX, MFACT
    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an instance of class MFA, MFAQUAL, MFAMIX, MFACT")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Create coordinates
    coord = pd.DataFrame().astype("float")
    for grp in self.partial_axes_["coord"].columns.get_level_values(0).unique().tolist():
        data = self.partial_axes_["coord"][grp].T
        data.insert(0,"group",grp)
        data.index = [x+"."+str(grp) for x in data.index]
        coord = pd.concat([coord,data],axis=0)
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

    if color == "group":
        if legend_title is None:
            legend_title = "Groups"
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color="group"))
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color="group"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color="group"),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point()
        if "arrow" in geom:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if "text" in geom :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=color_circle, fill=None)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Partial axes - MFA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_mfa_mod(self,
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
                 marker = "o",
                 add_grid =True,
                 quali_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 legend_title=None,
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
    Visualize Multiple Factor Analysis (MFA) - Graph of qualitative variable categories
    -----------------------------------------------------------------------------

    Description
    -----------
    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (quantitative and/or qualitative) structured into groups. fviz_mfa_mod() provides plotnine-based elegant visualization of MFA qualitative variables outputs.

    Usage
    -----
    ```python
    >>> fviz_mfa_mod(self,
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
                    marker = "o",
                    add_grid =True,
                    quali_sup=True,
                    color_sup = "blue",
                    marker_sup = "^",
                    legend_title=None,
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
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class MFAQUAL, MFAMIX

    see fviz_pca_ind

    Returns
    -------
    a plotnine
    
    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierlab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> group_name = ["desc","desc2","symptom","eat"]
    >>> group = [2,2,5,6]
    >>> group_type = ["s"]+["n"]*3
    >>> num_group_sup = [0,1]
    >>> from scientisttools import MFAQUAL, fviz_mfa_mod
    >>> res_mfaqual = MFAQUAL(group=group,name_group=group_name,group_type=group_type,var_weights_mfa=None,num_group_sup=[0,1],parallelize=True)
    >>> res_mfaqual.fit(poison)
    >>> p = fviz_mfa_mod(res_mfaqual)
    ```
    """
    # Check if self is an object of class MFAQUAL, MFAMIX
    if self.model_ not in ["mfaqual","mfamix"]:
        raise TypeError("'self' must be an object of class MFAQUAL, MFAMIX")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    # Initialize coordinates
    coord = self.quali_var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float) or isinstance(lim_cos2,int):
            lim_cos2 = float(lim_cos2)
            cos2 = self.quali_var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.quali_var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = self.quali_var_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.quali_var_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size)+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
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
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif color == "group":
        if legend_title is None:
            legend_title = "Groups"
        ###### Add group sup
        if quali_sup:
            if hasattr(self, "quali_var_sup_"):
                sup_coord = self.quali_var_sup_["coord"]
                coord = pd.concat([coord,sup_coord],axis=0)
        
        # Merge between summary quali and group_label
        groups = pd.merge(self.summary_quali_[["categorie","group name"]],self.group_label_,on=["group name"])[["categorie","group name"]]
        
        # Reset index and merge
        coord = coord.reset_index().rename(columns={"index" : "categorie"})
        coord = pd.merge(coord,groups,on=["categorie"]).set_index("categorie")
                
        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        if "point" in geom:
           p = (p + pn.geom_point(pn.aes(color="group name"),shape=marker,size=point_size)+
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color="group name"),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color="group name"),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
            
    ### Add supplementary qualitatives
    if quali_sup:
        if not hasattr(self, "quali_var_sup_"):
            raise ValueError("No supplementary qualitatives variables")
        quali_var_sup_coord = self.quali_var_sup_["coord"]
        if "point" in geom:
            p = p + pn.geom_point(quali_var_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index),
                                    color=color_sup,size=point_size,shape=marker_sup)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,data=quali_var_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,data=quali_var_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index),
                                    color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Qualitative variable categories - MFA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mfa_freq(self,
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
                 marker = "o",
                 add_grid =True,
                 freq_sup=False,
                 color_sup = "blue",
                 marker_sup = "^",
                 legend_title=None,
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
    Visualize Multiple Factor Analysis (MFA) - Graph of frequences
    --------------------------------------------------------------

    Description
    -----------
    Multiple factor analysis (MFA) is used to analyze a data set in which individuals are described by several sets of variables (quantitative and/or qualitative) structured into groups. fviz_mfa_freq() provides plotnine-based elegant visualization of MFA frequences outputs.

    Usage
    -----
    ```python
    >>> fviz_mfa_freq(self,
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
                    marker = "o",
                    add_grid =True,
                    freq_sup=False,
                    color_sup = "blue",
                    marker_sup = "^",
                    legend_title=None,
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
                    ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class MFACT

    see fviz_pca_ind

    Returns
    -------
    a plotnine
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if self is an object of class MFACT
    if self.model_ != "mfact":
        raise TypeError("'self' must be an object of class MFACT")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    # Initialize coordinates
    coord = self.freq_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float) or isinstance(lim_cos2,int):
            lim_cos2 = float(lim_cos2)
            cos2 = self.freq_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.freq_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = self.freq_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = self.freq_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size)+
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif color == "group":
        if legend_title is None:
            legend_title = "Groups"
        ###### Add group sup
        if freq_sup:
            if hasattr(self, "freq_sup_"):
                sup_coord = self.freq_sup_["coord"]
                coord = pd.concat([coord,sup_coord],axis=0)
        
        # Reset index and merge
        coord = coord.reset_index().rename(columns={"index" : "variable"})
        coord = pd.merge(coord,self.group_label_,on=["variable"]).set_index("variable")
                
        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        if "point" in geom:
           p = (p + pn.geom_point(pn.aes(color="group name"),shape=marker,size=point_size)+
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color="group name"),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color="group name"),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
            
    ### Add supplementary qualitatives
    if freq_sup:
        if not hasattr(self, "freq_sup_"):
            raise ValueError("No supplementary frequences")
        freq_sup_coord = self.freq_sup_["coord"]
        if "point" in geom:
            p = p + pn.geom_point(freq_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=freq_sup_coord.index),
                                    color=color_sup,size=point_size,shape=marker_sup)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,data=freq_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=freq_sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,data=freq_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=freq_sup_coord.index),
                                    color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Contingency tables - MFA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
   
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mfa(self,choice="ind",**kwargs):
    """
    Visualize Multiple Factor Analysis (MFA)
    ----------------------------------------

    Description
    -----------
    Plot the graphs for a Multiple Factor Analysis (MFA) with supplementary individuals and supplementary groups.

        * fviz_mfa_ind() : Graph of individuals
        * fviz_mfa_var() : Graph of quantitative variables (= Correlation circle)
        * fviz_mfa_mod() : Graph of qualitative variables
        * fviz_mfa_freq() : Graph of frequences
        * fviz_mfa_group() : Graph of groups
        * fviz_mfa_axes() : Graph of axes

    Usage
    -----
    ```
    >>> fviz_mfa(self,choice=("ind","quanti_var","quali_var","freq","axes","group"),**kwargs)
    ```

    Parameters
    ----------
    'self' : an object of class MFA, MFAQUAL, MFAMIX, MFACT

    `choice` : the element to plot from the output. Possible value are : 
        * 'ind' for the individuals graphs
        * 'quanti_var' for the quantitative variables graphs (= Correlation circle)
        * 'quali_var' for the qualitative variables graphs
        * 'freq' for frequences  graphs
        * 'group' for groups graphs
        * 'axes' for partial axes graphs
    
    `**kwargs` : further arguments passed to or from other methods

    Returns
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    see fviz_mfa_ind, fviz_mfa_var, fviz_mfa_mod, fviz_mfa_freq, fviz_mfa_group, fviz_mfa_axes
    """
    # Check if self is an object of class
    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX, MFACT ")
    
    if choice not in ["ind","quanti_var","quali_var","freq","group","axes"]:
        raise ValueError("'choice' should be one of 'ind', 'quanti_var', 'quali_var', 'freq', 'group', 'axes'")

    if choice == "ind":
        return fviz_mfa_ind(self,**kwargs)
    elif choice == "quanti_var":
        return fviz_mfa_var(self,**kwargs)
    elif choice == "quali_var":
        return fviz_mfa_mod(self,**kwargs)
    elif choice == "freq":
        return fviz_mfa_freq(self,**kwargs)
    elif choice == "group":
        return fviz_mfa_group(self,**kwargs)
    elif choice == "axes":
        return fviz_mfa_axes(self,**kwargs)