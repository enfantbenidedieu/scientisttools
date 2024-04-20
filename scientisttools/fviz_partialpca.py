# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from .text_label import text_label
from .gg_circle import gg_circle


def fviz_partialpca_ind(self,
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
                 ind_sup = True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_grid =True,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 lim_cos2 = None,
                 lim_contrib = None,
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Draw individuals Factor Map - Partial PCA
    -----------------------------------------

    Description
    -----------


    Parameters
    ----------


    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    
    """
    
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an instance of class PartialPCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")

    coord = self.ind_["coord"]

    ##### Add initial data
    coord = pd.concat((coord,self.call_["Xtot"]),axis=1)

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
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns.tolist()]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],
                                                name = legend_title)
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '->',"lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),size=point_size,show_legend=False)+
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
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    ############################## Add supplementary individuals informations
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color=color_sup,size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color = color_sup,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    if title is None:
        title = "Individuals factor map - Partial PCA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
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

def fviz_partialpca_var(self,
                 axis=[0,1],
                 x_label= None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["arrow","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
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
                 color_circle = "gray",
                 arrow_angle=10,
                 arrow_length =0.1,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    
    
    """
    
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    coord = self.var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("'lim_contrib' must be a float or an integer.")

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
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                     pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if "text" in geom:
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
        title = "Variables factor map - Partial PCA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    # Add theme
    p = p + ggtheme

    return p

def fviz_partialpca(self,choice="ind",**kwargs)->pn:
    """
    Draw 

    Description
    -----------


    Parameters
    ----------

    Return
    ------
    a plotnine

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")

    if choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'ind', 'var'")

    if choice == "ind":
        return fviz_partialpca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_partialpca_var(self,**kwargs)