# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from .text_label import text_label
from .gg_circle import gg_circle

def fviz_famd_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label=None,
                 y_label=None,
                 title =None,
                 color ="black",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title = None,
                 add_grid =True,
                 color_quali_var = "blue",
                 marker_quali_var = "v",
                 ind_sup=True,
                 color_sup = "darkblue",
                 marker_sup = "^",
                 quali_sup = True,
                 color_quali_sup = "red",
                 marker_quali_sup = ">",
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
                 lim_cos2 = None,
                 lim_contrib = None,
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) individuals graphs
    --------------------------------------------------------------------------


    Return
    ------
    a plotnine

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")
    
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
        X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
        if self.ind_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
        coord = pd.concat([coord,X_quanti_sup],axis=1)
    
    ################ Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
        if self.ind_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
        coord = pd.concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = (self.ind_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2"))
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
        elif color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise TypeError("'color' must me a numeric variable.")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

    if habillage is None :        
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns.tolist()]) or isinstance(color,np.ndarray):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                        pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
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
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle':"-","lw":1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    else:
        if habillage not in coord.columns:
            raise ValueError(f"'{habillage}' not in DataFrame.")
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size)
        if "text":
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipses:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    ############################## Add qualitatives variables
    quali_coord = self.quali_var_["coord"]
    if "point" in geom:
        p = p + pn.geom_point(quali_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_coord.index.tolist()),
                              color = color_quali_var,shape = marker_quali_var,size=point_size)
    if "text" in geom:
        if repel:
            p = p + text_label(text_type,data=quali_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_coord.index.tolist()),
                                color=color_quali_var,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,data=quali_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_coord.index.tolist()),
                               color = color_quali_var,size=text_size,va=va,ha=ha)

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
    
    ############## Add supplementary qualitatives
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index.tolist()),
                                      color = color_quali_sup,shape = marker_quali_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index.tolist()),
                                        color=color_quali_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index.tolist()),
                                        color = color_quali_sup,size=text_size,va=va,ha=ha)
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - FAMD"
    
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim:
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

def fviz_famd_col(self,
                 axis=[0,1],
                 title =None,
                 color ="black",
                 x_label= None,
                 y_label = None,
                 geom = ["arrow", "text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 text_size = 8,
                 legend_title=None,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "blue",
                 linestyle_sup="dashed",
                 add_hline = True,
                 add_vline=True,
                 arrow_length=0.1,
                 arrow_angle=10,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 lim_cos2 = None,
                 lim_contrib = None,
                 add_circle = True,
                 color_circle = "gray",
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) correlation circle graphs
    ---------------------------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    coord = self.quanti_var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.quanti_var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.quanti_var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")
    
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
    
     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
    
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                     arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(angle=arrow_angle,length=arrow_length),color=color)
        if "text" in geom:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if quanti_sup:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_["coord"]
            if "arrow" in geom:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                     arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
            if "text" in geom:
                p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    
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
        title = "Continuous variables factor map - FAMD"
    
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

# Graph for categories
def fviz_famd_mod(self,
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
                 legend_title=None,
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
                 lim_cos2 = None,
                 lim_contrib = None,
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) variables/categories graphs
    -----------------------------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize
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
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
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
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Add supplementary categories
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(data=quali_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                      color=color_sup,size=point_size,shape=marker_sup)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_coord,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Qualitatives variables categories - FAMD"
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
    

def fviz_famd_var(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 title=None,
                 x_label = None,
                 y_label = None,
                 geom = ["point","text"],
                 color_quanti ="black",
                 color_quali = "blue",
                 color_quali_sup = "green",
                 color_quanti_sup = "red",
                 point_size = 1.5,
                 text_size = 8,
                 add_quali_sup = True,
                 add_quanti_sup =True,
                 marker_quanti ="o",
                 marker_quali ="^",
                 marker_quanti_sup = "v",
                 marker_quali_sup = "<",
                 text_type="text",
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
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) variables graphs
    ------------------------------------------------------------------------


    Return
    ------
    a plotnine

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize
    quanti_var_cos2 = self.quanti_var_["cos2"]
    quali_var_eta2 = self.var_["coord"].loc[self.call_["quali"].columns.tolist(),:]
    
    # Initialize
    p = pn.ggplot(data=quanti_var_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_var_cos2.index))
    if "point" in geom:
        p = p + pn.geom_point(color=color_quanti,shape=marker_quanti,size=point_size,show_legend=False)
    
    if "text" in geom:
        if repel :
            p = p + text_label(text_type,color=color_quanti,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color_quanti,size=text_size,va=va,ha=ha)
    
    # Add Qualitatives variables
    if "point" in geom:
        p = p + pn.geom_point(data=quali_var_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_eta2.index.tolist()),
                              color=color_quali,size=point_size,shape=marker_quali)
    if "text" in geom:
        if repel:
            p = p + text_label(text_type,data=quali_var_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_eta2.index.tolist()),
                                color=color_quali,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,data=quali_var_eta2,
                                mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_eta2.index),
                                color=color_quali,size=text_size,va=va,ha=ha)
    
    # Add supplementary continuous variables
    if add_quanti_sup:
        if hasattr(self, "quanti_sup_"):
            quanti_sup_cos2 = self.quanti_sup_["cos2"]
            if "point" in geom:
                p = p + pn.geom_point(data=quanti_sup_cos2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_sup_cos2.index),
                                    color=color_quanti_sup,size=point_size,shape=marker_quanti_sup)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quanti_sup_cos2,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_sup_cos2.index),
                                       color=color_quanti_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quanti_sup_cos2,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_sup_cos2.index),
                                       color=color_quanti_sup,size=text_size,va=va,ha=ha)
    
    # Add supplementary categoricals variables
    if add_quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_eta2 = self.quali_sup_["eta2"]
            if "point" in geom:
                p = p + pn.geom_point(data=quali_sup_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                    color=color_quali_sup,size=point_size,shape=marker_quali_sup)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color=color_quali_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color=color_quali_sup,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Graphe of variables - FAMD"
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim:
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


def fviz_famd(self,choice="ind",**kwargs) -> pn:
    """
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) graphs
    --------------------------------------------------------------
    
    Description
    -----------
    It provides the graphical outputs associated with the principal component method for mixed data: FAMD.

    Parameters
    ----------
    self : an object of class FAMD
    choice : a string corresponding to the graph that you want to do.
                - "ind" for the individual graphs
                - "quanti_var" for the correlation circle
                - "quali_var" for the categorical variables graphs
                - "var" for all the variables (quantitatives and categorical)
    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    figure : The individuals factor map and the variables factor map.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")
    
    if choice not in ["ind","quanti_var","quali_var","var"]:
        raise ValueError("'choice' should be one of 'ind','quanti_var','quali_var' and 'var'.")
    
    if choice == "ind":
        return fviz_famd_ind(self,**kwargs)
    elif choice == "quanti_var":
        return fviz_famd_col(self,**kwargs)
    elif choice == "quali_var":
        return fviz_famd_mod(self,**kwargs)
    elif choice == "var":
        return fviz_famd_var(self,**kwargs)