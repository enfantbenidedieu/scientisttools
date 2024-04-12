# -*- coding: utf-8 -*-

import plotnine as pn
import plotnine3d as pn3d
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
from scientisttools.extractfactor import get_eigenvalue
from scientisttools.ggcorrplot import ggcorrplot, no_panel
from scientisttools.utils import get_melt
import matplotlib.pyplot as plt


##################################################################################################
#                       Visualize the cosines of row/column elements
###################################################################################################

def fviz_cosines(self,
                 choice="ind",
                 axis=None,
                 x_label=None,
                 top_cos2=10,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 short_labels=False,
                 ggtheme=pn.theme_minimal()) -> plt:
    
    """ Plot the row and columns cosines graph
            
        For the selected axis, the graph represents the row or column
        cosines sorted in descending order.            
        
        Parameters
        ----------
        choice : {'ind','var','mod','quanti_sup','quali_sup','ind_sup'}
                    'ind' :   individuals
                    'var' :   continues variables
                    'mod' :   categories
                    'quanti_sup' : supplementary continues variables
                    'quali_sup' : supplementary categories variables
                    'ind_sup ' : supplementary individuals
        
        axis : None or int
            Select the axis for which the row/col cosines are plotted. If None, axis = 0.
        
        xlabel : None or str (default)
            The label text.
        
        top_cos2 : int
            Set the maximum number of values to plot.
            If top_cos2 is None : all the values are plotted.
            
        bar_width : None, float or array-like.
            The width(s) of the bars.

        add_grid : bool or None, default = True.
            Whether to show the grid lines

        color : color or list of color, default = "steelblue".
            The colors of the bar faces.

        short_labels : bool, default = False
        
        Returns
        -------
        None
        """

    if choice not in ["row","col","ind","var","mod","quanti_var","quali_var","ind_sup"]:
        raise ValueError("Error : 'choice' not allowed.")
    
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise ValueError("Error : 'axis' must be an integer.")
    elif axis < 0 or axis > self.n_components_:
        raise ValueError(f"Error : 'axis' must be an integer between 0 and {self.n_components_ - 1}")

    if xlabel is None:
        xlabel = "Cos2 - Quality of representation"
    if bar_width is None:
        bar_width = 0.5
    if top_cos2 is None:
        top_cos2 = 10
        
    if choice == "ind":
        name = "individuals"
        if self.model_ == "ca":
            name = "rows"
        cos2 = self.row_cos2_[:,axis]
        labels = self.row_labels_
    elif choice == "var" :
        if self.model_ != "mca":
            name = "continues variables"
            cos2 = self.col_cos2_[:,axis]
            labels  = self.col_labels_
            if self.model_ == "ca":
                name = "columns"
        else:
            name = "categorical variables"
            cos2 = self.var_cos2_[:,axis]
            labels  = self.var_labels_
    elif choice == "mod" and self.model_ in ["mca","famd"]:
        name = "categories"
        cos2 = self.mod_cos2_[:,axis]
        if short_labels:
            labels = self.short_labels_
        else:
            labels = self.mod_labels_
    elif choice == "quanti_sup" and self.model_ != "ca":
        if ((self.quanti_sup_labels_ is not None) and (len(self.col_sup_labels_) >= 2)):
            name = "supplementary continues variables"
            cos2 = self.col_sup_cos2_[:,axis]
            labels = self.col_sup_labels_
        else:
            raise ValueError("Error : Factor Model must have at least two supplementary continuous variables.")
    elif choice == "quali_sup" and self.model_ !="ca":
        if self.quali_sup_labels_ is not None:
            name = "supplementary categories"
            cos2 = self.mod_sup_cos2_[:,axis]
            if short_labels:
                labels = self.short_sup_labels_
            else:
                labels = self.mod_sup_labels_
    
    # Start
    n = len(labels)
    n_labels = len(labels)
    if (top_cos2 is not None) & (top_cos2 < n_labels):
        n_labels = top_cos2
        
    limit = n - n_labels
    cos2_sorted = np.sort(cos2)[limit:n]
    labels_sort = pd.Series(labels)[np.argsort(cos2)][limit:n]

    df = pd.DataFrame({"labels" : labels_sort, "cos2" : cos2_sorted})

    p = pn.ggplot(df,pn.aes(x = "reorder(labels,cos2)", y = "cos2"))+pn.geom_bar(stat="identity",fill=color,width=bar_width)

    title = f"Cosinus of {name} to Dim-{axis+1}"
    p = p + pn.ggtitle(title)+pn.xlab(name)+pn.ylab(xlabel)
    p = p + pn.coord_flip()

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"),
                         axis_text_x = pn.element_text(angle = 60, ha = "center", va = "center"))

    return p+ggtheme







######################################################################################################
##                             Multiple Correspondence Analysis (MCA)
######################################################################################################




####################################################################################################################
#               Factor Analyis of Mixed Data (FAMD)
####################################################################################################################

def fviz_famd_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label=None,
                 y_label=None,
                 title =None,
                 color ="black",
                 geom_type = ["point","text"],
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
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

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
            raise ValueError("Error : 'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query(f"contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("Error : 'lim_contrib' must be a float or an integer")
    
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
                raise ValueError("Error : 'color' must me a numeric variable.")
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
            if "point" in geom_type:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                        pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
            c = [str(x+1) for x in color.labels_]
            if legend_title is None:
                legend_title = "Cluster"
            if "point" in geom_type:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom_type:
                p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle':"-","lw":1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        if "point" in geom_type:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
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
    if "point" in geom_type:
        p = p + pn.geom_point(quali_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_coord.index.tolist()),
                              color = color_quali_var,shape = marker_quali_var,size=point_size)
    if "text" in geom_type:
        if repel:
            p = p + text_label(text_type,data=quali_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_coord.index.tolist()),
                                color=color_quali_var,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,data=quali_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_coord.index.tolist()),
                               color = color_quali_var,size=text_size,va=va,ha=ha)

    ############################## Add supplementary individuals informations
    if ind_sup:
        if self.ind_sup is not None:
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color=color_sup,size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color = color_sup,size=text_size,va=va,ha=ha)
    
    ############## Add supplementary qualitatives
    if quali_sup:
        if self.quali_sup is not None:
            quali_sup_coord = self.quali_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index.tolist()),
                                      color = color_quali_sup,shape = marker_quali_sup,size=point_size)
            if "text" in geom_type:
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
                 geom_type = ["arrow", "text"],
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
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = self.quanti_var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.quanti_var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("Error : 'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.quanti_var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("Error : 'lim_contrib' must be a float or an integer.")
    
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
        if "arrow" in geom_type:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                     arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom_type:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "arrow" in geom_type:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom_type:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom_type:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(angle=arrow_angle,length=arrow_length),color=color)
        if "text" in geom_type:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if quanti_sup:
        if self.quanti_sup is not None:
            sup_coord = self.quanti_sup_["coord"]
            if "arrow" in geom_type:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                     arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
            if "text" in geom_type:
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
                 geom_type = ["point","text"],
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
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

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
            raise TypeError("Error : 'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.quali_var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("Error : 'lim_contrib' must be a float or an integer")

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
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom_type:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Add supplementary categories
    if quali_sup:
        if self.quali_sup is not None:
            quali_sup_coord = self.quali_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(data=quali_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                      color=color_sup,size=point_size,shape=marker_sup)
            if "text" in geom_type:
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
                 geom_type = ["point","text"],
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
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    quanti_var_cos2 = self.quanti_var_["cos2"]
    quali_var_eta2 = self.var_["coord"].loc[self.call_["quali"].columns.tolist(),:]
    
    # Initialize
    p = pn.ggplot(data=quanti_var_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_var_cos2.index))
    if "point" in geom_type:
        p = p + pn.geom_point(color=color_quanti,shape=marker_quanti,size=point_size,show_legend=False)
    
    if "text" in geom_type:
        if repel :
            p = p + text_label(text_type,color=color_quanti,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color_quanti,size=text_size,va=va,ha=ha)
    
    # Add Qualitatives variables
    if "point" in geom_type:
        p = p + pn.geom_point(data=quali_var_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_eta2.index.tolist()),
                              color=color_quali,size=point_size,shape=marker_quali)
    if "text" in geom_type:
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
        if self.quanti_sup is not None:
            quanti_sup_cos2 = self.quanti_sup_["cos2"]
            if "point" in geom_type:
                p = p + pn.geom_point(data=quanti_sup_cos2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quanti_sup_cos2.index),
                                    color=color_quanti_sup,size=point_size,shape=marker_quanti_sup)
            if "text" in geom_type:
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
        if self.quali_sup is not None:
            quali_sup_eta2 = self.quali_sup_["eta2"]
            if "point" in geom_type:
                p = p + pn.geom_point(data=quali_sup_eta2,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                    color=color_quali_sup,size=point_size,shape=marker_quali_sup)
            if "text" in geom_type:
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
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if choice not in ["ind","quanti_var","quali_var","var"]:
        raise ValueError("Error : 'choice' should be one of 'ind','quanti_var','quali_var' and 'var'.")
    
    if choice == "ind":
        return fviz_famd_ind(self,**kwargs)
    elif choice == "quanti_var":
        return fviz_famd_col(self,**kwargs)
    elif choice == "quali_var":
        return fviz_famd_mod(self,**kwargs)
    elif choice == "var":
        return fviz_famd_var(self,**kwargs)

##########################################################################################################
###### Principal Components Analysis with partial correlation matrix (PartialPCA)
###########################################################################################################

def fviz_partialpca_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom_type = ["point","text"],
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
        raise ValueError("Error : 'self' must be an instance of class PartialPCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

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
            raise ValueError("Error : 'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("Error : 'lim_contrib' must be a float or an integer.")

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
                raise ValueError("Error : 'color' must me a numeric variable.")
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
        if "point" in geom_type:
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],
                                                name = legend_title)
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '->',"lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom_type:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    ############################## Add supplementary individuals informations
    if ind_sup:
        if self.ind_sup is not None:
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
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
                 geom_type = ["arrow","text"],
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
        raise ValueError("Error : 'self' must be an object of class PartialPCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = self.var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise ValueError("Error : 'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("Error : 'lim_contrib' must be a float or an integer.")

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
        if "arrow" in geom_type:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom_type:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "arrow" in geom_type:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                     pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom_type:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom_type:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if "text" in geom_type:
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
        raise ValueError("Error : 'self' must be an object of class PartialPCA.")

    if choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'var'")

    if choice == "ind":
        return fviz_partialpca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_partialpca_var(self,**kwargs)

###################################################################################################################################
#           Exploratory Factor Analysis (EFA)
#################################################################################################################################

def fviz_efa_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label=None,
                 title =None,
                 geom_type = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 color ="black",
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title = None,
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
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Exploratory Factor Analysis (EFA) - Graph of individuals
    ------------------------------------------------------------------

    Description
    -----------

    Parameters
    ----------
    self : an object of class EFA


    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = self.ind_["coord"]

    ##### Add initial data
    coord = pd.concat((coord,self.call_["Xtot"]),axis=1)

    if isinstance(color,str):
        if color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise ValueError("Error : 'color' must me a numeric variable.")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    # Using cosine and contributions
    if (isinstance(color,str) and color in coord.columns.tolist()) or (isinstance(color,np.ndarray)):
            # Add gradients colors
        if "point" in geom_type:
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],
                                                name = legend_title)
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '->',"lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom_type:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    ############################## Add supplementary individuals informations
    if ind_sup:
        if self.ind_sup is not None:
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                        color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
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
        title = "Individuals factor map - EFA"
    
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

def fviz_efa_var(self,
                 axis=[0,1],
                 title =None,
                 x_label = None,
                 y_label = None,
                 color ="black",
                 geom_type = ["arrow", "text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 text_size = 8,
                 legend_title = None,
                 add_grid =True,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 lim_contrib = None,
                 add_circle = True,
                 color_circle = "gray",
                 arrow_angle=10,
                 arrow_length =0.1,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Exploratory Factor Analysis (EFA) - Graph of variables
    ----------------------------------------------------------------

    Description
    -----------

    Parameters
    ----------
    self : an object of class EFA


    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = self.var_["coord"]

    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("Error : 'lim_contrib' must be a float or an integer.")

    if isinstance(color,str):
        if color == "contrib":
            c = self.var_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(color,str) and color in ["contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        if "arrow" in geom_type:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom_type:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        if "arrow" in geom_type:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                     pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom_type:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom_type:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if "text" in geom_type:
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
        title = "Variables factor map - EFA"
    
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

def fviz_efa(self,choice="ind",**kwargs)->plt:
    """
    Visualize Exploratory Factor Analysis (EFA)
    -------------------------------------------

    Description
    -----------
    Exploratory factor analysis is a statistical technique that is used to reduce data 
    to a smaller set of summary variables and to explore the underlying theoretical structure of the phenomena.
    It is used to identify the structure of the relationship between the variable and the respondent.
    fviz_efa() provides plotnine-based elegant visualization of EFA outputs
    
    * fviz_efa_ind(): Graph of individuals
    * fviz_efa_var(): Graph of variables

    Parameters
    ----------



    Return
    ------
    a plotnine

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")

    if choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'var'")

    if choice == "ind":
        return fviz_efa_ind(self,**kwargs)
    elif choice == "var":
        return fviz_efa_var(self,**kwargs)

#############################################################################################################################################
####################################################### Multiple Factor Analysie (MFA) plot #################################################
#############################################################################################################################################

def fviz_mfa_ind(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="blue",
                 geom_type = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "red",
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


    
    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an object of class MFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize coordinates
    coord = self.ind_["coord"]

    # Concatenate with actives columns
    for grp in self.call_["X"].columns.get_level_values(0).unique():
        coord = pd.concat((coord,self.call_["X"][grp]),axis=1)

    #### Add supplementary columns
    if self.group_sup is not None:
        not_in = [x for x in self.call_["Xtot"].columns.get_level_values(0).unique() if x not in self.call_["X"].columns.get_level_values(0).unique()]
        if len(not_in)!=1:
            X_group_sup = self.call_["Xtot"][not_in]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=[name for i, name in enumerate(self.call_["Xtot"].index.tolist()) if i in self.ind_sup_["coord"].index.toliost()])
            for grp in not_in:
                coord = pd.concat((coord,X_group_sup[grp]),axis=1)
        else:
            X_group_sup = self.call_["Xtot"][not_in]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=[name for i, name in enumerate(self.call_["Xtot"].index.tolist()) if i in self.ind_sup_["coord"].index.toliost()])
            coord = pd.concat((coord,X_group_sup),axis=1)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float) or isinstance(lim_cos2,int):
            lim_cos2 = float(lim_cos2)
            cos2 = self.ind_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("Error : 'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float) or isinstance(lim_contrib,int):
            lim_contrib = float(lim_contrib)
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("Error : 'lim_contrib' must be a float or an integer")

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
                raise ValueError("Error : 'color' must me a numeric variable.")
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
            if "point" in geom_type:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                        pn.scale_color_gradient2(midpoint=np.mean(c),low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
            c = [str(x+1) for x in color.labels_]
            if legend_title is None:
                legend_title = "Cluster"
            if "point" in geom_type:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom_type:
                p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.group_sup is not None:
            if self.quali_var_sup_ is None:
                raise ValueError(f"Error : {habillage} not in DataFrame.")
            if "point" in geom_type:
                p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if "text" in geom_type:
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
        if self.ind_sup is not None:
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)
    ### Add supplementary qualitatives
    if quali_sup:
       if self.group_sup is not None:
        if self.quali_var_sup_ is None:
            raise ValueError("Error : No supplementary qualitatives")
        if habillage is None:
            quali_var_sup_coord = self.quali_var_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(quali_var_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_var_sup_coord.index.tolist()),
                                      color=color_quali_sup,size=point_size)
                
            if "text" in geom_type:
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


def fviz_mfa_col(self,
                 axis=[0,1],
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="group",
                 geom = ["arrow","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 add_labels = True,
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an instance of class MFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index=self.col_labels_,columns=self.dim_index_)
    coord["Groups"] =  self.col_group_labels_

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cosinus = pd.DataFrame(self.col_cos2_,index=self.col_labels_,columns=self.dim_index_)
            cos2 = (cosinus.iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = pd.DataFrame(self.col_contrib_,index=self.col_labels_,columns=self.dim_index_)
            contrib = (contrib.iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.col_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.col_contrib_[:,axis],axis=1)
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
            p = p + pn.geom_point(color=c,shape=marker,size=point_size)
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
            if self.col_sup_labels_ is not None:
                sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
                sup_coord["Groups"] = self.col_sup_group_labels_
                coord = pd.concat([coord,sup_coord],axis=0)
        p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(color="Groups"),shape=marker,size=point_size)
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color="Groups"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color="Groups"),size=text_size,va=va,ha=ha)
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
        if self.col_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_)
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
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Quantitatives variables - MFA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p


###### Group
def fviz_mfa_group(self,
                   axis=[0,1],
                   xlim=None,
                    ylim=None,
                    title =None,
                    color ="red",
                    gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    marker = "o",
                    add_grid =True,
                    add_labels=True,
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
                    repel=False,
                    lim_cos2 = None,
                    lim_contrib = None,
                    ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    """
    
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an instance of class MFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize coordinates
    coord = self.group_coord_
    
    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (self.group_cos2_
                        .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                        .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (self.group_contrib_
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.group_cos2_.iloc[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.group_contrib_.iloc[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(midpoint=np.mean(c),low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if add_labels:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
            
    # Add supplementary groups
    if group_sup:
       if self.group_sup is not None:
           sup_coord = self.group_sup_coord_
           p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                 color = color_sup,shape = marker_sup,size=point_size)
           if add_labels:
               if repel:
                   p = p + text_label(text_type,data=sup_coord,
                                      mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
               else:
                   p = p + text_label(text_type,data=sup_coord,
                                      mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Variable groups - MFA"
    
    if ((xlim is not None) and ((isinstance(xlim,list) or (isinstance(xlim,tuple))))):
        p = p + pn.xlim(xlim)
    if ((ylim is not None) and ((isinstance(ylim,list) or (isinstance(ylim,tuple))))):
        p = p + pn.ylim(ylim)
   
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

######################################## axes correlation plot ################################################
def fviz_mfa_axes(self,
                 axis=[0,1],
                 title =None,
                 color="black",
                 color_circle = "lightgray",
                 geom = ["arrow","text"],
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 group_sup=True,
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an instance of class MFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Create coordinates
    coord = pd.DataFrame().astype("float")
    for grp, cols in self.group_.items():
        data = self.partial_axes_coord_[grp].T
        data["Groups"] = grp
        data.index = [x+"_"+str(grp) for x in data.index]
        coord = pd.concat([coord,data],axis=0)
    
    if group_sup:
        if self.group_sup is not None:
            for grp, cols in self.group_sup_.items():
                data = self.partial_axes_sup_coord_[grp].T
                data["Groups"] = grp
                data.index = [x+"_"+str(grp) for x in data.index]
                coord = pd.concat([coord,data],axis=0)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if color == "group":
        if legend_title is None:
            legend_title = "Groups"
        if "arrow" in geom:
            p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color="Groups"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                    pn.guides(color=pn.guide_legend(title=legend_title)))
        if "text" in geom:
            p = p + text_label(text_type,mapping=pn.aes(color="Groups"),size=text_size,va=va,ha=ha)
    else:
        if "arrow" in geom:
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if "text" in geom :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=color_circle, fill=None)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Partial axes - MFA"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

#################################################################################################################
#                   Hierarchical Clustering on Principal Components (HCPC)
#################################################################################################################

def fviz_hcpc_cluster(self,
                      axis=(0,1),
                      x_lim=None,
                      y_lim=None,
                      x_label=None,
                      y_label=None,
                      title=None,
                      legend_title = None,
                      geom_type = ["point","text"],
                      show_clust_cent = False, 
                      cluster = None,
                      center_marker_size=5,
                      marker = None,
                      repel=True,
                      ha = "center",
                      va = "center",
                      point_size = 1.5,
                      text_size = 8,
                      text_type = "text",
                      add_grid=True,
                      add_hline=True,
                      add_vline=True,
                      hline_color="black",
                      vline_color="black",
                      hline_style = "dashed",
                      vline_style = "dashed",
                      add_ellipse=True,
                      ellipse_type = "t",
                      confint_level = 0.95,
                      geom_ellipse = "polygon",
                      ggtheme = pn.theme_minimal()):
    """
    
    
    
    """

    if self.model_ != "hcpc":
        raise ValueError("Error : 'self' must be an object of class HCPC.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["model"].call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    if legend_title is None:
        legend_title = "Cluster"

    #### Extract data
    coord = self.call_["tree"]["data"]

    if cluster is None:
        coord = pd.concat([coord,self.cluster_["cluster"]],axis=1)
    else:
        coord = pd.concat([coord,cluster],axis=1)
        
    # Rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]
     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))
    # if "point" in geom_type:
    #     p = p + pn.geom_point(pn.aes(color = "cluster"),size=point_size,shape=marker)
    # if "text" in geom_type:
    #     if repel:
    #         p = p + text_label(text_type,mapping=pn.aes(color="cluster"),size=text_size,va=va,ha=ha,
    #                            adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
    #     else:
    #         p = p + text_label(text_type,mapping=pn.aes(color="cluster"),size=text_size,va=va,ha=ha)
    
    if "point" in geom_type:
        p = (p + pn.geom_point(pn.aes(color="Cluster"),shape=marker,size=point_size,show_legend=False)+
                 pn.guides(color=pn.guide_legend(title=legend_title)))
    if "text" in geom_type:
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=legend_title),size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=legend_title),size=text_size,va=va,ha=ha)
    if add_ellipse:
        p = (p + pn.geom_point(pn.aes(color = legend_title))+ 
                 pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=legend_title),type = ellipse_type,alpha = 0.25,level=confint_level))
    
    if show_clust_cent:
        cluster_center = self.cluster_["coord"]
        if "point" in geom_type:
            p = p + pn.geom_point(cluster_center,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=cluster_center.index,color=cluster_center.index),size=center_marker_size)
    
    # Add additionnal        
    proportion = self.call_["model"].eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set title
    if title is None:
        title = "Individuals factor map - HCPC"
    p = p + pn.labs(title=title,x=x_label,y = y_label)
    
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
    
    p = p + ggtheme

    return p

######################################################################################################################
#                   Classical Multidimensional Scaling (CMDSCALE)
######################################################################################################################

def fviz_cmds(self,
            axis=[0,1],
            text_type = "text",
            point_size = 1.5,
            text_size = 8,
            xlim=None,
            ylim=None,
            title =None,
            xlabel=None,
            ylabel=None,
            color="blue",
            color_sup ="red",
            add_labels = True,
            marker="o",
            marker_sup = "^",
            add_sup = True,
            add_grid =True,
            add_hline = True,
            add_vline=True,
            ha="center",
            va="center",
            hline_color="black",
            hline_style="dashed",
            vline_color="black",
            vline_style ="dashed",
            repel=True,
            ggtheme=pn.theme_gray()) -> pn:
    """
    Draw the Classical multidimensional scaling (CMDSCALE) graphs
    ----------------------------------------------------------

    
    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "cmds":
        raise ValueError("Error : 'self' must be an object of class CMDSCALE.")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid axis")
    
    coord = pd.DataFrame(self.coord_,index = self.labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    # Add point
    p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if add_labels:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                        adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    if add_sup:
        if self.sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.sup_coord_, index= self.sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(data=sup_coord,
                                  mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                  color=color_sup,size=point_size,shape=marker_sup)

            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    
    if title is None:
        title = "Classical multidimensional scaling (PCoA, Principal Coordinates Analysis)"

    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if ((xlim is not None) and ((isinstance(xlim,list) or (isinstance(xlim,tuple))))):
        p = p + pn.xlim(xlim)
    if ((ylim is not None) and ((isinstance(ylim,list) or (isinstance(ylim,tuple))))):
        p = p + pn.ylim(ylim)
   
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p + pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    
    p = p + ggtheme
    return p

######################################################################################################################
#                   Metric and Non - Metric Multidimensional Scaling (CMDSCALE)
######################################################################################################################

def fviz_mds(self,
            axis=[0,1],
            text_type = "text",
            point_size = 1.5,
            text_size = 8,
            xlim=None,
            ylim=None,
            title =None,
            xlabel=None,
            ylabel=None,
            color="blue",
            color_sup ="red",
            marker="o",
            marker_sup = "^",
            add_sup = True,
            add_grid =True,
            add_hline = True,
            add_vline=True,
            add_labels = True,
            ha="center",
            va="center",
            hline_color="black",
            hline_style="dashed",
            vline_color="black",
            vline_style ="dashed",
            repel=False,
            ggtheme=pn.theme_gray()) -> pn:
    """
    
    
    """
    
    if self.model_ != "mds":
        raise ValueError("Error : 'self' must be an instance of class MDS.")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid axis")
    
    coord = pd.DataFrame(self.coord_,index = self.labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    # Add point
    p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if add_labels:
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                        adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if add_sup:
        if self.sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.sup_coord_, index= self.sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(data=sup_coord,
                                  mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                  color=color_sup,size=point_size,shape=marker_sup)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    if title is None:
        title = self.title_

    p = p + pn.ggtitle(title)
    if xlabel is not None:
        p = p + pn.xlab(xlab=xlabel)
    if ylabel is not None:
        p = p + pn.ylab(ylab=ylabel)
    
    if ((xlim is not None) and ((isinstance(xlim,list) or (isinstance(xlim,tuple))))):
        p = p + pn.xlim(xlim)
    if ((ylim is not None) and ((isinstance(ylim,list) or (isinstance(ylim,tuple))))):
        p = p + pn.ylim(ylim)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p + pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    
    p = p + ggtheme
    return p

# Shepard Diagram
def fviz_shepard(self,
                 xlim=None,
                 ylim=None,
                 color="blue",
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 add_grid=True,
                 ggtheme=pn.theme_gray())-> plt:
    """Computes the Shepard plot
    
    
    """
    
    if self.model_ not in ["cmds","mds"]:
        raise ValueError("Error : 'Method' is allowed only for multidimensional scaling.")
    
    coord = pd.DataFrame({"InDist": self.dist_[np.triu_indices(self.nobs_, k = 1)],
                          "OutDist": self.res_dist_[np.triu_indices(self.nobs_, k = 1)]})
    
    p = pn.ggplot(coord,pn.aes(x = "InDist",y = "OutDist"))+pn.geom_point(color=color)

    if xlabel is None:
        xlabel = "Input Distances"
    if ylabel is None:
        ylabel = "Output Distances"
    if title is None:
        title = "Shepard Diagram"
    
    p = p + pn.ggtitle(title)+pn.xlab(xlabel)+pn.ylab(ylabel)

    if ((xlim is not None) and ((isinstance(xlim,list) or (isinstance(xlim,tuple))))):
        p = p + pn.xlim(xlim)
    if ((ylim is not None) and ((isinstance(ylim,list) or (isinstance(ylim,tuple))))):
        p = p + pn.ylim(ylim)

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    return p+ ggtheme




    


def fviz_candisc(self,
                axis = [0,1],
                xlabel = None,
                ylabel = None,
                point_size = 1.5,
                xlim =(-5,5),
                ylim = (-5,5),
                text_size = 8,
                text_type = "text",
                title = None,
                add_grid = True,
                add_hline = True,
                add_vline=True,
                marker = None,
                repel = False,
                add_labels = True,
                hline_color="black",
                hline_style="dashed",
                vline_color="black",
                vline_style ="dashed",
                ha = "center",
                va = "center",
                ggtheme=pn.theme_gray()):
    

    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class 'CANDISC'.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    # Initialize coordinates
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add target variable
    coord = pd.concat([coord, self.data_[self.target_]],axis=1)

    p = (pn.ggplot(data=coord,mapping=pn.aes(x = f"LD{axis[0]+1}",y=f"LD{axis[1]+1}",label=coord.index))+
        pn.geom_point(pn.aes(color=self.target_[0]),size=point_size,shape=marker))
    
    if add_labels:
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(color=self.target_[0]),size=text_size,va=va,ha=ha,
                            adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=self.target_[0]),size=text_size,va=va,ha=ha)

    if xlabel is None:
        xlabel = f"Canonical {axis[0]+1}"
    
    if ylabel is None:
        ylabel = f"Canonical {axis[1]+1}"
    
    if title is None:
        title = "Canonical Discriminant Analysis"
    
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)
    p = p + pn.xlim(xlim)+pn.ylim(ylim)
    
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme
    
    return p

################################################ Correspondance Discriminant Analysis (CDA) ##################################

##
def fviz_disca_ind(self,
                   axis=[0,1],
                   xlabel=None,
                   ylabel=None,
                   title = None,
                   xlim = None,
                   ylim = None,
                   color="black",
                   gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                   legend_title=None,
                   add_labels=True,
                   repel=True,
                   point_size = 1.5,
                   text_size = 8,
                   text_type = "text",
                   add_grid = True,
                   add_hline = True,
                   add_vline=True,
                   marker = None,
                   ha = "center",
                   va = "center",
                   hline_color = "black",
                   hline_style = "dashed",
                   vline_color = "black",
                   vline_style = "dashed",
                   lim_cos2 = None,
                   lim_contrib = None,
                   add_ellipse = False,
                   ellipse_type = "t",
                   confint_level = 0.95,
                   geom_ellipse = "polygon",
                   ggtheme=pn.theme_gray()):
    

    if self.model_ != "disca":
        raise ValueError("Error : 'self' must be an instance of class 'DISCA'.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    # Initialize coordinates
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add target variable
    coord = pd.concat([coord, self.data_[self.target_]],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.row_cos2_,index = self.row_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.row_contrib_,index = self.row_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.row_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.row_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                 pn.scale_color_gradient2(low = gradient_cols[0],
                                          high = gradient_cols[2],
                                          mid = gradient_cols[1],
                                          name = legend_title))
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+
                     pn.guides(color=pn.guide_legend(title=legend_title)))
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
            
            if add_ellipse:
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=c),type = ellipse_type,alpha = 0.25,level=confint_level)
    elif color == self.target_[0]:
        habillage = self.target_[0]
        p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
        if add_labels:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            
        if add_ellipse:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if add_labels:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - DISCA"
    
    if ((xlim is not None) and ((isinstance(xlim,list) or (isinstance(xlim,tuple))))):
        p = p + pn.xlim(xlim)
    if ((ylim is not None) and ((isinstance(ylim,list) or (isinstance(ylim,tuple))))):
        p = p + pn.ylim(ylim)
   
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p
    

def fviz_disca_group(self):
    pass

def fviz_disca_mod(self):
    pass