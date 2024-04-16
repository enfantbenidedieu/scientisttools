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



#################################################################################################################
#                   Hierarchical Clustering on Principal Components (HCPC)
#################################################################################################################


######################################################################################################################
#                   Classical Multidimensional Scaling (CMDSCALE)
######################################################################################################################

def fviz_cmdscale(self,
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
    
    if self.model_ != "cmdscale":
        raise ValueError("'self' must be an object of class CMDSCALE.")
     
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