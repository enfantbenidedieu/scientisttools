# -*- coding: utf-8 -*-

import plotnine as pn
import numpy as np
import pandas as pd
import sklearn
from scientisttools.extractfactor import get_eigenvalue
from scientisttools.ggcorrplot import ggcorrplot, no_panel
from scientisttools.utils import get_melt
import matplotlib.pyplot as plt
import plydata as ply

def text_label(texttype,**kwargs):
    """Function to choose between ``geom_text`` and ``geom_label``

    Parameters
    ----------
    text_type : {"text", "label"}, default = "text"

    **kwargs : geom parameters

    return
    ------

    
    """
    if texttype == "text":
        return pn.geom_text(**kwargs)
    elif texttype == "label":
        return pn.geom_label(**kwargs)
    
def gg_circle(r, xc, yc, color="black",fill=None,**kwargs):
    seq1 = np.linspace(0,np.pi,num=100)
    seq2 = np.linspace(0,-np.pi,num=100)
    x = xc + r*np.cos(seq1)
    ymax = yc + r*np.sin(seq1)
    ymin = yc + r*np.sin(seq2)
    return pn.annotate("ribbon", x=x, ymin=ymin, ymax=ymax, color=color, fill=fill,**kwargs)


def fviz_screeplot(self,
                   choice="proportion",
                   geom_type=["bar","line"],
                   ylim=None,
                   bar_fill = "steelblue",
                   bar_color="steelblue",
                   line_color="black",
                   line_type="solid",
                   bar_width=None,
                   add_kaiser=False,
                   add_kss = False,
                   add_broken_stick = False,
                   n_components=10,
                   add_labels=False,
                   ha= "center",
                   va = "bottom",
                   title=None,
                   x_label=None,
                   y_label=None,
                   ggtheme=pn.theme_gray())-> pn:
    """
    Extract and visualize the eigenvalues/proportions of dimensions
    -------------------------------------------------------------

    Parameters
    ----------
    self : an object of class PCA, CA, MCA, FAMD, MFA, CMDS, DISQUAL, MIXDISC
    choice : a text specifying the data to be plotted. Allowed values are "proportion" or "eigenvalue".
    geom_type : a text specifying the geometry to be used for the graph. Allowed values are "bar" for barplot, 
                "line" for lineplot or ["bar", "line"] to use both types.
    ylim : y-axis limits, default = None
    bar_fill : 	fill color for bar plot.
    bar_color : outline color for bar plot.
    line_color : color for line plot (when geom contains "line").
    line_type : line type
    bar_width : float, the width(s) of the bars
    add_kaiser : Kaiser criterion
    add_kss : KSS criterion
    add_broken_stick : Broken Stick criterion
    n_components : a numeric value specifying the number of dimensions to be shown.
    add_labels : logical value. If TRUE, labels are added at the top of bars or points showing the information retained by each dimension.
    ha : horizontal adjustment of the labels.
    va : vertical adjustment of the labels.
    title : title of the graph
    x_label : x-axis title
    y_label : y-axis title
    ggtheme : function plotnine theme name. Default value is theme_gray(). Allowed values include plotnine official themes: 
                theme_gray(), theme_bw(), theme_minimal(), theme_classic(), theme_void(), ....
    
    Return
    ------
    figure : a plotnine graphs

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
        
    if self.model_ not in ["pca","ca","mca","famd","mfa","cmds","disqual","mixdisc"]:
        raise ValueError("'res' must be an object of class PCA, CA, MCA, FAMD, MFA, CMDS, DISQUAL, MIXDISC")

    eig = get_eigenvalue(self)
    eig = eig.iloc[:min(n_components,self.n_components_),:]

    if choice == "eigenvalue":
        eig = eig["eigenvalue"]
        text_labels = list([str(np.around(x,3)) for x in eig.values])
        if self.model_ not in ["famd","mds","mfa","cmds"]:
            kaiser = self.kaiser_threshold_
        if y_label is None:
            y_label = "Eigenvalue"
    elif choice == "proportion":
        eig = eig["proportion"]
        text_labels = list([str(np.around(x,1))+"%" for x in eig.values])
        if self.model_ not in ["pca","famd","mfa","cmds"]:
            kaiser = self.kaiser_proportion_threshold_
    else:
        raise ValueError("Allowed values for the argument choice are : 'proportion' or 'eigenvalue'")
    
    df_eig = pd.DataFrame({"dim" : pd.Categorical(np.arange(1,len(eig)+1)),"eig" : eig.values})
    
    p = pn.ggplot(df_eig,pn.aes(x = "dim",y="eig",group = 1))
    if "bar" in geom_type :
        p = p   +   pn.geom_bar(stat="identity",fill=bar_fill,color=bar_color,width=bar_width)
    if "line" in geom_type :
        p = (p  +   pn.geom_line(color=line_color,linetype=line_type)+\
                    pn.geom_point(shape="o",color=line_color))
    if add_labels:
        p = p + pn.geom_text(label=text_labels,ha = ha,va = va)
    if add_kaiser :
        if self.model_ not in ["famd","mds","mfa","cmds"]:
            p = (p +  pn.geom_hline(yintercept=kaiser,linetype="--", color="red")+\
                      pn.annotate("text", x=int(np.median(np.arange(1,len(eig)+1))), y=kaiser, label="Kaiser threshold"))

    if add_kss:
        if self.model_ in ["pca","ppca"]:
            if choice == "eigenvalue":
                p = (p  +   pn.geom_hline(yintercept=self.kss_threshold_,linetype="--", color="yellow")+ \
                            pn.annotate("text", x=int(np.mean(np.arange(1,len(eig)+1))), y=self.kss_threshold_, 
                                        label="Karlis - Saporta - Spinaki threshold",colour = "yellow"))
            else:
                raise ValueError("'add_kss' is only with 'choice=eigenvalue'.")
        else:
            raise ValueError("'add_kss' is only for class PCA or PPCA")
    if add_broken_stick:
        if choice == "eigenvalue":
            if self.model_ in ["pca","ppca"]:
                bst = self.broken_stick_threshold_[:min(n_components,self.n_components_)]
                p = (p  +   pn.geom_line(pn.aes(x="dim",y=bst),color="green",linetype="--")+\
                            pn.geom_point(pn.aes(x="dim",y=bst),colour="green")+\
                            pn.annotate("text", x=int(np.mean(np.arange(1,len(eig)+1))), y=np.median(bst), 
                                        label="Broken stick threshold",colour = "green"))
            else:
                raise ValueError("'add_broken_stick' is only for class PCA or PPCA")
        else:
            raise ValueError("'add_broken_stick' is only with 'choice==eigenvalue'")

    if title is None:
        title = "Scree plot"
    if x_label is None:
        x_label = "Dimensions"
    if y_label is None:
        y_label = "Percentage of explained variances"
    
    if ylim is not None:
        p = p + pn.ylim(ylim)
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    p = p + ggtheme
    return p

def fviz_eigenvalue(self,**kwargs) -> pn:
    """
    Extract and visualize the eigenvalues/proportions of dimensions
    -------------------------------------------------------------

    see fviz_screeplot(...)
    """
    return fviz_screeplot(self,**kwargs)

def fviz_eig(self,**kwargs) -> pn:
    """
    Extract and visualize the eigenvalues/proportions of dimensions
    -------------------------------------------------------------

    see fviz_screeplot(...)
    """
    return fviz_screeplot(self,**kwargs)

####################################################################################
#       Principal Components Analysis (PCA)
####################################################################################

def fviz_pca_ind(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 add_grid =True,
                 add_labels=True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 legend_title=None,
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 habillage = None,
                 quali_sup = True,
                 color_quali_sup = "red",
                 short_labels=True,
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
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    Draw the Principal Component Analysis (PCA) individuals graphs
    --------------------------------------------------------------

    Author:
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")
    

    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add Active Data
    coord = pd.concat([coord, self.active_data_],axis=1)

    # Add categorical supplementary variables
    if self.quali_sup_labels_ is not None:
        coord[self.quali_sup_labels_] = self.data_[self.quali_sup_labels_]
    
    # Add Supplementary continous variables
    if self.quanti_sup_labels_ is not None:
        coord[self.quanti_sup_labels_] = self.data_[self.quanti_sup_labels_]
    
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
        elif color in coord.columns:
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
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
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
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if add_labels:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
        if add_labels:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color':"black","lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipse:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)

            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                        color=color_sup,size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                        color = color_sup,size=text_size,va=va,ha=ha)
    if quali_sup:
        if self.quali_sup_labels_ is not None:
            if habillage is None:
                if short_labels:
                    mod_sup_labels = self.short_sup_labels_
                else:
                    mod_sup_labels = self.mod_sup_labels_

                mod_sup_coord = pd.DataFrame(self.mod_sup_coord_,columns=self.dim_index_,index=mod_sup_labels)
                
                p = p + pn.geom_point(mod_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                      color=color_quali_sup,size=point_size)
                if add_labels:
                    if repel:
                        p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                           color=color_quali_sup,size=text_size,va=va,ha=ha,
                                           adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,'lw':1.0}})
                    else:
                        p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                           color ="red",size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - PCA"
    
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

def fviz_pca_var(self,
                 axis=[0,1],
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 add_labels = True,
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "blue",
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
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    Draw the Principal Component Analysis (PCA) variables graphs
    ------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
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
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                 arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+
                 pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if add_labels:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                 arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                 pn.guides(color=pn.guide_legend(title=legend_title)))
        if add_labels:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)

    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
        if add_labels:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add supplmentary continuous variables
    if quanti_sup:
        if self.quanti_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                 arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
            if add_labels:
                p  = p + text_label(text_type,
                                    data=sup_coord,
                                    mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color="black", fill=None)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Variables factor map - PCA"
    
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

def fviz_pca(self,choice="ind",**kwargs)->pn:
    """
    Draw the Principal Component Analysis (PCA) graphs
    --------------------------------------------------

    Description
    -----------
    Plot the graphs for a Principal Component Analysis (PCA) with supplementary individuals, 
    supplementary quantitative variables and supplementary categorical variables.

    Parameters
    ----------
    self : an object of class PCA
    choice : the graph to plot
                - 'ind' for the individuals graphs
                - 'var' for the variables graphs (correlation circle)
    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    figure : The individuals factor map and the variables factor map

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an instance of class PCA.")
    
    if choice not in ["ind","var"]:
        raise ValueError("Error : Allowed 'choice' values are 'ind' or 'var'.")

    if choice == "ind":
        return fviz_pca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_pca_var(self,**kwargs)

#################################################################################################################
#               Correspondence Analysis (CA) graphs
#################################################################################################################

# Row points Factor Map
def fviz_ca_row(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 add_grid =True,
                 add_labels=True,
                 row_sup=True,
                 color_sup = "red",
                 marker_sup = "^",
                 add_hline = True,
                 add_vline=True,
                 legend_title = None,
                 add_ellipse=False, 
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
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    
    
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

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
    
    # Set color if cos2, contrib or continuous variables
    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.row_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Cos2"
        elif color == "contrib":
            c = np.sum(self.row_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
 
        # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],
                                         high = gradient_cols[2],
                                         mid = gradient_cols[1],
                                         name = legend_title)
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': "black","lw":1.0}})
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
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if add_labels:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    if row_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Row points - CA"
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

# Columns points factor map
def fviz_ca_col(self,
                 axis=[0,1],
                 xlim= None,
                 ylim=None,
                 title =None,
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 marker = "o",
                 point_size = 1.5,
                 text_size = 8,
                 add_grid =True,
                 add_labels=True,
                 legend_title = None,
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 col_sup=True,
                 color_sup = "red",
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
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    
    
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.col_cos2_[:,axis],axis=1)
            if legend_title is None:
                 legend_title = "Cos2"
        elif color == "contrib":
            c = np.sum(self.col_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],
                                         high = gradient_cols[2],
                                         mid = gradient_cols[1],
                                         name = legend_title)
        if add_labels:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': 'black','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
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
    else:
        p = p + pn.geom_point(pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),color=color,shape=marker,size=point_size)
        if add_labels:
            if repel:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if col_sup:
        if self.col_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p  = p + pn.geom_point(sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                   color=color_sup,shape=marker_sup,size=point_size)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                        color=color_sup,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Columns points - CA"
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

def fviz_ca(self,choice,**kwargs)->pn:
    """
    Draw the Correspondence Analysis (CA) graphs
    --------------------------------------------

    Description
    -----------
    Draw the Correspondence Analysis (CA) graphs.

    Parameters
    ----------
    self : an object of class CA
    choice : the graph to plot
                - 'row' for the row points factor map
                - 'col' for the columns points factor map
    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    figure : The row points factor map and the columns points factor map.

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if choice not in ["row","col"]:
        raise ValueError("Error : Allowed values for choice are :'row' or 'col'.")


    if choice == "row":
        return fviz_ca_row(self,**kwargs)
    elif choice == "col":
        return fviz_ca_col(self,**kwargs)


########################################################
def fviz_corrcircle(self,
                    axis=[0,1],
                    xlabel=None,
                    ylabel=None,
                    title=None,
                    color = "black",
                    color_sup = "blue",
                    text_type="text",
                    arrow_length=0.1,
                    text_size=8,
                    arrow_angle=10,
                    add_labels=True,
                    add_circle=True,
                    add_hline=True,
                    add_vline=True,
                    add_grid=True,
                    ggtheme=pn.theme_gray()) -> pn:
    """
    
    """
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if self.model_ not in ["pca","mca","famd","mfa"]:
        raise ValueError("Error : Factor method not allowed.")
    
    if self.model_ in ["pca","famd","mfa"]:
        coord = pd.DataFrame(self.col_coord_,index=self.col_labels_,columns=self.dim_index_)
    else:
        if self.quanti_sup_labels_ is not None:
            coord = pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_)

    # Initialize
    p = (pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))+
         pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color))
    if add_labels:
            p = p + text_label(text_type,color=color,size=text_size,va="center",ha="center")
        
    if self.model_ in ["pca","famd"]:
        if self.quanti_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                 arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype="--")
            if add_labels:
                p  = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va="center",ha="center")
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color="black", fill=None)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Correlation circle"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour="black", linetype ="dashed")
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour="black", linetype ="dashed")
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

######################################################################################################
##                             Multiple Correspondence Analysis (MCA)
######################################################################################################

def fviz_mca_ind(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 add_labels=True,
                 marker = "o",
                 legend_title=None,
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_ellipse=False, 
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
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    Draw the Multiple Correspondence Analysis (MCA) individuals graphs
    ------------------------------------------------------------------

    Author
    -----
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add Categorical supplementary Variables
    coord = pd.concat([coord, self.original_data_],axis=1)

    # Add categorical supplementary variables
    if self.quali_sup_labels_ is not None:
        coord[self.quali_sup_labels_] = self.data_[self.quali_sup_labels_]
    
    # Add Supplementary continous variables
    if self.quanti_sup_labels_ is not None:
        coord[self.quanti_sup_labels_] = self.data_[self.quanti_sup_labels_]

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
    
    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.row_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.row_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns:
            if not np.issubdtype(coord[color].dtype, np.number):
                raise ValueError("Error : 'color' must me a numeric variable.")
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
        color_list = ["cos2","contrib"]
        for i in range(len(coord.columns)):
            val = coord.columns.values[i]
            color_list.append(val)    
        # Using cosine and contributions
        if (isinstance(color,str) and color in color_list) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title)
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
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if add_labels:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
        if add_labels:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color':"black","lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipse:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(data=sup_coord,
                                  mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,data=sup_coord,
                                   mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
            else:
                p = p + text_label(text_type,data=sup_coord,
                                   mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                     color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - MCA"
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

# Graph for categories
def fviz_mca_mod(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 legend_title=None,
                 marker = "o",
                 add_grid =True,
                 add_labels=True,
                 quali_sup = True,
                 color_sup = "blue",
                 marker_sup = "^",
                 short_labels=True,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 corrected = False,
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 repel=False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    Draw the Multiple Correspondence Analysis (MCA) categorical graphs
    ------------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    # Categories labels
    if short_labels:
        labels = self.short_labels_
    else:
        labels = self.mod_labels_
    
    # Corrected 
    if corrected:
        coord = pd.DataFrame(self.corrected_mod_coord_,index = labels,columns=self.dim_index_)
    else:
        coord = pd.DataFrame(self.mod_coord_,index = labels,columns=self.dim_index_)

    # Initialize
    coord = pd.DataFrame(self.mod_coord_,index = labels,columns=self.dim_index_)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.mod_cos2_,index = labels,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.mod_contrib_,index = labels,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.mod_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.mod_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c= np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                 pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1], name = legend_title))
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
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if add_labels:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Add supplementary categories
    if quali_sup:
        if self.quali_sup_labels_ is not None:
            if short_labels:
                mod_sup_labels = self.short_sup_labels_
            else:
                mod_sup_labels = self.mod_sup_labels_

            mod_sup_coord = pd.DataFrame(self.mod_sup_coord_,columns=self.dim_index_,index=mod_sup_labels)
            
            p = p + pn.geom_point(mod_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                  color=color_sup,size=point_size,shape=marker_sup)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=mod_sup_coord,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text_label(text_type,data=mod_sup_coord,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Qualitatives variables categories - MCA"
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

    # Add theme
    p = p + ggtheme
    
    return p

#------------------------------------------------------------------------
def fviz_mca_var(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title=None,
                 color="black",
                 color_sup = "blue",
                 color_quanti_sup = "red",
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
                 add_labels=True,
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
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_gray()) -> pn:
    """
    Draw the Multiple Correspondence Analysis (MCA) variables graphs
    ----------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.var_eta2_,index =  self.var_labels_,columns=self.dim_index_)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.var_cos2_,index = self.var_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.var_contrib_,index = self.var_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.var_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.var_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                 pn.scale_color_gradient2(low=gradient_cols[0],high=gradient_cols[2],mid=gradient_cols[1],name=legend_title))
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
                                   adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if add_quanti_sup:
        if self.quanti_sup_labels_ is not None:
            var_quant_sup_cos2 = pd.DataFrame(self.col_sup_cos2_,index=self.quanti_sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(var_quant_sup_cos2,
                                  pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_quant_sup_cos2.index),
                                  color = color_quanti_sup,shape = marker_quanti_sup,size=point_size)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=var_quant_sup_cos2,
                                       mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_quant_sup_cos2.index),
                                       color=color_quanti_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quanti_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=var_quant_sup_cos2,
                                       mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_quant_sup_cos2.index),
                                       color = color_quanti_sup,size=text_size,va=va,ha=ha)
    
    if add_quali_sup:
        if self.quali_sup_labels_ is not None:
            var_quali_sup_eta2 = self.quali_sup_eta2_
            p = p + pn.geom_point(var_quali_sup_eta2,
                                  pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_quali_sup_eta2.index),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=var_quali_sup_eta2,
                                       mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_quali_sup_eta2.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=var_quali_sup_eta2,
                                       mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_quali_sup_eta2.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Graphe of variables - MCA"
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

    # Add theme
    p = p + ggtheme
    
    return p

def fviz_mca(self,choice="ind",**kwargs)->pn:
    """
    Draw the Multiple Correspondence Analysis (MCA) graphs
    ------------------------------------------------------

    Description
    -----------
    Draw the Multiple Correspondence Analysis (MCA) graphs.

    Parameters
    ----------
    self : an object of class MCA
    choice : the graph to plot
                - "ind" for the individuals graphs
                - "mod" for the categories graphs
                - "var" for the variables graphs
                - "quanti_sup" for the supplementary quantitatives variables.
    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    figure : The individuals factor map and the variables factor map.

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if choice not in ["ind","mod","var","quanti_sup"]:
        raise ValueError("Error : 'choice' values allowed are : 'ind', 'mod', 'var' and 'quanti_sup'.")
    
    if choice == "ind":
        return fviz_mca_ind(self,**kwargs)
    elif choice == "mod":
        return fviz_mca_mod(self,**kwargs)
    elif choice == "var":
        return fviz_mca_var(self,**kwargs)
    elif choice == "quanti_sup":
        if self.quanti_sup_labels_ is not None:
            return fviz_corrcircle(self,**kwargs)
        else:
            raise ValueError("Error : No supplementary continuous variables available.")

####################################################################################################################
#               Factor Analyis of Mixed Data (FAMD)
####################################################################################################################

def fviz_famd_ind(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title = None,
                 add_labels=True,
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_ellipse=False, 
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
                 ggtheme=pn.theme_gray()) -> pn:
    
    """
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) individuals graphs
    --------------------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add Categorical supplementary Variables
    coord = pd.concat([coord, self.active_data_],axis=1)
    
    # Add categorical supplementary variables
    if self.quali_sup_labels_ is not None:
        coord[self.quali_sup_labels_] = self.data_[self.quali_sup_labels_]

    # Add Supplementary continous variables
    if self.quanti_sup_labels_ is not None:
        coord[self.quanti_sup_labels_] = self.data_[self.quanti_sup_labels_]

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

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.row_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.row_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns:
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

    if habillage is None :        
        # Using cosine and contributions
        color_list = ["cos2","contrib"]
        for i in range(len(coord.columns)):
            val = coord.columns.values[i]
            color_list.append(val)
        if (isinstance(color,str) and color in color_list) or isinstance(color,np.ndarray):
            # Add gradients colors
            p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                      pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if add_labels:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': "black","lw":1.0}})
                else:
                    p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
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
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if add_labels:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
        if add_labels:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color':"black","lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipse:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                     color=color_sup,size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
            else:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                     color = color_sup,size=text_size,va=va,ha=ha)
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - FAMD"
    
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

def fviz_famd_col(self,
                 axis=[0,1],
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 text_size = 8,
                 legend_title=None,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "blue",
                 linestyle_sup="dashed",
                 add_labels=True,
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
                 ggtheme=pn.theme_gray()) -> pn:
    
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
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
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
    
     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
    
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                arrow = pn.arrow(angle=arrow_angle,length=arrow_length))+ 
                 pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if add_labels:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
        c = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"
        p = (p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=c), 
                                 arrow = pn.arrow(length=arrow_length,angle=arrow_angle))+ 
                 pn.guides(color=pn.guide_legend(title=legend_title)))
        if add_labels:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                arrow = pn.arrow(angle=arrow_angle,length=arrow_length),color=color)
        if add_labels:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if quanti_sup:
        if self.quanti_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                 arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
            if add_labels:
                p  = p + text_label(text_type,data=sup_coord,
                                    mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color="black", fill=None)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Continuous variables factor map - FAMD"
    
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

# Graph for categories
def fviz_famd_mod(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="black",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title=None,
                 add_labels=True,
                 add_grid =True,
                 quali_sup = True,
                 color_sup = "blue",
                 marker_sup = "^",
                 short_labels=True,
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
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
                 ggtheme=pn.theme_gray()) -> pn:
    
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
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    # Categories labels
    if short_labels:
        labels = self.short_labels_
    else:
        labels = self.mod_labels_

    # Initialize
    coord = pd.DataFrame(self.mod_coord_,index = labels,columns=self.dim_index_)

    # Using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,float):
            cos2 = (pd.DataFrame(self.mod_cos2_,index = self.mod_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False)
                       .query(f"cosinus > {lim_cos2}"))
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
    
    # Using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,float):
            contrib = (pd.DataFrame(self.mod_contrib_,index = self.mod_labels_,columns=self.dim_index_)
                       .iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False)
                       .query(f"contrib > {lim_contrib}"))
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]

    if isinstance(color,str):
        if color == "cos2":
            c = np.sum(self.mod_cos2_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "cos2"
        elif color == "contrib":
            c = np.sum(self.mod_contrib_[:,axis],axis=1)
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Contrib"
    
    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                 pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': "black","lw":1.0}})
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
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if add_labels:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
        
    # Add supplementary categories
    if quali_sup:
        if self.quali_sup_labels_ is not None:
            if short_labels:
                mod_sup_labels = self.short_sup_labels_
            else:
                mod_sup_labels = self.mod_sup_labels_

            mod_sup_coord = pd.DataFrame(self.mod_sup_coord_,columns=self.dim_index_,index=mod_sup_labels)
            
            p = p + pn.geom_point(data=mod_sup_coord,
                                  mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                  color=color_sup,size=point_size,shape=marker_sup)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=mod_sup_coord,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=mod_sup_coord,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Qualitatives variables categories - FAMD"
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

    # Add theme
    p = p + ggtheme
    
    return p
    

def fviz_famd_var(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title=None,
                 color_quanti ="black",
                 color_quali = "blue",
                 color_quali_sup = "green",
                 color_quanti_sup = "red",
                 point_size = 1.5,
                 text_size = 8,
                 add_labels=True,
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
                 ggtheme=pn.theme_gray()) -> pn:
    """
    Draw the Multiple Factor Analysis for Mixed Data (FAMD) variables graphs
    ------------------------------------------------------------------------

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    col_cos2 = pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_)
    var_eta2 = pd.DataFrame(self.var_eta2_,index = self.quali_labels_,columns=self.dim_index_)
    
    # Initialize
    p = (pn.ggplot(data=col_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_cos2.index))+ 
         pn.geom_point(color=color_quanti,shape=marker_quanti,size=point_size,show_legend=False))
    
    if add_labels:
        if repel :
            p = p + text_label(text_type,color=color_quanti,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quanti,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color_quanti,size=text_size,va=va,ha=ha)
    
    # Add Qualitatives variables
    p = p + pn.geom_point(data=var_eta2,
                        mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_eta2.index),
                        color=color_quali,size=point_size,shape=marker_quali)
    if add_labels:
        if repel:
            p = p + text_label(text_type,data=var_eta2,
                                mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_eta2.index),
                                color=color_quali,size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali,"lw":1.0}})
        else:
            p = p + text_label(text_type,data=var_eta2,
                                mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_eta2.index),
                                color=color_quali,size=text_size,va=va,ha=ha)
    
    # Add supplementary continuous variables
    if add_quanti_sup:
        if self.quanti_sup_labels_ is not None:
            col_sup_cos2 = pd.DataFrame(self.col_sup_cos2_,columns=self.dim_index_,index=self.col_sup_labels_)
            p = p + pn.geom_point(data=col_sup_cos2,
                                  mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_cos2.index),
                                  color=color_quanti_sup,size=point_size,shape=marker_quanti_sup)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=col_sup_cos2,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_cos2.index),
                                    color=color_quanti_sup,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quanti_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=col_sup_cos2,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_cos2.index),
                                       color=color_quanti_sup,size=text_size,va=va,ha=ha)
    
    # Add supplementary categoricals variables
    if add_quali_sup:
        if self.quali_sup_labels_ is not None:
            quali_sup_eta2 = pd.DataFrame(self.quali_sup_eta2_,columns=self.dim_index_,index=self.quali_sup_labels_)
            p = p + pn.geom_point(data=quali_sup_eta2,
                                  mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                  color=color_quali_sup,size=point_size,shape=marker_quali_sup)
            if add_labels:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_eta2,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color=color_quali_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_eta2,
                                       mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color=color_quali_sup,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Graphe of variables - FAMD"
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
                - "col" for the correlation circle
                - "mod" for the categorical variables graphs
                - "var" for all the variables (quantitatives and categorical)
    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    figure : The individuals factor map and the variables factor map.

    Author
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if choice not in ["ind","col","mod","var"]:
        raise ValueError("Error : 'choice' values allowed are 'ind','col','mod' and 'var'.")
    
    if choice == "ind":
        return fviz_famd_ind(self,**kwargs)
    elif choice == "col":
        return fviz_famd_col(self,**kwargs)
    elif choice == "mod":
        return fviz_famd_mod(self,**kwargs)
    elif choice == "var":
        return fviz_famd_var(self,**kwargs)
    

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

##########################################################################################################
###### Principal Components Analysis with partial correlation matrix (PartialPCA)
###########################################################################################################

def fviz_ppca_ind(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 legend_title=None,
                 marker = "o",
                 add_grid =True,
                 add_hline = True,
                 add_vline=True,
                 add_labels = True,
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 repel=False,
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    """
    
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an instance of class PartialPCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

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
    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],
                                              name = legend_title)
        if add_labels:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
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
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if add_labels:
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - Partial PCA"
    
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

def fviz_ppca_var(self,
                 axis=[0,1],
                 title =None,
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 legend_title = None,
                 add_hline = True,
                 add_vline=True,
                 add_labels= None,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 repel=False,
                 add_circle = True,
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an instance of class PartialPCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

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

    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",colour=c), arrow = pn.arrow())
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title)
        if add_labels:
            if repel:
                p = p + text_label(text_type,mappping=pn.aes(colour=c),size=text_size,va=va,ha=ha,
                                adjust_text={'arrowprops': {'arrowstyle': '->','color': 'black','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if add_labels:
            if repel:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '->','color': color,'lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color="black", fill=None)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Variables factor map - Partial PCA"
    
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

def fviz_ppca(self,choice="ind",**kwargs)->plt:
    """
    
    """

    if choice == "ind":
        return fviz_ppca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_ppca_var(self,**kwargs)
    else:
        raise ValueError("Error : Allowed values are 'ind' or 'var'.")
    
###################################################################################################################################
#           Exploratory Factor Analysis (EFA)
#################################################################################################################################

def fviz_efa_ind(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="blue",
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    """
    
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an instance of class EFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if repel :
        p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                           adjust_text={'arrowprops': {'arrowstyle': '->','color': color,'lw':1.0}})
    else:
        p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - EFA"
    
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

def fviz_efa_var(self,
                 axis=[0,1],
                 title =None,
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 add_hline = True,
                 add_vline=True,
                 ha="center",
                 va="center",
                 limits = None,
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 repel=False,
                 add_circle = True,
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an instance of class EFA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if color == "contrib":
        midpoint = 50
        legend_title = "Contrib"
        c = np.sum(self.col_contrib_[:,axis],axis=1)
        if limits is None:
            limits = list([np.min(c),np.max(c)])
    
    if color == "contrib":
        # Add gradients colors
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",colour=c), arrow = pn.arrow())
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,name = legend_title)
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': 'black','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if repel:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color="black", fill=None)
    
    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Variables factor map - EFA"
    
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

def fviz_efa(self,choice="ind",**kwargs)->plt:
    """
    
    """

    if choice == "ind":
        return fviz_efa_ind(self,**kwargs)
    elif choice == "var":
        return fviz_efa_var(self,**kwargs)
    else:
        raise ValueError("Error : Allowed values are 'ind' or 'var'.")
    

##################################################################################################
#                       Visualize the contributions of row/column elements
###################################################################################################

def fviz_contrib(self,
                 choice="ind",
                 axis=None,
                 xlabel=None,
                 top_contrib=10,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 palette = "Set2",
                 short_labels=False,
                 ggtheme=pn.theme_gray()) -> plt:
    
    """ Plot the row and column contributions graph
            
    For the selected axis, the graph represents the row or column
    cosines sorted in descending order.            
        
    Parameters
    ----------
    choice : {'ind','var','mod'}.
            'ind' :   individuals
            'var' :   continues/categorical variables
            'mod' :   categories
        
    axis : None or int.
        Select the axis for which the row/col contributions are plotted. If None, axis = 0.
        
    xlabel : None or str (default).
        The label text.
        
    top_contrib : None or int.
        Set the maximum number of values to plot.
        If top_contrib is None : all the values are plotted.
            
    bar_width : None, float or array-like.
        The width(s) of the bars.

    add_grid : bool or None, default = True.
        Whether to show the grid lines.

    color : color or list of color, default = "steelblue".
        The colors of the bar faces.

    short_labels : bool, default = False
        
    Returns
    -------
    None
    """    
        
    if choice not in ["ind","var","mod"]:
        raise ValueError("Error : 'choice' not allowed.")

    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise ValueError("Error : 'axis' must be an integer.")
    elif axis < 0 or axis > self.n_components_:
        raise ValueError(f"Error : 'axis' must be an integer between 0 and {self.n_components_ - 1}.")
            
    if xlabel is None:
        xlabel = "Contributions (%)"
            
    if bar_width is None:
        bar_width = 0.5
    if top_contrib is None:
        top_contrib = 10
    elif not isinstance(top_contrib,int):
        raise ValueError("Error : 'top_contrib' must be an integer.")
        
    if choice == "ind":
        name = "individuals"
        contrib = self.row_contrib_[:,axis]
        labels = self.row_labels_
        if self.model_ == "ca":
            name = "rows"
    elif choice == "var":
        if self.model_ != "mca":
            name = "continues variables"
            contrib = self.col_contrib_[:,axis]
            labels  = self.col_labels_
            if self.model_ == "ca":
                name = "columns"
            if self.model_ == "famd":
                contrib = np.append(contrib,self.var_contrib_[:,axis],axis=0)
                labels = labels + self.quali_labels_
                name = "Variables"
        else:
            name = "Categorical variables"
            contrib = self.var_contrib_[:,axis]
            labels = self.var_labels_     
    elif choice == "mod" and self.model_ in ["mca","famd"]:
        name = "categories"
        contrib = self.mod_contrib_[:,axis]
        if short_labels:
            labels = self.short_labels_
        else:
            labels = self.mod_labels_
    
    n = len(labels)
    n_labels = len(labels)
        
    if (top_contrib is not None) & (top_contrib < n_labels):
        n_labels = top_contrib
        
    limit = n - n_labels
    contrib_sorted = np.sort(contrib)[limit:n]
    labels_sort = pd.Series(labels)[np.argsort(contrib)][limit:n]

    # Add group
    if (choice == "var" and self.model_ == "mfa"):
        group_sort = pd.Series(self.col_group_labels_)[np.argsort(contrib)][limit:n]

    # Add hline
    if self.model_ == "pca":
        hvalue = 100/len(self.col_labels_)
    elif self.model_ == "ca":
        hvalue = 100/(min(len(self.row_labels_)-1,len(self.col_labels_)-1))
    elif self.model_ == "mca":
        hvalue = 100/len(self.mod_labels_)
    elif self.model_ == "famd":
        hvalue = 100/(len(self.quanti_labels_) + len(self.mod_labels_) - len(self.quali_labels_))
    elif self.model_ == "mfa":
        if self.global_pca_.model_ == "pca":
            hvalue = 100/len(self.col_labels_)
        elif self.global_pca_.model_ == "mca":
            hvalue = 100/len(self.mod_labels_)

    df = pd.DataFrame({"labels" : labels_sort, "contrib" : contrib_sorted})

    if (choice == "var" and self.model_ == "mfa"):
        df["Groups"] = group_sort
        p = (pn.ggplot(df,pn.aes(x = "reorder(labels,contrib)", y = "contrib",fill="Groups"))+
             pn.geom_bar(stat="identity",width=bar_width))
        if palette is not None:
            p = p + pn.scale_color_brewer(type="qual",palette=palette)
    else:
        p = (pn.ggplot(df,pn.aes(x = "reorder(labels,contrib)", y = "contrib"))+
             pn.geom_bar(stat="identity",fill=color,width=bar_width))

    title = f"Contribution of {name} to Dim-{axis+1}"
    p = p + pn.ggtitle(title)+pn.xlab(name)+pn.ylab(xlabel)
    p = p + pn.coord_flip()
    
    if not (choice == "var" and self.model_ =="mca"):
        p = p + pn.geom_hline(yintercept=hvalue,linetype="dashed",color="red")

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"),
                         axis_text_x = pn.element_text(angle = 90, ha = "center", va = "center"))

    return p+ggtheme

##################################################################################################
#                       Visualize the cosines of row/column elements
###################################################################################################

def fviz_cosines(self,
                 choice="ind",
                 axis=None,
                 xlabel=None,
                 top_cos2=10,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 short_labels=False,
                 ggtheme=pn.theme_gray()) -> plt:
    
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

    if choice not in ["ind","var","mod","quanti_sup","quali_sup","ind_sup"]:
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



#################################################################################################################
#                   Hierarchical Clustering on Principal Components (HCPC)
#################################################################################################################

def fviz_hcpc_cluster(self,
                      axis=(0,1),
                      xlabel=None,
                      ylabel=None,
                      title=None,
                      legend_title = None,
                      xlim=None,
                      ylim=None,
                      show_clust_cent = False, 
                      cluster = None,
                      center_marker_size=5,
                      marker = None,
                      add_labels = True,
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
        (axis[1] > self.factor_model_.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if legend_title is None:
        legend_title = "cluster"

    coord = pd.DataFrame(self.factor_model_.row_coord_,index = self.labels_,columns=self.factor_model_.dim_index_)

    if cluster is None:
        coord = pd.concat([coord,self.cluster_],axis=1)
    else:
        coord = pd.concat([coord,cluster],axis=1)
        
    # Rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]
     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    p = p + pn.geom_point(pn.aes(color = "cluster"),size=point_size,shape=marker)
    if add_labels:
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(color="cluster"),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color="cluster"),size=text_size,va=va,ha=ha)
    if add_ellipse:
        p = (p + pn.geom_point(pn.aes(color = "cluster"))+ 
                 pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill="cluster"),type = ellipse_type,alpha = 0.25,level=confint_level))
    
    if show_clust_cent:
        cluster_center = self.cluster_centers_
        p = p + pn.geom_point(cluster_center,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=cluster_center.index,color=cluster_center.index),
                              size=center_marker_size)
    
    # Add additionnal        
    proportion = self.factor_model_.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - HCPC"
    
    if xlim is not None:
        p = p + pn.xlim(xlim)
    if ylim is not None:
        p = p + pn.ylim(ylim)
   
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=xlabel)+pn.ylab(ylab=ylabel)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    
    p = p + pn.labs(color = legend_title)

    p = p + ggtheme

    return p
    

def fviz_corrplot(X,
                  method = "square",
                  type = "full",
                  xlabel=None,
                  ylabel=None,
                  title=None,
                  outline_color = "gray",
                  colors = ["blue","white","red"],
                  legend_title = "Corr",
                  is_corr = False,
                  show_legend = True,
                  ggtheme = pn.theme_minimal(),
                  show_diag = None,
                  hc_order = False,
                  hc_method = "complete",
                  lab = False,
                  lab_col = "black",
                  lab_size = 11,
                  p_mat = None,
                  sig_level=0.05,
                  insig = "pch",
                  pch = 4,
                  pch_col = "black",
                  pch_cex = 5,
                  tl_cex = 12,
                  tl_col = "black",
                  tl_srt = 45,
                  digits = 2
                  ):
    
    if not isinstance(X,pd.DataFrame):
        raise ValueError("Error : 'X' must be a DataFrame.")
    
    if is_corr:
        p = ggcorrplot(x=X,
                       method=method,
                       type=type,
                       ggtheme = ggtheme,
                       title = title,
                       show_legend = show_legend,
                       legend_title = legend_title,
                       show_diag = show_diag,
                       colors = colors,
                       outline_color = outline_color,
                       hc_order = hc_order,
                       hc_method = hc_method,
                       lab = lab,
                       lab_col = lab_col,
                       lab_size = lab_size,
                       p_mat = p_mat,
                       sig_level=sig_level,
                       insig = insig,
                       pch = pch,
                       pch_col = pch_col,
                       pch_cex = pch_cex,
                       tl_cex = tl_cex,
                       tl_col = tl_col,
                        tl_srt = tl_srt,
                        digits = digits)
    else:
        X.columns = pd.Categorical(X.columns,categories=X.columns)
        X.index = pd.Categorical(X.index,categories=X.index)
        melt = get_melt(X)

        p = (pn.ggplot(melt,pn.aes(x="Var1",y="Var2",fill="value")) + 
             pn.geom_point(pn.aes(size="value"),color=outline_color,shape="o")+pn.guides(size=None)+pn.coord_flip())

        # Adding colors
        p =p + pn.scale_fill_gradient2(low = colors[0],high = colors[2],mid = colors[1],name = legend_title) 

        # Add theme
        p = p + ggtheme

        # Add axis elements and title
        if title is None:
            title = "Correlation"
        if ylabel is None:
            ylabel = "Dimensions"
        if xlabel is None:
            xlabel = "Variables"
        p = p + pn.labs(title=title,x=xlabel,y=ylabel)
            
        # Removing legend
        if not show_legend:
            p =p+pn.theme(legend_position=None)
    
    return p
    
####################################################### Multiple Factor Analysie (MFA) plot #################################################


def fviz_mfa_ind(self,
                 axis=[0,1],
                 xlim=None,
                 ylim=None,
                 title =None,
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 add_grid =True,
                 add_labels=True,
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
                 short_labels=True,
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
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add categorical supplementary variables
    if habillage is not None:
        if self.group_sup_ is not None:
            # Check if habillage in columns (level) #https://kanoki.org/2022/07/25/pandas-select-slice-rows-columns-multiindex-dataframe/
            if habillage in self.data_.columns.levels[1]:
                # Extract habillage from Data
                data = self.data_.loc[:, (slice(None), habillage)].droplevel(0, axis=1)
                # Check if category or object
                if data.dtypes[0] in ["object","category"]:
                    coord = pd.concat([coord,data],axis=1) 
                else:
                    raise ValueError("Error : 'habillage' must be an object of category")  
    
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

    if habillage is None :        
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
            
            if add_ellipse:
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=c),type = ellipse_type,alpha = 0.25,level=confint_level)
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if add_labels:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.group_sup_ is not None:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if add_labels:
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
            
    
    #if ind_sup:
    #    if self.row_sup_labels_ is not None:
    #        sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)

    #        p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
    #                              color = color_sup,shape = marker_sup,size=point_size)
    #        if add_labels:
    #            if repel:
    #                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
    #                                    color=color_sup,size=text_size,va=va,ha=ha,
    #                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
    #            else:
    #                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
    #                                    color = color_sup,size=text_size,va=va,ha=ha)
    #if quali_sup:
    #    if self.quali_sup_labels_ is not None:
    #        if habillage is None:
    #            if short_labels:
    #                mod_sup_labels = self.short_sup_labels_
    #            else:
    #                mod_sup_labels = self.mod_sup_labels_

    #            mod_sup_coord = pd.DataFrame(self.mod_sup_coord_,columns=self.dim_index_)
    #            
    #            p = p + pn.geom_point(mod_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
    #                                  color=color_quali_sup,size=point_size)
                
    #            if add_labels:
    #                if repel:
    #                    p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
    #                                       color=color_quali_sup,size=text_size,va=va,ha=ha,
    #                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,'lw':1.0}})
    #                else:
    #                    p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
    #                                       color ="red",size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - MFA"
    
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


def fviz_mfa_col(self,
                 axis=[0,1],
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