# -*- coding: utf-8 -*-

import plotnine as pn
import numpy as np
import pandas as pd
from scientisttools.extractfactor import get_eigenvalue
import matplotlib.pyplot as plt

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
                   h_align= "center",
                   v_align = "bottom",
                   title=None,
                   x_label=None,
                   y_label=None,
                   ggtheme=pn.theme_gray())-> plt:
    """
    
    
    """
        
    if self.model_ not in ["pca","ca","mca","famd","mfa","cmds"]:
        raise ValueError("'res' must be an object of class PCA, CA, MCA, FAMD, MFA, CMDS")

    eig = get_eigenvalue(self)
    eig = eig.iloc[:min(n_components,self.n_components_),:]

    if choice == "eigenvalue":
        eig = eig["eigenvalue"]
        text_labels = list([str(np.around(x,3)) for x in eig.values])
        if self.model_ != "mds":
            kaiser = self.kaiser_threshold_
        if y_label is None:
            y_label = "Eigenvalue"
    elif choice == "proportion":
        eig = eig["proportion"]
        text_labels = list([str(np.around(x,1))+"%" for x in eig.values])
        if self.model_ !=  "pca":
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
        p = p + pn.geom_text(label=text_labels,ha = h_align,va = v_align)
    if add_kaiser :
        p = (p +  pn.geom_hline(yintercept=kaiser,linetype="--", color="red")+\
                  pn.annotate("text", x=int(np.median(np.arange(1,len(eig)+1))), y=kaiser, 
                              label="Kaiser threshold"))

    if add_kss:
        if self.model_ in ["pca","ppca"]:
            if choice == "eigenvalue":
                p = (p  +   pn.geom_hline(yintercept=self.kss_threshold_,linetype="--", color="yellow")+ \
                            pn.annotate("text", x=int(np.mean(np.arange(1,len(eig)+1))), y=self.kss_threshold_, 
                                        label="Karlis - Saporta - Spinaki threshold",colour = "yellow"))
            else:
                raise ValueError("'add_kss' is only with 'choice=eigenvalue'.")
        else:
            raise ValueError("'add_kss' is only for class PCA.")
    if add_broken_stick:
        if choice == "eigenvalue":
            if self.model_ == ["pca","ppca"]:
                bst = self.broken_stick_threshold_[:min(n_components,self.n_components_)]
                p = (p  +   pn.geom_line(pn.aes(x="dim",y=bst),color="green",linetype="--")+\
                            pn.geom_point(pn.aes(x="dim",y=bst),colour="green")+\
                            pn.annotate("text", x=int(np.mean(np.arange(1,len(eig)+1))), y=np.median(bst), 
                                        label="Broken stick threshold",colour = "green"))
        else:
            raise ValueError("'add_broken_stick' is only for class PCA.")

    if title is None:
        title = "Scree plot"
    if x_label is None:
        x_label = "Dimensions"
    if y_label is None:
        y_label = "Percentage of explained variances"
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    p = p + ggtheme
    return p

def fviz_eigenvalue(self,**kwargs):
    return fviz_screeplot(self,**kwargs)

def fviz_eig(self,**kwargs):
    return fviz_screeplot(self,**kwargs)

###### PCA plotnine

def fviz_pca_ind(self,
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
                 ind_sup=False,
                 color_sup = "red",
                 marker_sup = "^",
                 add_ellipse=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 hotelling_ellipse=False,
                 habillage = None,
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    """
    
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an instance of class PCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add categorical supplementary variables
    if self.quali_sup_labels_ is not None:
        coord[self.quali_sup_labels] = self.data_[self.quali_sup_labels_]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_labels_))

    if color == "cos2":
        limits = [0,1]
        legend_title = "cos2"
        midpoint = 0.5
        c = np.sum(self.row_cos2_[:,axis],axis=1)
    elif color == "contrib":
        midpoint = 50
        limits = [0,100]
        legend_title = "Contrib"
        c = np.sum(self.row_contrib_[:,axis],axis=1)

    if habillage is None :        
        # Using cosine and contributions
        if color in ["cos2","contrib"]:
            # Add gradients colors
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                     adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': "black"}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': color}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup_labels_ is not None:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if repel:
                p = p + text_label(text_type,pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                     adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->'}})
            else:
                p = p + text_label(text_type,pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            
            if add_ellipse:
                p = p + pn.geom_point(pn.aes(color = habillage))
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)

            p = p + pn.geom_point(sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                     color=color_sup,size=text_size,va=va,ha=ha,
                                     adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': color_sup}})
            else:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                     color = color_sup,size=text_size,va=va,ha=ha)
    if self.quali_sup_labels_ is not None:
        if habillage is None:
            if short_labels:
                mod_sup_labels = self.short_sup_labels_
            else:
                mod_sup_labels = self.mod_sup_labels_

            mod_sup_coord = pd.DataFrame(self.mod_sup_coord_,columns=self.dim_index_)
            
            p = p + pn.geom_point(mod_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),color="red",size=point_size)

            if repel:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
                                   color="red",size=text_size,va=va,ha=ha,
                                     adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': "red"}})
            else:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
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
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 text_size = 8,
                 add_grid =True,
                 quanti_sup=True,
                 color_sup = "red",
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
    
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an instance of class PCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.col_labels_))

    if color == "cos2":
        legend_title = "cos2"
        midpoint = 0.5
        c = np.sum(self.col_cos2_[:,axis],axis=1)
        if limits is None:
            limits = list([np.min(c),np.max(c)])
    elif color == "contrib":
        midpoint = 50
        legend_title = "Contrib"
        c = np.sum(self.col_contrib_[:,axis],axis=1)
        if limits is None:
            limits = list([np.min(c),np.max(c)])
    
    if color in ["cos2","contrib"]:
        # Add gradients colors
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",colour=c), arrow = pn.arrow())
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,name = legend_title)
        if repel:
            p = p + text_label(text_type,mappping=pn.aes(colour=c),size=text_size,va=va,ha=ha,
                               adjust_text={'expand_points': (-1,1),'arrowprops': {'arrowstyle': '->','color': 'red'}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if repel:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'expand_points': (0,0)})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if quanti_sup:
        if self.quanti_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=sup_coord.iloc[:,axis[0]],yend=sup_coord.iloc[:,axis[1]]),arrow = pn.arrow(),color=color_sup)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(x=sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.col_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'expand_points': (0,0)})
            else:
                p  = p + text_label(text_type,mapping=pn.aes(x=sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.col_sup_labels_),
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

######################################################################################################
##                             Multiple Correspondence Analysis (MCA)
######################################################################################################

def fviz_mca_ind(self,
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
                 ind_sup=True,
                 color_sup = "red",
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an instance of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    # Add categorical supplementary variables
    if self.quali_sup_labels_ is not None:
        coord[self.quali_sup_labels] = self.data_[self.quali_sup_labels_]
    
    if color == "cos2":
        limits = [0,1]
        legend_title = "cos2"
        midpoint = 0.5
        c = np.sum(self.row_cos2_[:,axis],axis=1)
    elif color == "contrib":
        midpoint = 50
        limits = [0,100]
        legend_title = "Contrib"
        c = np.sum(self.row_contrib_[:,axis],axis=1)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_labels_))

    if habillage is None :        
        # Using cosine and contributions
        if color in ["cos2","contrib"]:
            # Add gradients colors
            p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
            p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                     adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': "black"}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': color}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup_labels_ is not None:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                     adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->'}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            
            if add_ellipse:
                p = p + pn.geom_point(pn.aes(color = habillage))
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(mapping=pn.aes(x = sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.row_sup_labels_),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(x = sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.row_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': color_sup}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(x = sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.row_sup_labels_),
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
                 color ="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 add_grid =True,
                 quali_sup = True,
                 color_sup = "red",
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
                 repel=False,
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an instance of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.mod_coord_,index = self.mod_labels_,columns=self.dim_index_)

    # Categories labels
    if short_labels:
        labels = self.short_labels_
    else:
        labels = self.mod_labels_

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=labels))

    if color == "cos2":
        limits = [0,1]
        legend_title = "cos2"
        midpoint = 0.5
        c = np.sum(self.mod_cos2_[:,axis],axis=1)
    elif color == "contrib":
        midpoint = 50
        limits = [0,100]
        legend_title = "Contrib"
        c = np.sum(self.mod_contrib_[:,axis],axis=1)
     
        # Using cosine and contributions
    if color in ["cos2","contrib"]:
        # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                               adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': "black"}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': color}})
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
            
            p = p + pn.geom_point(data=mod_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
                                  color=color_sup,size=point_size,shape=marker_sup)

            if repel:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'expand_points': (2, 2),'arrowprops': {'arrowstyle': '->','color': color_sup}})
            else:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
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
    

def fviz_mca_var(self.
                 axis=[0,1],
                 xlim=(None,None),
                 ylim=(None,None),
                 title=None,
                 color="blue",
                 marker="o",
                 add_grid =True,
                 ind_sup=False,
                 color_sup = "red",
                 marker_sup ="^",
                 hotelling_ellipse=False,
                 habillage=None,
                 short_labels=False,
                 add_mod_sup=True,
                 color_map ="jet",
                 add_hline = True,
                 add_vline =True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 random_state=None,
                 repel=False,
                 ax=None) -> plt:
    