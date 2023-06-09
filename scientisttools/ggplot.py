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
        
    if self.model_ not in ["pca","ca","mca","famd","mfa","cmds","disqual","mixdisc"]:
        raise ValueError("'res' must be an object of class PCA, CA, MCA, FAMD, MFA, CMDS, DISQUAL, MIXDISC")

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

####################################################################################
#       Principal Components Analysis (PCA)
####################################################################################

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
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black",'lw':1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color,'lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup_labels_ is not None:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if repel:
                p = p + text_label(text_type,pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','lw':1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            
            if add_ellipse:
                p = p + pn.geom_point(pn.aes(color = habillage))
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)

            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                     color=color_sup,size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,'lw':1.0}})
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
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black",'lw':1.0}})
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
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': 'black','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if repel:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': color,'lw':1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if quanti_sup:
        if self.quanti_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=sup_coord.iloc[:,axis[0]],yend=sup_coord.iloc[:,axis[1]]),arrow = pn.arrow(),color=color_sup)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(x=sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.col_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,'lw':1.0}})
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

def fviz_pca(self,choice="ind",**kwargs)->plt:
    """
    
    """

    if choice == "ind":
        return fviz_pca_ind(self,**kwargs)
    elif choice == "var":
        return fviz_pca_var(self,**kwargs)
    else:
        raise ValueError("Error : Allowed values are 'ind' or 'var'.")


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
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black",'lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color,'lw':1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup_labels_ is not None:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color':"black","lw":1.0}})
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
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,'lw':1.0}})
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
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': "black",'lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
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
            
            p = p + pn.geom_point(mod_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
                                  color=color_sup,size=point_size,shape=marker_sup)

            if repel:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,'lw':1.0}})
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
    

def fviz_mca_var(self,
                 axis=[0,1],
                 xlim=(0,1),
                 ylim=(0,1),
                 title=None,
                 color="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 marker="o",
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an instance of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.var_eta2_,index =  self.var_labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.var_labels_))

    if color == "cos2":
        limits = [0,1]
        legend_title = "cos2"
        midpoint = 0.5
        c = np.sum(self.var_cos2_[:,axis],axis=1)
    elif color == "contrib":
        midpoint = 50
        limits = [0,100]
        legend_title = "Contrib"
        c = np.sum(self.var_contrib_[:,axis],axis=1)
    
    # Using cosine and contributions
    if color in ["cos2","contrib"]:
        # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': "black",'lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
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

def fviz_mca(self,choice="ind",**kwargs)->plt:
    """
    
    
    """
    if choice == "ind":
        return fviz_mca_ind(self,**kwargs)
    elif choice == "mod":
        return fviz_mca_mod(self,**kwargs)
    elif choice == "var":
        return fviz_mca_var(self,**kwargs)
    else:
        raise ValueError("Error : Allowed values are 'ind', 'mod' or 'var'.")


#################################################################################################################
#               Correspondence Analysis (CA)
#################################################################################################################

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
                 row_sup=True,
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
                 repel=False,
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an instance of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)
    
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
 
        # Using cosine and contributions
    if color in ["cos2","contrib"]:
        # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if row_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(pn.aes(x = sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.row_sup_labels_),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(x = sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.row_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(x = sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.row_sup_labels_),
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
                 col_sup=True,
                 color_sup = "red",
                 marker_sup = "^",
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    """
    
    
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an instance of class CA.")
    
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
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,name = legend_title)
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': 'black','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),color=color,shape=marker,size=point_size)
        if repel:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if col_sup:
        if self.col_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p  = p+pn.geom_point(sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.col_sup_labels_),
                                 color=color_sup,shape=marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(x=sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.col_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
            else:
                p  = p + text_label(text_type,mapping=pn.aes(x=sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.col_sup_labels_),
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

def fviz_ca(self,choice,**kwargs)->plt:
    """
    
    
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an instance of class CA.")


    if choice == "row":
        return fviz_ca_row(self,**kwargs)
    elif choice == "col":
        return fviz_ca_col(self,**kwargs)
    else:
        raise ValueError("Error : Allowed values for choice are :'row' or 'col'.")


####################################################################################################################
#               Factor Analyis of Mixed Data (FAMD)
####################################################################################################################

def fviz_famd_ind(self,
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
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an instance of class FAMD.")
    
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
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if repel :
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup_labels_ is not None:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if repel:
                p = p + text_label(text_type,pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->',"color":"black","lw":1.0}})
            else:
                p = p + text_label(text_type,pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            
            if add_ellipse:
                p = p + pn.geom_point(pn.aes(color = habillage))
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.row_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)

            p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_sup_labels_),
                                     color=color_sup,size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
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
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
            else:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
                                   color ="red",size=text_size,va=va,ha=ha)

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
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an instance of class FAMD.")
    
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
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': 'black',"lw":1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
        if repel:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if quanti_sup:
        if self.quanti_sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
            p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=sup_coord.iloc[:,axis[0]],yend=sup_coord.iloc[:,axis[1]]),arrow = pn.arrow(),color=color_sup)
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(x=sup_coord.iloc[:,axis[0]],y=sup_coord.iloc[:,axis[1]],label=self.col_sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
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
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an instance of class FAMD.")
    
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
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
        if repel :
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
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
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
            else:
                p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_labels),
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
                 xlim=(0,1),
                 ylim=(0,1),
                 title=None,
                 color="blue",
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 marker="o",
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
                 ggtheme=pn.theme_gray()) -> plt:
    
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an instance of class FAMD.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    col_cos2 = pd.DataFrame(self.col_cos2_[:,axis],index = self.col_labels_,columns=self.dim_index_)
    var_eta2 = pd.DataFrame(self.var_eta2_[:,axis],index = self.quali_labels_,columns=self.dim_index_)
    coord = pd.concat([col_cos2,var_eta2],axis=0)
    labels = self.col_labels_ + self.quali_labels_

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=labels))

    if color == "contrib":
        midpoint = 50
        limits = [0,100]
        legend_title = "Contrib"
        contrib = np.append(self.col_contrib_[:,axis],self.var_contrib_[:,axis],axis=0)
        c = np.sum(contrib,axis=1)
    
    # Using cosine and contributions
    if color == "contrib":
        # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
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


def fviz_famd(self,choice="ind",**kwargs)->plt:
    """
    
    
    """
    if choice == "ind":
        return fviz_famd_ind(self,**kwargs)
    elif choice == "col":
        return fviz_famd_col(self,**kwargs)
    elif choice == "mod":
        return fviz_famd_mod(self,**kwargs)
    elif choice == "var":
        return fviz_famd_var(self,**kwargs)
    else:
        raise ValueError("Error : Allowed values are 'ind', 'col', 'mod' or 'var'.")
    

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
            repel=False,
            ggtheme=pn.theme_gray()) -> plt:
    """
    
    
    """
    
    if self.model_ != "cmds":
        raise ValueError("Error : 'self' must be an instance of class CMDSCALE.")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid axis")
    
    coord = pd.DataFrame(self.coord_,index = self.labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.labels_))

    # Add point
    p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if repel :
        p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                    adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
    else:
        p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if add_sup:
        if self.sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.sup_coord_, index= self.sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(data=sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.sup_labels_),
                                  color=color_sup,size=point_size,shape=marker_sup)

            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
            else:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.sup_labels_),
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
    
    if self.model_ != "mds":
        raise ValueError("Error : 'self' must be an instance of class MDS.")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid axis")
    
    coord = pd.DataFrame(self.coord_,index = self.labels_,columns=self.dim_index_)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.labels_))

    # Add point
    p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
    if repel :
        p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,
                    adjust_text={'arrowprops': {'arrowstyle': '->','color': color,"lw":1.0}})
    else:
        p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    if add_sup:
        if self.sup_labels_ is not None:
            sup_coord = pd.DataFrame(self.sup_coord_, index= self.sup_labels_,columns=self.dim_index_)
            p = p + pn.geom_point(data=sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.sup_labels_),
                                  color=color_sup,size=point_size,shape=marker_sup)
            if repel:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.sup_labels_),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '->','color': color_sup,"lw":1.0}})
            else:
                p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.sup_labels_),
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
    
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an instance of class PartialPCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

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

     
    # Using cosine and contributions
    if color in ["cos2","contrib"]:
            # Add gradients colors
        p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
        p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],midpoint=midpoint,limits = limits,
                                              name = legend_title)
        if repel :
            p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                     adjust_text={'arrowprops': {'arrowstyle': '->','color': "black","lw":1.0}})
        else:
            p = p + text_label(text_type,pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
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
    
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an instance of class PartialPCA.")
    
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
                               adjust_text={'arrowprops': {'arrowstyle': '->','color': 'black','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    else:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), arrow = pn.arrow(),color=color)
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
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_labels_))

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
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.col_labels_))

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
    elif choice == "var" and self.model_ != "mca":
        name = "continues variables"
        contrib = self.col_contrib_[:,axis]
        labels  = self.col_labels_
        if self.model_ == "famd":
            contrib = np.append(contrib,self.var_contrib_[:,axis],axis=0)
            labels = labels + self.quali_labels_
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

    df = pd.DataFrame({"labels" : labels_sort, "contrib" : contrib_sorted})

    p = pn.ggplot(df,pn.aes(x = "reorder(labels,contrib)", y = "contrib"))+pn.geom_bar(stat="identity",fill=color,width=bar_width)

    title = f"Contribution of {name} to Dim-{axis+1}"
    p = p + pn.ggtitle(title)+pn.xlab(name)+pn.ylab(xlabel)
    p = p + pn.coord_flip()

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"),
                         axis_text_x = pn.element_text(angle = 60, ha = "center", va = "center"))

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
        cos2 = self.row_cos2_[:,axis]
        labels = self.row_labels_
    elif choice == "var" and self.model_ != "mca":
        name = "continues variables"
        cos2 = self.col_cos2_[:,axis]
        labels  = self.col_labels_
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
            raise ValueError("Error : 'quanti_sup'")
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
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if legend_title is None:
        legend_title = "cluster"

    coord = pd.DataFrame(self.row_coord_,index = self.row_labels_,columns=self.dim_index_)

    if cluster is None:
        coord = pd.concat([coord,self.cluster_],axis=1)
    else:
        coord = pd.concat([coord,cluster],axis=1)
        
    # Rename last columns
    coord.columns = [*coord.columns[:-1], legend_title]

     # Extract coordinates
     # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=self.row_labels_))

    p = p + pn.geom_point(pn.aes(color = "cluster"),size=point_size,shape=marker)
    if repel:
        p = p + pn.geom_text(pn.aes(color="cluster"),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '->','lw':1.0}})
    else:
        p = p + pn.geom_text(pn.aes(color="cluster"),size=text_size,va=va,ha=ha)
    if add_ellipse:
        p = p + pn.geom_point(pn.aes(color = "cluster"))
        p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill="cluster"),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if show_clust_cent:
        cluster_center = self.cluster_centers_
        p = p + pn.geom_point(cluster_center,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=cluster_center.index,color=cluster_center.index),
                              size=center_marker_size)
    
    # Add additionnal        
    proportion = self.eig_[2]
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
    



