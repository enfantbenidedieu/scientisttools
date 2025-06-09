# -*- coding: utf-8 -*-
from plotnine import ggplot,aes,geom_segment,annotate,arrow,theme_minimal
from numpy import asarray

#intern functions
from .fviz_add import text_label,fviz_add,gg_circle

def fviz_corrcircle(self,
                    axis = [0,1],
                    geom = ["arrow","text"],
                    x_label = None,
                    y_label = None,
                    title = None,
                    alpha_var = 1,
                    col_var = "black",
                    linetype_var = "solid",
                    line_size_var = 0.5,
                    arrow_length_var =0.1,
                    arrow_angle_var = 10,
                    arrow_type_var = "closed",
                    text_type_var = "text",
                    text_size_var = 8,
                    alpha_quanti_sup = 1,
                    col_quanti_sup = "blue",
                    linetype_quanti_sup = "dashed",
                    line_size_quanti_sup = 0.5,
                    arrow_length_quanti_sup =0.1,
                    arrow_angle_quanti_sup = 10,
                    arrow_type_quanti_sup = "closed",
                    text_type_quanti_sup = "text",
                    text_size_quanti_sup = 8,
                    add_circle = True,
                    col_circle = "gray",
                    add_grid=True,
                    add_hline = True,
                    alpha_hline = 0.5,
                    col_hline = "black",
                    size_hline = 0.5,
                    linetype_hline = "dashed",
                    add_vline = True,
                    alpha_vline = 0.5,
                    col_vline = "black",
                    size_vline = 0.5,
                    linetype_vline = "dashed",
                    ha_var = "center",
                    va_var = "center",
                    ggtheme=theme_minimal()):
    """
    Draw correlation circle
    -----------------------

    Description
    -----------


    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if self.model_ not in ["pca","ca","mca","specificmca","famd","mpca","pcamix","mfa","mfaqual","mfamix","partialpca","efa"]:
        raise TypeError("'self' must be an object of class PCA, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, PartialPCA, EFA")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    if self.model_ in ["pca","partialpca","efa"]:
        coord = self.var_.coord
    elif self.model_ in ["famd","mpca","pcamix","mfa","mfamix"]:
        coord = self.quanti_var_.coord
    else:
        if hasattr(self, "quanti_sup_"):
            coord = self.quanti_sup_.coord
        if hasattr(self, "quanti_var_sup_"):
            coord = self.quanti_var_sup_.coord

    # Initialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "arrow" in geom:
        p = p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"),alpha=alpha_var,color=col_var,linetype=linetype_var,size=line_size_var,arrow = arrow(length=arrow_length_var,angle=arrow_angle_var,type=arrow_type_var))
    if "text" in geom:
            p = p + text_label(text_type_var,False,color=col_var,size=text_size_var,ha=ha_var,va=va_var)
        
    if self.model_ in ["pca","famd","mpca","pcamix","mfa","mfamix"]:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_.coord
        elif hasattr(self, "quanti_var_sup_"):
            sup_coord = self.quanti_var_sup_.coord
        
        if hasattr(self, "quanti_sup_") or hasattr(self, "quanti_var_sup_"):
            if "arrow" in geom:
                p  = p + annotate("segment",x=0,y=0,xend=asarray(sup_coord.iloc[:,axis[0]]),yend=asarray(sup_coord.iloc[:,axis[1]]),alpha=alpha_quanti_sup,color=col_quanti_sup,linetype=linetype_quanti_sup,size=line_size_quanti_sup,arrow = arrow(length=arrow_length_quanti_sup,angle=arrow_angle_quanti_sup,type=arrow_type_quanti_sup))
            if "text" in geom:
                p  = p + text_label(text_type_quanti_sup,False,data=sup_coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),color=col_quanti_sup,size=text_size_quanti_sup,ha=ha_var,va=va_var)
    
    # Create circle
    if add_circle:
        p = p + gg_circle(r=1.0, xc=0.0, yc=0.0, color=col_circle, fill=None)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add additionnal informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Correlation circle"
    p = fviz_add(p,self,axis,x_label,y_label,title,(-1,1),(-1,1),add_hline,alpha_hline,col_hline,linetype_hline,size_hline,add_vline,alpha_vline,col_vline,linetype_vline,size_vline,add_grid,ggtheme)     
    
    return p