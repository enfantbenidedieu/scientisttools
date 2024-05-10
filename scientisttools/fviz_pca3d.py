# -*- coding: utf-8 -*-
import plotnine as pn
import plotnine3d as pn3d
import pandas as pd
import numpy as np

from .text3d_label import text3d_label

def fviz_pca3d_ind(self,
                   axis=[0,1,2],
                   x_lim=None,
                   y_lim=None,
                   x_label = None,
                   y_label = None,
                   z_label = None,
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
                   habillage = None,
                   quali_sup = True,
                   color_quali_sup = "red",
                   ha="center",
                   va="center",
                   repel=False,
                   lim_cos2 = None,
                   lim_contrib = None,
                   ggtheme=pn.theme_minimal()) -> pn3d:
    
    """
    Draw the Principal Component Analysis (PCA) individuals graphs
    --------------------------------------------------------------

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    

    if ((len(axis) !=3) or 
        (axis[0] < 0) or 
        (axis[2] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1]) or 
        (axis[0] > axis[2]) or 
        (axis[1] > axis[2])) :
        raise ValueError("You must pass a valid 'axis'.")

    #### Extract individuals coordinates
    coord = self.ind_["coord"]

    # Add Active Data
    coord = pd.concat([coord,self.call_["X"]],axis=1)

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
    
    if isinstance(color,str):
        if color == "cos2":
            coord["cos2"] = self.ind_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Cos2"
        elif color == "contrib":
            coord["contrib"] = self.ind_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
        elif color in coord.columns.tolist():
            if not np.issubdtype(coord[color].dtype, np.number):
                raise ValueError("Error : 'color' must me a numeric variable.")
            c = coord[color].values
            if legend_title is None:
                legend_title = color
    elif isinstance(color,np.ndarray):
        coord["num_var"] = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    elif hasattr(color, "labels_"):
        coord["cluster"] = [str(x+1) for x in color.labels_]
        if legend_title is None:
            legend_title = "Cluster"

    # Initialize
    p = pn3d.ggplot_3d(data=coord) + pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",z=f"Dim.{axis[2]+1}",label=coord.index)

    if habillage is None :  
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn3d.geom_point_3d(pn.aes(color=color),shape=marker,size=point_size,show_legend=False)+
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,mapping=pn.aes(color=color),size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,mapping=pn.aes(color=color),size=text_size,va=va,ha=ha)
        elif isinstance(color,np.ndarray):
            # Add gradients colors
            if "point" in geom:
                p = (p + pn3d.geom_point_3d(pn.aes(color="num_var"),shape=marker,size=point_size,show_legend=False)+
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,mapping=pn.aes(color="num_var"),size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
            if "point" in geom:
                p = (p + pn3d.geom_point_3d(pn.aes(color="cluster",linetype = "cluster"),size=point_size,show_legend=False)+
                         pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,mapping=pn.aes(color="cluster"),size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text3d_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom:
                p = p + pn3d.geom_point_3d(color=color,shape=marker,size=point_size,show_legend=False)
            if "text" in geom:
                if repel :
                    p = p + text3d_label(text_type,color=color,size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        if "point" in geom:
            p = p + pn3d.geom_point_3d(pn.aes(color = habillage,linetype = habillage),size=point_size)
        if "text" in geom:
            if repel:
                p = p + text3d_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text3d_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
    
    ##### Add supplementary individuals coordinates
    if ind_sup:
        if hasattr(self, "ind_sup_"):
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom:
                p = p + pn3d.geom_point_3d(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                           color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text3d_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                         color=color_sup,size=text_size,va=va,ha=ha,
                                         adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text3d_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                         color = color_sup,size=text_size,va=va,ha=ha)
    ############## Add supplementary qualitatives coordinates
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            if habillage is None:
                mod_sup_coord = self.quali_sup_["coord"]
                if "point" in geom:
                    p = p + pn3d.geom_point_3d(mod_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                               color=color_quali_sup,size=point_size)
                if "text" in geom:
                    if repel:
                        p = p + text3d_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                             color=color_quali_sup,size=text_size,va=va,ha=ha,
                                             adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,'lw':1.0}})
                    else:
                        p = p + text3d_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                             color =color_quali_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set Z label
    if z_label is None:
        z_label = "Dim."+str(axis[2]+1)+" ("+str(round(proportion[axis[2]],2))+"%)"
    
    # Set title
    if title is None:
        title = "Individuals Factor Map3D - PCA"
    p = p + pn.labs(title=title,x=x_label,y=y_label)+pn3d.zlab(z_label)
    
    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    # Add theme
    p = p + ggtheme
    
    return p