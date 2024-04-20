# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

from .text_label import text_label

def fviz_ca_row(self,
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
                 row_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 quali_sup = True,
                 color_quali_sup = "red",
                 marker_quali_sup = ">",
                 add_hline = True,
                 add_vline=True,
                 legend_title = None,
                 habillage=None,
                 add_ellipses=False, 
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
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Correspondence Analysis - Graph of row variables
    ----------------------------------------------------------

    Description
    -----------
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables.

    Parameters
    ----------
    self : an object of class CA

    axis : a numeric list or vector of length 2 specifying the dimensions to be plotted, default = [0,1]

    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    # Initialize
    coord = self.row_["coord"]

    ################################### Add active data
    coord = pd.concat([coord,self.call_["X"]],axis=1)

    ################ Add supplementary columns
    if self.col_sup is not None:
        X_col_sup = self.call_["Xtot"].loc[:,self.col_sup_["coord"].index.tolist()].astype("float")
        if self.row_sup is not None:
            X_col_sup = X_col_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.row_sup_["coord"].index.tolist()])
        coord = pd.concat([coord,X_col_sup],axis=1)

    ################ Add supplementary quantitatives columns
    if self.quanti_sup is not None:
        X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
        if self.row_sup is not None:
            X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.row_sup_["coord"].index.tolist()])
        coord = pd.concat([coord,X_quanti_sup],axis=1)
    
    ################ Add supplementary qualitatives columns
    if self.quali_sup is not None:
        X_quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
        if self.row_sup is not None:
            X_quali_sup = X_quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.row_sup_["coord"].index.tolist()])
        coord = pd.concat([coord,X_quali_sup],axis=1)
    
    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.row_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.row_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")
    
    # Set color if cos2, contrib or continuous variables
    if isinstance(color,str):
        if color == "cos2":
            c = self.row_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Cos2"
        elif color == "contrib":
            c = self.row_["contrib"].iloc[:,axis].sum(axis=1).values
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
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if habillage is None:
        # Using cosine and contributions
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            if "point" in geom:
                p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
                p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title)
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
                c = [str(x+1) for x in color.labels_]
                if legend_title is None:
                    legend_title = "Cluster"
                #####################################
                if "point" in geom:
                    p = (p + pn.geom_point(pn.aes(color=c),size=point_size,show_legend=False)+
                            pn.guides(color=pn.guide_legend(title=legend_title)))
                if "text" in geom:
                    if repel :
                        p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                    else:
                        p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        else:
            if "point" in geom:
                p = p + pn.geom_point(color=color,shape=marker,size=point_size,show_legend=False)
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup is not None:
            if habillage not in coord.columns.tolist():
                raise ValueError(f"Error : {habillage} not in DataFrame")
            if "point" in geom:
                p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            if add_ellipses:
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
        
    if row_sup:
        if hasattr(self, "row_sup_"):
            row_sup_coord = self.row_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(row_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)
    
    if quali_sup:
        if hasattr(self, "quali_sup_"):
            quali_sup_coord = self.quali_sup_["coord"]
            if "point" in geom:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                      color = color_quali_sup,shape = marker_quali_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                       color=color_quali_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                       color = color_quali_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2]
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Row points - CA"
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

def fviz_ca_col(self,
                 axis=[0,1],
                 x_lim= None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 text_type = "text",
                 marker = "o",
                 point_size = 1.5,
                 text_size = 8,
                 add_grid =True,
                 legend_title = None,
                 col_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_hline = True,
                 add_vline = True,
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
    Visualize Correspondence Analysis - Graph of column variables
    -------------------------------------------------------------

    Description
    -----------
    Correspondence analysis (CA) is an extension of Principal Component Analysis (PCA) suited to analyze frequencies formed by two categorical variables.

    Parameters
    ----------
    self : an object of class CA

    axis : a numeric list or vector of length 2 specifying the dimensions to be plotted, default = [0,1]

    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    ###### Initialize coordinates
    coord = self.col_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.col_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.col_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer.")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if isinstance(color,str):
        if color == "cos2":
            c = self.col_["cos2"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                 legend_title = "Cos2"
        elif color == "contrib":
            c = self.col_["contrib"].iloc[:,axis].sum(axis=1).values
            if legend_title is None:
                legend_title = "Contrib"
    elif isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    if (isinstance(color,str) and color in ["cos2","contrib"]) or (isinstance(color,np.ndarray)):
        # Add gradients colors
        if "point" in geom:
            p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
    elif hasattr(color, "labels_"):
            c = [str(x+1) for x in color.labels_]
            if legend_title is None:
                legend_title = "Cluster"
            if "point" in geom:
                p = (p + pn.geom_point(pn.aes(color=c),size=point_size,show_legend=False)+
                        pn.guides(color=pn.guide_legend(title=legend_title)))
            if "text" in geom:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
    else:
        if "point" in geom:
            p = p + pn.geom_point(pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),color=color,shape=marker,size=point_size)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    ###################### Add supplementary columns coordinates
    if col_sup:
        if hasattr(self, "col_sup_"):
            sup_coord = self.col_sup_["coord"]
            if "point" in geom:
                p  = p + pn.geom_point(sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,shape=marker_sup,size=point_size)
            if "text" in geom:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                        color=color_sup,size=text_size,va=va,ha=ha)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2]
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Columns points - CA"
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
    
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=x_label)+pn.ylab(ylab=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_ca_biplot(self,
                   axis=[0,1],
                   x_lim = None,
                   y_lim = None,
                   x_label = None,
                   y_label = None,
                   title = None,
                   row_geom = ["point","text"],
                   col_geom = ["point","text"],
                   row_color = "black",
                   col_color = "blue",
                   row_point_size = 1.5,
                   col_point_size = 1.5,
                   row_text_size = 8,
                   col_text_size = 8,
                   row_text_type = "text",
                   col_text_type = "text",
                   row_marker = "o",
                   col_marker = "^",
                   add_grid = True,
                   row_sup = True,
                   col_sup = True,
                   row_color_sup = "red",
                   col_color_sup = "darkblue",
                   row_marker_sup = ">",
                   col_marker_sup = "v",
                   add_hline = True,
                   add_vline = True,
                   row_ha = "center",
                   row_va = "center",
                   col_ha = "center",
                   col_va = "center",
                   hline_color = "black",
                   hline_style = "dashed",
                   vline_color = "black",
                   vline_style = "dashed",
                   row_repel = True,
                   col_repel = True,
                   ggtheme = pn.theme_minimal()) ->pn:
    """
    Visualize Correspondence Analysis - Biplot of row and columns variables
    -----------------------------------------------------------------------

    Parameters
    ----------
    self : an object of class CA

    axis : a numeric list or vector of length 2 specifying the dimensions to be plotted, default = [0,1]
    
    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "ca":
        raise ValueError("'self' must be an object of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    ###### Initialize coordinates
    row_coord = self.row_["coord"]
    col_coord = self.col_["coord"]

    ###############" Initialize
    p = pn.ggplot()
    ########### Add rows coordinates
    if "point" in row_geom:
        p = p + pn.geom_point(data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label = row_coord.index),
                              color=row_color,shape=row_marker,size=row_point_size)
    if "text" in row_geom:
        if row_repel:
            p = p + text_label(row_text_type,data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_coord.index),
                               color=row_color,size=row_text_size,va=row_va,ha=row_ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': row_color,"lw":1.0}})
        else:
            p = p + text_label(row_text_type,data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_coord.index),
                               color=row_color,size=row_text_size,va=row_va,ha=row_ha)
    
    ############ Add columns coordinates
    if "point" in col_geom:
        p = p + pn.geom_point(data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label = col_coord.index),
                              color=col_color,shape=col_marker,size=col_point_size)
    if "text" in col_geom:
        if col_repel:
            p = p + text_label(col_text_type,data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_coord.index),
                               color=col_color,size=col_text_size,va=col_va,ha=col_ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': col_color,"lw":1.0}})
        else:
            p = p + text_label(col_text_type,data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_coord.index),
                               color=col_color,size=col_text_size,va=col_va,ha=col_ha)
    
    ################################ Add supplementary elements
    if row_sup:
        if hasattr(self, "row_sup_"):
            row_sup_coord = self.row_sup_["coord"]
            if "point" in row_geom:
                p = p + pn.geom_point(row_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                      color = row_color_sup,shape = row_marker_sup,size=row_point_size)
            if "text" in row_geom:
                if row_repel:
                    p = p + text_label(row_text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color=row_color_sup,size=row_text_size,va=row_va,ha=row_ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': row_color_sup,"lw":1.0}})
                else:
                    p = p + text_label(row_text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color = row_color_sup,size=row_text_size,va=row_va,ha=row_ha)
    
    if col_sup:
        if hasattr(self, "col_sup_"):
            col_sup_coord = self.col_sup_["coord"]
            if "point" in col_geom:
                p  = p + pn.geom_point(col_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_coord.index),
                                       color=col_color_sup,shape=col_marker_sup,size=col_point_size)
            if "text" in col_geom:
                if col_repel:
                    p = p + text_label(col_text_type,data=col_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_coord.index),
                                       color=col_color_sup,size=col_text_size,va=col_va,ha=col_ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': col_color_sup,"lw":1.0}})
                else:
                    p  = p + text_label(col_text_type,data=col_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_coord.index),
                                        color=col_color_sup,size=col_text_size,va=col_va,ha=col_ha)
    
    # Add additionnal        
    proportion = self.eig_.iloc[:,2]
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "CA - Biplot"
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
    
    p = p + pn.ggtitle(title)+ pn.xlab(xlab=x_label)+pn.ylab(ylab=y_label)

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
                - 'biplot' for biplot and row and columns factor map

    **kwargs : 	further arguments passed to or from other methods

    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    if choice not in ["row","col","biplot"]:
        raise ValueError("choice should be one of 'row', 'col', 'biplot'")

    if choice == "row":
        return fviz_ca_row(self,**kwargs)
    elif choice == "col":
        return fviz_ca_col(self,**kwargs)
    elif choice == "biplot":
        return fviz_ca_biplot(self,**kwargs)