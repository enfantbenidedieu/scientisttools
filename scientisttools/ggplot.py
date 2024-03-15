# -*- coding: utf-8 -*-

import plotnine as pn
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
from scientisttools.extractfactor import get_eigenvalue
from scientisttools.ggcorrplot import ggcorrplot, no_panel
from scientisttools.utils import get_melt
import matplotlib.pyplot as plt
import plydata as ply
from mizani.formatters import percent_format

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

################################################## Visualize eigenvalues ##################################################
def fviz_screeplot(self,
                   choice="proportion",
                   geom_type=["bar","line"],
                   y_lim=None,
                   bar_fill = "steelblue",
                   bar_color="steelblue",
                   line_color="black",
                   line_type="solid",
                   bar_width=None,
                   ncp=10,
                   add_labels=False,
                   ha = "center",
                   va = "bottom",
                   title=None,
                   x_label=None,
                   y_label=None,
                   ggtheme=pn.theme_minimal())-> pn:
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
    barfill : 	fill color for bar plot.
    barcolor : outline color for bar plot.
    linecolor : color for line plot (when geom contains "line").
    linetype : line type
    barwidth : float, the width(s) of the bars
    ncp : a numeric value specifying the number of dimensions to be shown.
    addlabels : logical value. If TRUE, labels are added at the top of bars or points showing the information retained by each dimension.
    ha : horizontal adjustment of the labels.
    va : vertical adjustment of the labels.
    title : title of the graph
    xlabel : x-axis title
    ylabel : y-axis title
    ggtheme : function plotnine theme name. Default value is theme_gray(). Allowed values include plotnine official themes: 
                theme_gray(), theme_bw(), theme_minimal(), theme_classic(), theme_void(), ....
    
    Return
    ------
    figure : a plotnine graphs

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
        
    if self.model_ not in ["pca","ca","mca","famd","partialpca","efa","mfa","hmfa"]:
        raise ValueError("Error : 'self' must be an object of class PCA, CA, MCA, FAMD, PartialPCA, EFA, MFA, HMFA")

    eig = get_eigenvalue(self)
    eig = eig.iloc[:min(ncp,self.call_["n_components"]),:]

    if choice == "eigenvalue":
        eig = eig["eigenvalue"]
        text_labels = list([str(np.around(x,3)) for x in eig.values])
        if y_label is None:
            y_label = "Eigenvalue"
    elif choice == "proportion":
        eig = (1/100)*eig["proportion"]
        text_labels = list([str(np.around(100*x,2))+"%" for x in eig.values])
    else:
        raise ValueError("Error : 'choice' must be one of 'proportion', 'eigenvalue'")

    if isinstance(geom_type,str):
        if geom_type not in ["bar","line"]:
            raise ValueError("Error : The specified value for the argument geomtype are not allowed ")
    elif (isinstance(geom_type,list) or isinstance(geom_type,tuple)):
        intersect = [x for x in geom_type if x in ["bar","line"]]
        if len(intersect)==0:
            raise ValueError("Error : The specified value(s) for the argument geom are not allowed ")
    
    df_eig = pd.DataFrame({"dim" : pd.Categorical(np.arange(1,len(eig)+1)),"eig" : eig.values})
    
    p = pn.ggplot(df_eig,pn.aes(x = "dim",y="eig",group = 1))
    if "bar" in geom_type :
        p = p   +   pn.geom_bar(stat="identity",fill=bar_fill,color=bar_color,width=bar_width)
    if "line" in geom_type :
        p = p  +   pn.geom_line(color=line_color,linetype=line_type) + pn.geom_point(shape="o",color=line_color)
    if add_labels:
        p = p + pn.geom_text(label=text_labels,ha = ha,va = va)
    
    # Scale y continuous
    if choice == "proportion":
        p = p + pn.scale_y_continuous(labels=percent_format())

    if title is None:
        title = "Scree plot"
    if x_label is None:
        x_label = "Dimensions"
    if y_label is None:
        y_label = "Percentage of explained variances"
    
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
    
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

##################################################################################################
#                       Visualize the contributions of row/column elements
###################################################################################################

def fviz_contrib(self,
                 choice="ind",
                 axis=None,
                 y_label=None,
                 top_contrib=None,
                 bar_width=None,
                 add_grid=True,
                 fill_color = "steelblue",
                 color = "steelblue",
                 palette = "Set2",
                 sort_contrib = "desc",
                 xtickslab_rotation = 45,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize the contributions of row/columns elements
    ---------------------------------------------------

    Description
    -----------
    This function can be used to visualize the contribution of rows/columns from the results of Principal Component Analysis (PCA), 
    Correspondence Analysis (CA), Multiple Correspondence Analysis (MCA), Factor Analysis of Mixed Data (FAMD), and Multiple Factor Analysis (MFA) functions.
    

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
        
    if choice not in ["row","col","var","ind","quanti_var","quali_var","group","partial_axes"]:
        raise ValueError("Error : 'choice' should be one of 'row', 'col', 'var', 'ind', 'quanti_var', 'quali_var', 'group' or 'partial_axes'.")

    ncp = self.call_["n_components"]
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise TypeError("Error : 'axis' must be an integer.")
    elif axis not in list(range(0,ncp)):
        raise TypeError(f"Error : 'axis' must be an integer between 0 and {ncp-1}.")
            
    if bar_width is None:
        bar_width = 0.8

    ################## set
    if self.model_ in ["pca","mca","partialpca"] and choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' should be one of 'var', 'ind'.")
    
    if self.model_ == "ca" and choice not in ["row","col"]:
        raise ValueError("Error : 'choice' should be one of 'row', 'col'.")
    
    if self.model_ == "famd" and choice not in ["ind","var","quali_var"]:
        raise ValueError("Error : 'choice' should be one of 'var', 'ind', 'quali_var'")

    if sort_contrib not in ["desc","asc","none"]:
        raise ValueError("Error : 'sort_contrib' should be one of 'desc', 'asc' or 'none'.")

    #### Set names
    if choice == "ind":
        name = "individuals"
    elif choice == "var":
        name = "variables"
    elif choice == "quali_var":
        name = "qualitative variables"
    elif choice == "row":
        name = "rows"
    elif choice == "col":
        name = "columns"
    
    # Extract contribution
    if choice == "ind":
        contrib = self.ind_["contrib"]
    elif choice == "var":
        contrib = self.var_["contrib"]
    elif choice == "quali_var":
        contrib = self.quali_var_["contrib"]
    elif choice == "row":
        contrib = self.row_["contrib"]
    elif choice == "col":
        contrib = self.col_["contrib"]
    
    ####
    contrib = contrib.iloc[:,[axis]].reset_index()
    contrib.columns = ["name","contrib"]

    # Add hline
    hvalue = 100/contrib.shape[0]
    
    #####
    if top_contrib is not None:
            contrib = contrib.sort_values(by="contrib",ascending=False).head(top_contrib)
    
    p = pn.ggplot()
    if choice == "quanti_var" and self.model_ == "mfa":
        pass
    else:
        if sort_contrib == "desc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,-contrib)",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
        elif sort_contrib == "asc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,contrib)",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
        else:
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="contrib",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
    
    if y_label is None:
        y_label = "Contributions (%)"
    title = f"Contribution of {name} to Dim-{axis+1}"
    p = p + pn.labs(title=title,y=y_label,x="")
    
    # if not (choice == "var" and self.model_ =="mca"):
    p = p + pn.geom_hline(yintercept=hvalue,linetype="dashed",color="red")

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    p = p + ggtheme

    if xtickslab_rotation > 5:
        ha = "right"
    if xtickslab_rotation == 90:
        ha = "center"

    # Rotation
    p = p + pn.theme(axis_text_x = pn.element_text(rotation = xtickslab_rotation,ha=ha))

    return p

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

#########################################################################################
# Draw correlation plot
#########################################################################################

def fviz_corrplot(X,
                  method = "square",
                  type = "full",
                  x_label=None,
                  y_label=None,
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
                  digits = 2):
    
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
        X.columns = pd.Categorical(X.columns,categories=X.columns.tolist())
        X.index = pd.Categorical(X.index,categories=X.index.tolist())
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
        if y_label is None:
            y_label = "Dimensions"
        if x_label is None:
            x_label = "Variables"
        p = p + pn.labs(title=title,x=x_label,y=y_label)
            
        # Removing legend
        if not show_legend:
            p =p+pn.theme(legend_position=None)
    
    return p

####################################################################################
#       Principal Components Analysis (PCA)
####################################################################################

# Individuals Factor Map
def fviz_pca_ind(self,
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
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 legend_title=None,
                 add_ellipses=False, 
                 ellipse_type = "t",
                 confint_level = 0.95,
                 geom_ellipse = "polygon",
                 habillage = None,
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
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

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

    if habillage is None :  
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            if "point" in geom_type:
                p = (p + pn.geom_point(pn.aes(color=c),shape=marker,size=point_size,show_legend=False)+ 
                         pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': "black",'lw':1.0}})
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
                                    adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        if "point" in geom_type:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
        if "text" in geom_type:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                    adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipses:
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    ##### Add supplementary individuals coordinates
    if ind_sup:
        if self.ind_sup is not None:
            sup_coord = self.ind_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                        color=color_sup,size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
                else:
                    p = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                        color = color_sup,size=text_size,va=va,ha=ha)
    ############## Add supplementary qualitatives coordinates
    if quali_sup:
        if self.quali_sup is not None:
            if habillage is None:
                mod_sup_coord = self.quali_sup_["coord"]
                if "point" in geom_type:
                    p = p + pn.geom_point(mod_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                        color=color_quali_sup,size=point_size)
                if "text" in geom_type:
                    if repel:
                        p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                           color=color_quali_sup,size=text_size,va=va,ha=ha,
                                           adjust_text={'arrowprops': {'arrowstyle': '-','color': color_quali_sup,'lw':1.0}})
                    else:
                        p = p + text_label(text_type,data=mod_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=mod_sup_coord.index),
                                           color =color_quali_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    # Set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    # Set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    # Set title
    if title is None:
        title = "Individuals factor map - PCA"
    p = p + pn.labs(title=title,x=x_label,y = y_label)
    
    # Set x limits
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p + pn.ylim(y_lim)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0,colour=hline_color,linetype =hline_style)    
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0,colour=vline_color,linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    # Add theme
    p = p + ggtheme
    
    return p

# Variables Factor Map
def fviz_pca_var(self,
                 axis=[0,1],
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom_type = ["arrow","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
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
                 color_circle="gray",
                 arrow_angle=10,
                 arrow_length =0.1,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    
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
    
    # Add supplmentary continuous variables
    if quanti_sup:
        if self.quanti_sup is not None:
            sup_coord = self.quanti_sup_["coord"]
            if "arrow" in geom_type:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                    arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype=linestyle_sup)
            if "text" in geom_type:
                p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),color=color_sup,size=text_size,va=va,ha=ha)
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
        title = "Variables factor map - PCA"
    
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

# https://stackoverflow.com/questions/6578355/plotting-pca-biplot-with-ggplot2
# https://github.com/vqv/ggbiplot/blob/master/R/ggbiplot.r

def fviz_pca_biplot(self,axis=[0,1],circle_prob=0.69,scale=1) ->pn :

    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an instance of class PCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Number of observations
    nobs_factor = np.sqrt(self.call_["X"].shape[0])
    obs_scale = 1 - scale
    var_scale = scale
    
    # Extract individuals coordinates
    ind_coord = self.res_["ind"]["coord"].iloc[:,axis]
    ind_coord = ind_coord*nobs_factor

    # Extract variables coordinates
    var_coord = self.res_["var"]["coord"].iloc[:,axis]

    # Scale the radius of the correlation circle so that it corresponds to 
    # a data ellipse for the standardized PC scores
    r = np.sqrt(st.chi2.ppf(circle_prob, df = 2)) * (ind_coord**2).mean(axis=0).prod()**(1/4)

    # Scale direction


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

########################################################
def fviz_corrcircle(self,
                    axis=[0,1],
                    x_label=None,
                    y_label=None,
                    title=None,
                    geom_type = ["arrow","text"],
                    color = "black",
                    color_sup = "blue",
                    text_type = "text",
                    arrow_length=0.1,
                    text_size=8,
                    arrow_angle=10,
                    add_circle=True,
                    color_circle = "gray",
                    add_hline=True,
                    add_vline=True,
                    add_grid=True,
                    ggtheme=pn.theme_minimal()) -> pn:
    """
    Draw correlation circle
    -----------------------


    Description
    -----------


    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if self.model_ not in ["pca","ca","mca","famd","mfa"]:
        raise ValueError("Error : Factor method not allowed.")
    
    if self.model_ == "pca":
        coord = self.var_["coord"]
    elif self.model_ == "famd":
        coord = self.quanti_var_["coord"]
    else:
        if self.quanti_sup is not None:
            coord = self.quanti_sup_["coord"]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "arrow" in geom_type:
        p = p + pn.geom_segment(pn.aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"), 
                                arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color)
    if "text" in geom_type:
            p = p + text_label(text_type,color=color,size=text_size,va="center",ha="center")
        
    if self.model_ in ["pca","famd"]:
        if self.quanti_sup is not None:
            sup_coord = self.quanti_sup_["coord"]
            if "arrow" in geom_type:
                p  = p + pn.annotate("segment",x=0,y=0,xend=np.asarray(sup_coord.iloc[:,axis[0]]),yend=np.asarray(sup_coord.iloc[:,axis[1]]),
                                     arrow = pn.arrow(length=arrow_length,angle=arrow_angle),color=color_sup,linetype="--")
            if "text" in geom_type:
                p  = p + text_label(text_type,data=sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                    color=color_sup,size=text_size,va="center",ha="center")
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
        title = "Correlation circle"
    
    p = p + pn.xlim((-1,1))+ pn.ylim((-1,1))+ pn.labs(title=title,x=x_label,y=y_label)

    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour="black", linetype ="dashed")
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour="black", linetype ="dashed")
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

#################################################################################################################
#               Correspondence Analysis (CA) graphs
#################################################################################################################

# Row points Factor Map
def fviz_ca_row(self,
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
        raise ValueError("Error : You must pass a valid 'axis'.")

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
            raise ValueError("Error : 'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.row_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("Error : 'lim_contrib' must be a float or an integer.")
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

    if habillage is None:
        # Using cosine and contributions
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            if "point" in geom_type:
                p = p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)
                p = p + pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title)
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha)
        elif hasattr(color, "labels_"):
                c = [str(x+1) for x in color.labels_]
                if legend_title is None:
                    legend_title = "Cluster"
                #####################################
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
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if self.quali_sup is not None:
            if habillage not in coord.columns.tolist():
                raise ValueError(f"Error : {habillage} not in DataFrame")
            if "point" in geom_type:
                p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
            if add_ellipses:
                p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
        
    if row_sup:
        if self.row_sup is not None:
            row_sup_coord = self.row_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(row_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)
    
    if quali_sup:
        if self.quali_sup is not None:
            quali_sup_coord = self.quali_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(quali_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_coord.index),
                                      color = color_quali_sup,shape = marker_quali_sup,size=point_size)
            if "text" in geom_type:
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

# Columns points factor map
def fviz_ca_col(self,
                 axis=[0,1],
                 x_lim= None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom_type = ["point","text"],
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
        raise ValueError("Error : You must pass a valid 'axis'.")

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
            raise ValueError("Error : 'lim_cos2' must be a float or an integer.")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.col_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise ValueError("Error : 'lim_contrib' must be a float or an integer.")

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
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                     pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))
        if "text" in geom_type:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(colour=c),size=text_size,va=va,ha=ha)
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
            p = p + pn.geom_point(pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),color=color,shape=marker,size=point_size)
        if "text" in geom_type:
            if repel:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
            else:
                p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    ###################### Add supplementary columns coordinates
    if col_sup:
        if self.col_sup is not None:
            sup_coord = self.col_sup_["coord"]
            if "point" in geom_type:
                p  = p + pn.geom_point(sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index),
                                       color=color_sup,shape=marker_sup,size=point_size)
            if "text" in geom_type:
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
                   row_geom_type = ["point","text"],
                   col_geom_type = ["point","text"],
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
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    ###### Initialize coordinates
    row_coord = self.row_["coord"]
    col_coord = self.col_["coord"]

    ###############" Initialize
    p = pn.ggplot()
    ########### Add rows coordinates
    if "point" in row_geom_type:
        p = p + pn.geom_point(data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label = row_coord.index),
                              color=row_color,shape=row_marker,size=row_point_size)
    if "text" in row_geom_type:
        if row_repel:
            p = p + text_label(row_text_type,data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_coord.index),
                               color=row_color,size=row_text_size,va=row_va,ha=row_ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': row_color,"lw":1.0}})
        else:
            p = p + text_label(row_text_type,data=row_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_coord.index),
                               color=row_color,size=row_text_size,va=row_va,ha=row_ha)
    
    ############ Add columns coordinates
    if "point" in col_geom_type:
        p = p + pn.geom_point(data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label = col_coord.index),
                              color=col_color,shape=col_marker,size=col_point_size)
    if "text" in col_geom_type:
        if col_repel:
            p = p + text_label(col_text_type,data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_coord.index),
                               color=col_color,size=col_text_size,va=col_va,ha=col_ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','color': col_color,"lw":1.0}})
        else:
            p = p + text_label(col_text_type,data=col_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_coord.index),
                               color=col_color,size=col_text_size,va=col_va,ha=col_ha)
    
    ################################ Add supplementary elements
    if row_sup:
        if self.row_sup is not None:
            row_sup_coord = self.row_sup_["coord"]
            if "point" in row_geom_type:
                p = p + pn.geom_point(row_sup_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                      color = row_color_sup,shape = row_marker_sup,size=row_point_size)
            if "text" in row_geom_type:
                if row_repel:
                    p = p + text_label(row_text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color=row_color_sup,size=row_text_size,va=row_va,ha=row_ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': row_color_sup,"lw":1.0}})
                else:
                    p = p + text_label(row_text_type,data=row_sup_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=row_sup_coord.index),
                                       color = row_color_sup,size=row_text_size,va=row_va,ha=row_ha)
    
    if col_sup:
        if self.col_sup is not None:
            col_sup_coord = self.col_sup_["coord"]
            if "point" in col_geom_type:
                p  = p + pn.geom_point(col_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=col_sup_coord.index),
                                       color=col_color_sup,shape=col_marker_sup,size=col_point_size)
            if "text" in col_geom_type:
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
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if choice not in ["row","col","biplot"]:
        raise ValueError("Error : Allowed values for choice are :'row', 'col' or 'biplot'.")


    if choice == "row":
        return fviz_ca_row(self,**kwargs)
    elif choice == "col":
        return fviz_ca_col(self,**kwargs)
    elif choice == "biplot":
        return fviz_ca_biplot(self,**kwargs)

######################################################################################################
##                             Multiple Correspondence Analysis (MCA)
######################################################################################################

def fviz_mca_ind(self,
                 choice = "var",
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 title =None,
                 x_label = None,
                 y_label = None,
                 color ="black",
                 geom_type = ["point","text"],
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 marker = "o",
                 legend_title=None,
                 add_grid =True,
                 ind_sup=True,
                 color_sup = "blue",
                 marker_sup = "^",
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
                 repel=False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    
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
        if (isinstance(color,str) and color in [*["cos2","contrib"],*coord.columns.tolist()]) or (isinstance(color,np.ndarray)):
            # Add gradients colors
            if "point" in geom_type:
                p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+
                         pn.scale_color_gradient2(low = gradient_cols[0],
                                                  high = gradient_cols[2],
                                                  mid = gradient_cols[1],
                                                  name = legend_title))
            if "text" in geom_type:
                if repel :
                    p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
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
                                       adjust_text={'arrowprops': {'arrowstyle': '-','color': color,'lw':1.0}})
                else:
                    p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"Error : {habillage} not in DataFrame.")
        if "point" in geom_type:
            p = p + pn.geom_point(pn.aes(color = habillage,linetype = habillage),size=point_size,shape=marker)
        if "text" in geom_type:
            if repel:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
            else:
                p = p + text_label(text_type,mapping=pn.aes(color=habillage),size=text_size,va=va,ha=ha)
        
        if add_ellipses:
            p = p + pn.geom_point(pn.aes(color = habillage))
            p = p + pn.stat_ellipse(geom=geom_ellipse,mapping=pn.aes(fill=habillage),type = ellipse_type,alpha = 0.25,level=confint_level)
    
    if ind_sup:
        if self.ind_sup is not None:
            sup_coord = self.ind_sup_["coord"]
            p = p + pn.geom_point(data=sup_coord,
                                  mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                  color = color_sup,shape = marker_sup,size=point_size)
            if repel:
                p = p + text_label(text_type,data=sup_coord,
                                   mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                   color=color_sup,size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,'lw':1.0}})
            else:
                p = p + text_label(text_type,data=sup_coord,
                                   mapping=pn.aes(x =f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=sup_coord.index.tolist()),
                                     color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Individuals factor map - MCA"
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

# Graph for categories
def fviz_mca_mod(self,
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
                 corrected = False,
                 repel=False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Draw the Multiple Correspondence Analysis (MCA) categorical graphs
    ------------------------------------------------------------------

    Description
    -----------


    Parameters
    ----------
    self : an object of class MCA


    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    # Categories labels
    
    # Corrected 
    if corrected:
        coord = self.var_["corrected_coord"]
    else:
        coord = self.var_["coord"]

    # Using lim cos2
    if lim_cos2 is not None:
        if (isinstance(lim_cos2,float) or isinstance(lim_cos2,int)):
            lim_cos2 = float(lim_cos2)
            cos2 = self.var_["cos2"].iloc[:,axis].sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("Error : 'lim_cos2' must be a float or an integer")
    
    # Using lim contrib
    if lim_contrib is not None:
        if (isinstance(lim_contrib,float) or isinstance(lim_contrib,int)):
            lim_contrib = float(lim_contrib)
            contrib = self.var_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("Error : 'lim_cos2' must be a float or an integer")

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

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
        c= np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
     
    # Using cosine and contributions
    if (isinstance(color,str) and color in ["cos2","contrib"]) or isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1], name = legend_title))
        if "text" in geom_type:
            if repel :
                p = p + text_label(text_type,mapping=pn.aes(color=c),size=text_size,va=va,ha=ha,
                                   adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
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
            var_sup_coord = self.quali_sup_["coord"]
            if "point" in geom_type:
                p = p + pn.geom_point(var_sup_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index.tolist()),
                                      color=color_sup,size=point_size,shape=marker_sup)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=var_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index.tolist()),
                                       color=color_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
                else:
                    p = p + text_label(text_type,data=var_sup_coord,mapping=pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=var_sup_coord.index),
                                       color=color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Qualitatives variables categories - MCA"
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

#------------------------------------------------------------------------
def fviz_mca_var(self,
                 axis=[0,1],
                 x_lim=None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title=None,
                 color="black",
                 color_sup = "blue",
                 color_quanti_sup = "red",
                 geom_type = ["point","text"],
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
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")

    # Initialize
    coord = self.var_["eta2"]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index.tolist()))

    if isinstance(color,np.ndarray):
        c = np.asarray(color)
        if legend_title is None:
            legend_title = "Cont_Var"
    
    # Using cosine and contributions
    if isinstance(color,np.ndarray):
        # Add gradients colors
        if "point" in geom_type:
            p = (p + pn.geom_point(pn.aes(colour=c),shape=marker,size=point_size,show_legend=False)+ 
                    pn.scale_color_gradient2(low=gradient_cols[0],high=gradient_cols[2],mid=gradient_cols[1],name=legend_title))
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
    
    if add_quanti_sup:
        if self.quanti_sup is not None:
            quant_sup_cos2 = self.quanti_sup_["cos2"]
            if "point" in geom_type:
                p = p + pn.geom_point(quant_sup_cos2,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quant_sup_cos2.index),
                                      color = color_quanti_sup,shape = marker_quanti_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=quant_sup_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quant_sup_cos2.index),
                                       color=color_quanti_sup,size=text_size,va=va,ha=ha,
                                       adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quant_sup_cos2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quant_sup_cos2.index),
                                       color = color_quanti_sup,size=text_size,va=va,ha=ha)
    
    if add_quali_sup:
        if self.quali_sup is not None:
            quali_sup_eta2 = self.quali_sup_["eta2"]
            if "point" in geom_type:
                p = p + pn.geom_point(quali_sup_eta2,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                      color = color_sup,shape = marker_sup,size=point_size)
            if "text" in geom_type:
                if repel:
                    p = p + text_label(text_type,data=quali_sup_eta2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-',"lw":1.0}})
                else:
                    p = p + text_label(text_type,data=quali_sup_eta2,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=quali_sup_eta2.index),
                                       color = color_sup,size=text_size,va=va,ha=ha)

    # Add additionnal        
    proportion = self.eig_.iloc[:,2].values
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"

    if title is None:
        title = "Graphe of variables - MCA"
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
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
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
                                        color=color_sup,size=text_size,va=va,ha=ha,
                                        adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
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

    ##### Add actives variables
    Xtot = self.call_["Xtot"]
    Xtot.columns = Xtot.columns.droplevel()

    # Concatenate
    coord = pd.concat((coord,Xtot),axis=1)
    print(coord.columns)

    #### Drop supplementary rows
    if self.ind_sup is not None:
        coord = coord.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])

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
            contrib = self.ind_["contrib"].iloc[:,axis].sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query(f"contrib > {lim_contrib}")
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
            print(c)
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