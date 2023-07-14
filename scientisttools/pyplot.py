# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text
import scipy.stats as st
from matplotlib.patches import Ellipse
import math
import random
from scipy.cluster.hierarchy import dendrogram,fcluster
from matplotlib.patches import Rectangle

###################################################################
#           CORRESPONDENCE ANALYSIS (CA)
####################################################################

def plotCA(self,
           choice ="row",
           axis=(0,1),
           xlim=None,
           ylim=None,
           title=None,
           color="blue",
           marker="o",
           add_grid =True,
           add_sup=False,
           color_sup = "red",
           marker_sup ="^",
           color_map ="jet",
           add_hline = True,
           add_vline=True,
           arrow = False,
           ha="center",
           va="center",
           hline_color="black",
           hline_style="dashed",
           vline_color="black",
           vline_style ="dashed",
           repel = False,
           ax=None)->plt:
    
    """ Plot te Factor map for rows and columns

    Parameters
    ----------
    self : aninstance of class CA
    choice : str 
    axis : tuple or list of two elements
    xlim : tuple or list of two elements
    ylim : tuple of list of two elements
    title : str
    color : str
    marker : str
             The marker style for active points
    add_grid : bool
    add_sup : bool
    color_sup : str : 
                The markers colors
    marker_sup : str
                 The marker style for supplementary points
    color_map : str
    add_hline : bool
    add_vline : bool
    ha : horizontalalignment : {'left','center','right'}
    va : verticalalignment {"bottom","baseline","center","center_baseline","top"}
    hline_color :
    hline_style :
    vline_color :
    vline_style :
    ax :

    Returns
    -------
    None
    """
    
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an instance of class CA.")
    
    if choice not in ["row","col"]:
        raise ValueError("Error : 'choice' ")
    
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_-1)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    
    if choice == "row":
        coord = self.row_coord_[:,axis]
        cos2 = self.row_cos2_[:,axis]
        contrib = self.row_contrib_[:,axis]
        labels = self.row_labels_
        if title is None:
            title = "Row points - CA"
        if add_sup:
            if self.row_sup_labels_ is not None:
                sup_labels = self.row_sup_labels_
                sup_coord = self.row_sup_coord_[:,axis]
    else:
        coord = self.col_coord_[:,axis]
        cos2 = self.col_cos2_[:,axis]
        contrib = self.col_contrib_[:,axis]
        labels = self.col_labels_
        if title is None:
            title = "Columns points - CA"
        if add_sup:
            if self.col_sup_labels_ is not None:
                sup_labels = self.col_sup_labels_
                sup_coord = self.col_sup_coord_[:,axis]
        

    # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]
        
    if color == "cos2":
        c = np.sum(cos2,axis=1)
    elif color == "contrib":
        c = np.sum(contrib,axis=1)
    
    if color in ["cos2","contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))
        p = ax.scatter(xs,ys,c=c,s=len(c),marker=marker,cmap=plt.get_cmap(color_map))
        plt.colorbar(p).ax.set_title(label=color,weight='bold')
        # Add labels
        if repel:
            texts = list()
            for i, lab in enumerate(labels):
                colorVal = scalarMap.to_rgba(c[i])
                texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                if arrow:
                    ax.arrow(0,0,xs[i],ys[i],length_includes_head=True,color=colorVal)
            adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
        else:
            for i, lab in enumerate(labels):
                colorVal = scalarMap.to_rgba(c[i])
                ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)
                if arrow:
                    ax.arrow(0,0,xs[i],ys[i],length_includes_head=True,color=colorVal)
    else:
        ax.scatter(xs,ys,c=color,marker=marker)
        # Add labels
        if repel:
            texts = list()
            for i, lab in enumerate(labels):
                texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                if arrow:
                    ax.arrow(0,0,xs[i],ys[i],length_includes_head=True,color=color)
            adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
        else:
            for i, lab in enumerate(labels):
                ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
                if arrow:
                    ax.arrow(0,0,xs[i],ys[i],length_includes_head=True,color=color)
    if add_sup:
        xxs = sup_coord
        # Reset xlim and ylim
        xxs = sup_coord[:,axis[0]]
        yys = sup_coord[:,axis[1]]
        # Add supplementary row coordinates
        ax.scatter(xxs,yys,c=color_sup,marker=marker_sup)
        if repel:
            texts = list()
            for i,lab in enumerate(sup_labels):
                texts.append(ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup))
            adjust_text(texts,x=xxs,y=yys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
        else:
            for i,lab in enumerate(sup_labels):
                ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup)
       
    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    # Add horizontal and vertical lines
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)


#####################################################################################
#
####################################################################################
# -*- coding: utf-8 -*-

def plotCMDS(self,
            axis=[0,1],
            xlim=(None,None),
            ylim=(None,None),
            title =None,
            color="blue",
            marker="o",
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
            ax=None) -> plt:
    
    if self.model_ != "cmds":
        raise ValueError("Error : 'self' must be an instance of class CMDSCALE.")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_-1)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid 'axis'.")
        
    if ax is None:
        ax = plt.gca()
    
    xs = self.coord_[:,axis[0]]
    ys = self.coord_[:,axis[1]]
    ax.scatter(xs,ys,color=color,marker=marker)
    if repel:
        texts =list()
        for i,lab in enumerate(self.labels_):
            texts.append(ax.text(xs[i],ys[i],lab,color=color,ha=ha,va=va))
        adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
    else:
        for i,lab in enumerate(self.labels_):
            ax.text(xs[i],ys[i],lab,color=color,ha=ha,va=va)

    if title is None:
        title = "Classical multidimensional scaling (PCoA, Principal Coordinates Analysis)"

    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)  

#####################################################################################################
#           PLOT CONTRIBUTIONS
#####################################################################################################

def plot_contrib(self,
                 choice="ind",
                 axis=None,
                 xlabel=None,
                 top_contrib=10,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 short_labels=False,
                 ax=None) -> plt:
    
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
        
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active Axes.
        
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
            
    if ax is None:
        ax = plt.gca()
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
    r = np.arange(n_labels)
    ax.barh(r,contrib_sorted,height=bar_width,color=color,align="edge")
    ax.set_yticks([x + bar_width/2 for x in r], labels_sort)
    ax.set(title=f"Contribution of {name} to Dim-{axis+1}",xlabel=xlabel,ylabel=name)
    ax.grid(visible=add_grid)


################################################################################################
#           PLOT CORRELATION CIRCLE
################################################################################################

def plot_correlation_circle(self,
                            axis=[0,1],
                            title =None,
                            color="blue",
                            add_grid =True,
                            color_map ="jet",
                            add_hline = True,
                            add_vline=True,
                            ha="center",
                            va="center",
                            add_circle=True,
                            quanti_sup=True,
                            color_sup = "red",
                            hline_color="black",
                            hline_style="dashed",
                            vline_color="black",
                            vline_style ="dashed",
                            patch_color = "black",
                            repel=False,
                            ax=None) -> plt:

    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_-1)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid 'axis'.")
        
    xs = self.col_coord_[:,axis[0]]
    ys = self.col_coord_[:,axis[1]]

    if ax is None:
        ax = plt.gca()
    if color == "cos2":
        c = np.sum(self.col_cos2_,axis=1)
    elif color == "contrib":
        c = np.sum(self.col_contrib_,axis=1)
    
    if color in ["cos2","contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))

    if color in ["cos2","contrib"]:
        if repel:
            texts = list()
            for j, lab in enumerate(self.col_labels_):
                colorVal = scalarMap.to_rgba(c[j])
                ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal))
                #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
            adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
        else:
            for j, lab in enumerate(self.col_labels_):
                colorVal = scalarMap.to_rgba(c[j])
                ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal)
    else:
        if repel:
            texts = list()
            for j, lab in enumerate(self.col_labels_):
                ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color))
            adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
        else:
            for j, lab in enumerate(self.col_labels_):
                ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color)  
        
    if quanti_sup:
        if self.quanti_sup_labels_ is not None:
            xxs = self.col_sup_coord_[:,axis[0]]
            yys = self.col_sup_coord_[:,axis[1]]
            # Add labels
            if repel:
                texts = list()
                for j, lab in enumerate(self.quanti_sup_labels_):
                    ax.arrow(0,0,xxs[j],yys[j],head_width=0.02,length_includes_head=True,color=color_sup)
                    texts.append(ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color=color_sup))
                adjust_text(texts,x=xxs,y=yys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(self.quanti_sup_labels_):
                    ax.arrow(0,0,xxs[j],yys[j],head_width=0.02,length_includes_head=True,color=color_sup)
                    ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color=color_sup)
    if add_circle:
        ax.add_patch(plt.Circle((0,0),1, color=patch_color,fill=False))
            
    if title is None :
        title = "Correlation circle"

    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=(-1.1,1.1),ylim=(-1.1,1.1))
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   
    
    
######################################################################################
#               PLOT COSINES
#####################################################################################

def plot_cosines(self,
                 choice="ind",
                 axis=None,
                 xlabel=None,
                 top_cos2=10,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 short_labels=False,
                 ax=None) -> plt:
    
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
        
        ax : matplotlib Axes, optional
            Axes in which to draw the plot, otherwise use the currently-active Axes.
        
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

    if ax is None:
        ax = plt.gca()
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
    r = np.arange(n_labels)
    ax.barh(r,cos2_sorted,height=bar_width,color=color,align="edge")
    ax.set_yticks([x + bar_width/2 for x in r], labels_sort)
    ax.set(title=f"Cosinus of {name} to Dim-{axis+1}",xlabel=xlabel,ylabel=name,xlim=(0,1))
    ax.grid(visible=add_grid)

    
##############################################################################################
#                   EXPLORATORY FACTOR ANALYSIS
###############################################################################################

def plotEFA(self,
            choice ="ind",
            axis=[0,1],
            xlim=(None,None),
            ylim=(None,None),
            title =None,
            color="blue",
            marker="o",
            add_grid =True,
            color_map ="jet",
            add_hline = True,
            add_vline=True,
            ha="center",
            va="center",
            add_circle=True,
            hline_color="black",
            hline_style="dashed",
            vline_color="black",
            vline_style ="dashed",
            patch_color = "black",
            repel=False,
            ax=None) -> plt:
    
    """ Plot the Factor map for individuals and variables

    Parameters
    ----------
    self : aninstance of class PCA
    choice : str 
    axis : tuple or list of two elements
    xlim : tuple or list of two elements
    ylim : tuple or list of two elements
    title : str
    color : str
    marker : str
             The marker style for active points
    add_grid : bool
    color_map : str
    add_hline : bool
    add_vline : bool
    ha : horizontalalignment : {'left','center','right'}
    va : verticalalignment {"bottom","baseline","center","center_baseline","top"}
    hline_color :
    hline_style :
    vline_color :
    vline_style :
    ax :
    **kwargs : Collection properties

    Returns
    -------
    None
    """

    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an instance of class EFA.")
    
    if choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' ")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    
    if choice == "ind":
        coord = self.row_coord_[:,axis]
        labels = self.row_labels_
        if title is None:
            title = "Individuals factor map - EFA"
    else:
        coord = self.col_coord_[:,axis]
        contrib = self.col_contrib_[:,axis]
        labels = self.col_labels_
        if title is None:
            title = "Variables factor map - EFA"
            
    # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    if color == "contrib":
        c = np.sum(contrib,axis=1)
    
    if color in ["contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))
    
    if choice == "ind":
        if color in ["contrib"]:
            raise NotImplementedError("Error : This method is not implemented yet.")
        else:
            ax.scatter(xs,ys,c=color,marker=marker)
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
    else:
        if color == "contrib":
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal)
        else:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color)  
        
        if add_circle:
             ax.add_patch(plt.Circle((0,0),1,color=patch_color,fill=False))
    
    if choice == "var":
        xlim = ylim = (-1.1,1.1)

    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)  


###############################################################################################
#               Plot Eigenvalues
###############################################################################################

def plot_eigenvalues(self,
                     choice ="proportion",
                     n_components=10,
                     title=None,
                     xlabel=None,
                     ylabel=None,
                     bar_fill="steelblue",
                     bar_color = "steelblue",
                     line_color="black",
                     line_style="dashed",
                     bar_width=None,
                     add_kaiser=False,
                     add_kss = False,
                     add_broken_stick = False,
                     add_grid=True,
                     add_labels=False,
                     ha = "center",
                     va = "bottom",
                     ax=None):
        
    """ Plot the eigen values graph
        
        Parameters
        ----------
        choice : string
            Select the graph to plot :
                - If "eigenvalue" : plot the eigenvalues.
                - If "proportion" : plot the percentage of variance.
        n_components :
        title :
        x_label :
        y_label : 
        bar_fill :
        bar_color :
        line_color :
        line_tyle : 
        bar_width :
        add_labels :
        add_kss :
        add_broken_stick :
        add_grid :
        n_compon
        Returns
        -------
        None
    """

    if self.model_ == "mds":
        raise ValueError("Error :  ")

    if choice not in ["eigenvalue","proportion"]:
        raise ValueError("Error : Allowed values are 'eigenvalue' or 'proportion'.")

    # Set style size
    if ax is None:
        ax = plt.gca()
    if add_kaiser:
        add_kss = False
        add_broken_stick = False
    elif add_kss:
        add_kaiser = False
        add_broken_stick = False
    elif add_broken_stick:
        add_kaiser = False
        add_kss = False
        
    ncp = min(n_components,self.n_components_)
    if choice == "eigenvalue":
        eig = self.eig_[0][:ncp]
        text_labels = list([str(np.around(x,3)) for x in eig])
        if self.model_ not in ["famd","cmds","disqual","dismix"]:
            kaiser = self.kaiser_threshold_
        if self.model_ in ["pca","ppca","efa"]:
            kss = self.kss_threshold_
            bst = self.broken_stick_threshold_[:ncp]
        if ylabel is None:
            ylabel = "Eigenvalue"
    elif choice == "proportion":
        eig = self.eig_[2][:ncp]
        text_labels = list([str(np.around(x,1))+"%" for x in eig])
        if self.model_ not in ["famd","cmds","disqual","dismix"]:
            kaiser = self.kaiser_proportion_threshold_
    else:
        raise ValueError("Error : 'choice' variable must be 'eigenvalue' or 'proportion'.")
            
    if bar_width is None:
        bar_width = 0.5
    elif isinstance(bar_width,float)is False:
        raise ValueError("Error : 'bar_width' variable must be a float.")

    xs = pd.Categorical(np.arange(1,ncp+1))
    ys = eig

    ax.bar(xs,ys,color=bar_fill,edgecolor=bar_color,width=bar_width)
    ax.plot(xs,ys,marker="o",color=line_color,linestyle=line_style)
    if add_labels:
        for i, lab in enumerate(text_labels):
            ax.text(xs[i],ys[i],lab,ha=ha,va=va)
            
    if add_kaiser:
        ax.plot([1,ncp],[kaiser,kaiser],linestyle="dashed",color="red",label="Kaiser threshold")
        ax.legend()
            
    if choice == "eigenvalue":
        if add_kss :
            if self.model_ in ["pca","ppca","efa"]:
                ax.plot([1,ncp],[kss,kss],linestyle="dashed",color="red",label="Karlis - Saporta - Spinaki threshold")
                ax.legend()
            else:
                raise ValueError(f"Error : 'add_kss' is not allowed for an instance of class {self.model_.upper()}.")
                
        if add_broken_stick:
            if self.model_ in ["pca","ppca","efa"]:
                ax.plot(xs,bst,marker="o",color="red",linestyle="dashed",label ="Broken stick threshold")
                ax.legend()
            else:
                raise ValueError(f"Error : 'add_broken_stick' is not allowed for an instance of class {self.model_.upper()}.")

    if title is None:
        title = "Scree plot"
    if xlabel is None:
        xlabel = "Dimensions"
    if ylabel is None:
        ylabel = "Percentage of explained variances"
            
        # Set
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xticks=xs)
    ax.grid(visible=add_grid)


###################################################################################################################################"
#           PLOT FACTOR ANALYSIS OF MIXED DATA
####################################################################################################################""

def plotFAMD(self,
            choice ="ind",
            axis=[0,1],
            xlim=None,
            ylim=None,
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
            add_vline=True,
            ha="center",
            va="center",
            add_circle=True,
            quanti_sup=True,
            hline_color="black",
            hline_style="dashed",
            vline_color="black",
            vline_style ="dashed",
            patch_color = "black",
            random_state=None,
            repel=False,
            ax=None,
            **kwargs):
    """
    
    """

    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an instance of class FAMD.")
    
    if choice not in ["ind","var","col","mod"]:
        raise ValueError("Error : 'choice' ")
    
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_-1)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    
    if choice == "ind":
        coord = self.row_coord_[:,axis]
        cos2 = self.row_cos2_[:,axis]
        contrib = self.row_contrib_[:,axis]
        labels = self.row_labels_
        if title is None:
            title = "Individuals factor map - FAMD"
    elif choice == "col":
        coord = self.col_coord_[:,axis]
        cos2 = self.col_cos2_[:,axis]
        contrib = self.col_contrib_[:,axis]
        labels = self.col_labels_
        if title is None:
            title = "Graph of continuous variables - FAMD"
    elif choice == "mod":
        coord = self.mod_coord_[:,axis]
        cos2 = self.mod_cos2_[:,axis]
        contrib = self.mod_contrib_[:,axis]
        if short_labels:
            labels = self.short_labels_
        else:
            labels = self.mod_labels_
        if title is None:
            title = "Graph of the categories - FAMD"
    elif choice == "var":
        coord = self.col_cos2_[:,axis]
        contrib = np.append(self.col_contrib_[:,axis],self.var_contrib_[:,axis],axis=0)
        if title is None:
            title = "Graphe of variables - FAMD"
    else:
        raise ValueError("Error : Allowed values are 'ind', 'col', 'mod' and 'var'.")
    
    # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    if choice in ["ind","mod"]:
        if xlim is None:
            xlim = list([np.min(xs)-0.05,np.max(xs)+0.05])
        if ylim is None:
            ylim = list([np.min(ys)-0.05,np.max(ys)+0.05])
    elif choice == "var":
        xlim = (-0.1,1.1)
        ylim = (-0.1,1.1)
    else:
        xlim = (-1.1,1.1)
        ylim = (-1.1,1.1)

    #if choice in ["ind","mod","col"]:
    if color == "cos2" and choice != "var":
        c = np.sum(cos2,axis=1)
    elif color == "contrib":
        c = np.sum(contrib,axis=1)
    if color in ["cos2","contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))
    
    if choice == "ind":
        if habillage is None:
            if color in ["cos2","contrib"]:
                p = ax.scatter(xs,ys,c=c,s=len(c),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
                plt.colorbar(p).ax.set_title(label=color,weight='bold')
                # Add labels
                if repel:
                    texts = list()
                    for i, lab in enumerate(labels):
                        colorVal = scalarMap.to_rgba(c[i])
                        texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
                else:
                    for i, lab in enumerate(labels):
                        colorVal = scalarMap.to_rgba(c[i])
                        ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)

            else:
                ax.scatter(xs,ys,c=color,marker=marker,**kwargs)
                # Add labels
                if repel:
                    texts = list()
                    for i, lab in enumerate(labels):
                        texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
                else:
                    for i, lab in enumerate(labels):
                        ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)

        else:
            # Add Categorical variable
            if self.quali_sup_labels_ is not None:
                color_list=list([x[4:] for x in list(mcolors.TABLEAU_COLORS.keys())])
                marker_list = list(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
                vsQual = self.data_[habillage]
                modality_list = list(np.unique(vsQual))
                random.seed(random_state)
                color_dict = dict(zip(modality_list,random.sample(color_list,len(modality_list))))
                marker_dict = dict(zip(modality_list,random.sample(marker_list,len(modality_list))))
                for group in modality_list:
                    idx = np.where(vsQual==group)
                    ax.scatter(xs[idx[0]],ys[idx[0]],label=group,c= color_dict[group],marker = marker_dict[group])
                    if repel:
                        texts=list()
                        for i in idx[0]:
                            texts.append(ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va))
                        adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_dict[group],lw=1.0),ax=ax)
                    else:
                        for i in idx[0]:
                            ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc="center right",title=habillage, bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)

        if ind_sup:
            if self.row_sup_labels_ is not None:
                # Reset xlim and ylim
                xxs = self.row_sup_coord_[:,axis[0]]
                yys = self.row_sup_coord_[:,axis[1]]
                # Add supplementary row coordinates
                ax.scatter(xxs,yys,c=color_sup,marker=marker_sup)
                if repel:
                    texts = list()
                    for i,lab in enumerate(self.row_sup_labels_):
                        texts.append(ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for i,lab in enumerate(self.row_sup_labels_):
                        ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup)
                # Add Hotelling Ellipse 
                if hotelling_ellipse:
                    num = len(axis)*(len(xs)**2-1)*st.f.ppf(0.95,len(axis),len(xs)-len(axis))
                    denum = len(xs)*(len(xs)-len(axis))
                    c = num/denum
                    e1 = 2*math.sqrt(self.eig_[0][axis[0]]*c)
                    e2 = 2*math.sqrt(self.eig_[0][axis[1]]*c)
                    # Add Epplipse
                    ellipse = Ellipse((0,0),width=e1,height=e2,facecolor="none",edgecolor="tomato",linestyle="--")
                    ax.add_patch(ellipse)
    elif choice == "col":
        if color in ["cos2","contrib"]:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal)
        else:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color)  
        
        if quanti_sup:
            if self.quanti_sup_labels_ is not None:
                xxs = self.col_sup_coord_[:,axis[0]]
                yys = self.col_sup_coord_[:,axis[1]]
                # Add labels
                if repel:
                    texts=list()
                    for j, lab in enumerate(self.col_sup_labels_):
                        ax.arrow(0,0,xxs[j],yys[j],head_width=0.02,length_includes_head=True,color=color_sup)
                        texts.append(ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for j, lab in enumerate(self.quanti_sup_labels_):
                        ax.arrow(0,0,xxs[j],yys[j],head_width=0.02,length_includes_head=True,color=color_sup)
                        ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color=color_sup)
        if add_circle:
             ax.add_patch(plt.Circle((0,0),1,color=patch_color,fill=False))
    elif choice == "mod":
        if color in ["cos2","contrib"]:
            p = ax.scatter(xs,ys,c=c,s=len(c),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
            plt.colorbar(p).ax.set_title(label=color,weight='bold')
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[i])
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[i])
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)

        else:
            ax.scatter(xs,ys,c=color,marker=marker,**kwargs)
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
        
        if add_mod_sup:
            if self.quali_sup_labels_ is not None:
                # Reset xlim and ylim
                xxs = self.mod_sup_coord_[:,axis[0]]
                yys = self.mod_sup_coord_[:,axis[1]]
                # Add supplementary row coordinates
                ax.scatter(xxs,yys,color=color_sup,marker=marker_sup)
                if repel:
                    texts = list()
                    for i,lab in enumerate(self.mod_sup_labels_):
                        texts.append(ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for i,lab in enumerate(self.mod_sup_labels_):
                        ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup)
    else:
        # Add qualitative correlation ratio
        xxs = self.var_eta2_[:,axis[0]]
        yys = self.var_eta2_[:,axis[1]]
        if color == "contrib":
            # Append all informations
            xs = np.append(xs,xxs,axis=0)
            ys = np.append(ys,yys,axis=0)
            # Labels
            labels = self.quanti_labels_ + self.quali_labels_
            # Scatter plot
            p = ax.scatter(xs,ys,c=c,s=len(c),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
            plt.colorbar(p).ax.set_title(label=color,weight='bold')
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[i])
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[i])
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)
        elif color == "cos2":
            raise ValueError("Error : 'cos2' is not allowed.")
        else:
            ax.scatter(xs,ys, color="blue",marker=">")
            ax.scatter(xxs,yys, color="red",marker = "^")
            if repel:
                texts1 = list()
                for i, lab in enumerate(self.quanti_labels_):
                    texts1.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color="blue"))
                adjust_text(texts1,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color="blue",lw=1.0),ax=ax)
                texts2 = list()
                for j,lab in enumerate(self.quali_labels_):
                    texts2.append(ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color="red"))
                adjust_text(texts2,x=xxs,y=yys,arrowprops=dict(arrowstyle="->",color="red",lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(self.quanti_labels_):
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color="blue")
                for j, lab in enumerate(self.quali_labels_):
                    ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color="red")
                
    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   


#####################################################################################"
#               PLOT MULTIPLE CORRESPONDANCE ANALYSIS (MCA)
#######################################################################################

def plotMCA(self,
            choice ="ind",
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
            ax=None,
            **kwargs):

    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an instance of class MCA.")
    
    if choice not in ["ind","mod","var"]:
        raise ValueError("Error : 'choice' ")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    
    if choice == "ind":
        coord = self.row_coord_[:,axis]
        cos2 = self.row_cos2_[:,axis]
        contrib = self.row_contrib_[:,axis]
        labels = self.row_labels_
        if title is None:
            title = "Individuals - MCA"
    elif choice == "mod":
        coord = self.mod_coord_[:,axis]
        cos2 = self.mod_cos2_[:,axis]
        contrib = self.mod_contrib_[:,axis]
        if short_labels:
            labels = self.short_labels_
        else:
            labels = self.mod_labels_
        if title is None:
            title = "Qualitatives variables categories - MCA"
    elif choice == "var":
        coord = self.var_eta2_[:,axis]
        cos2 = self.var_cos2_[:,axis]
        contrib = self.var_contrib_[:,axis]
        labels = self.var_labels_
        if title is None:
            title = "Graphe of variables - MCA"
    else:
        raise ValueError("Error : 'choice'")
    
    # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    if color == "cos2":
        gradient = np.sum(cos2,axis=1)
    elif color == "contrib":
        gradient = np.sum(contrib,axis=1)
    
    # Set colors
    if color in ["cos2","contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(gradient), vmax=np.max(gradient))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))
    
    if choice == "ind":
        if habillage is None:
            if color in ["cos2","contrib"]:
                p = ax.scatter(xs,ys,c=gradient,s=len(gradient),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
                plt.colorbar(p).ax.set_title(label=color,weight='bold')
                # Add labels
                if repel:
                    texts = list()
                    for i, lab in enumerate(labels):
                        colorVal = scalarMap.to_rgba(gradient[i])
                        texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
                else:
                    for i, lab in enumerate(labels):
                        colorVal = scalarMap.to_rgba(gradient[i])
                        ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)

            else:
                ax.scatter(xs,ys,c=color,marker=marker,**kwargs)
                # Add labels
                if repel:
                    texts = list()
                    for i, lab in enumerate(labels):
                        texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
                else:
                    for i, lab in enumerate(labels):
                        ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)

        else:
            # Add Categorical variable
            if self.quali_sup_labels_ is not None:
                color_list=list([x[4:] for x in list(mcolors.TABLEAU_COLORS.keys())])
                marker_list = list(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
                vsQual = self.data_[habillage]
                modality_list = list(np.unique(vsQual))
                random.seed(random_state)
                color_dict = dict(zip(modality_list,random.sample(color_list,len(modality_list))))
                marker_dict = dict(zip(modality_list,random.sample(marker_list,len(modality_list))))
                for group in modality_list:
                    idx = np.where(vsQual==group)
                    ax.scatter(xs[idx[0]],ys[idx[0]],label=group,c= color_dict[group],marker = marker_dict[group])
                    if repel:
                        texts=list()
                        for i in idx[0]:
                            texts.append(ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va))
                        adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_dict[group],lw=1.0),ax=ax)
                    else:
                        for i in idx[0]:
                            ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc="center right",title=habillage, bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)

        if ind_sup:
            if self.row_sup_labels_ is not None:
                # Reset xlim and ylim
                xxs = self.row_sup_coord_[:,axis[0]]
                yys = self.row_sup_coord_[:,axis[1]]
                # Add supplementary row coordinates
                ax.scatter(xxs,yys,c=color_sup,marker=marker_sup)
                if repel:
                    texts = list()
                    for i,lab in enumerate(self.row_sup_labels_):
                        texts.append(ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for i,lab in enumerate(self.row_sup_labels_):
                        ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup)
                # Add Hotelling Ellipse 
                if hotelling_ellipse:
                    num = len(axis)*(len(xs)**2-1)*st.f.ppf(0.95,len(axis),len(xs)-len(axis))
                    denum = len(xs)*(len(xs)-len(axis))
                    c = num/denum
                    e1 = 2*math.sqrt(self.eig_[0][axis[0]]*c)
                    e2 = 2*math.sqrt(self.eig_[0][axis[1]]*c)
                    # Add Epplipse
                    ellipse = Ellipse((0,0),width=e1,height=e2,facecolor="none",edgecolor="tomato",linestyle="--")
                    ax.add_patch(ellipse)
    elif choice == "mod":
        if color in ["cos2","contrib"]:
            p = ax.scatter(xs,ys,c=gradient,s=len(gradient),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
            plt.colorbar(p).ax.set_title(label=color,weight='bold')
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(gradient[i])
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(gradient[i])
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)

        else:
            ax.scatter(xs,ys,c=color,marker=marker,**kwargs)
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
        
        if add_mod_sup:
            if self.quali_sup_labels_ is not None:
                # Reset xlim and ylim
                xxs = self.mod_sup_coord_[:,axis[0]]
                yys = self.mod_sup_coord_[:,axis[1]]
                # Add supplementary row coordinates
                ax.scatter(xxs,yys,color=color_sup,marker=marker_sup)
                # 
                if short_labels:
                    mod_sup_labels = self.short_sup_labels_
                else:
                    mod_sup_labels = self.mod_sup_labels_
                if repel:
                    texts = list()
                    for i,lab in enumerate(mod_sup_labels):
                        texts.append(ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for i,lab in enumerate(mod_sup_labels):
                        ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup)
    else:
        if color not in ["cos2","contrib"]:
            ax.scatter(xs,ys, color=color,marker=marker)
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
        else:
            p = ax.scatter(xs,ys,c=gradient,s=len(gradient),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
            plt.colorbar(p).ax.set_title(label=color,weight='bold')
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(gradient[i])
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(gradient[i])
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)
    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)


################################################################################
#           PLOT MULTIDIMENSIONAL SCALING (MDS)
###############################################################################

def plotMDS(self,
            axis=[0,1],
            xlim=(None,None),
            ylim=(None,None),
            title =None,
            xlabel=None,
            ylabel=None,
            color="blue",
            marker="o",
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
            ax=None) -> plt:
    
    if self.model_ != "mds":
        raise ValueError("Error : 'self' must be an instance of class MDS.")
     
    if ((len(axis) !=2) or 
            (axis[0] < 0) or 
            (axis[1] > self.n_components_)  or
            (axis[0] > axis[1])) :
            raise ValueError("Error : You must pass a valid axis")
        
    if ax is None:
        ax = plt.gca()
    
    xs = self.coord_[:,axis[0]]
    ys = self.coord_[:,axis[1]]
    ax.scatter(xs,ys,color=color,marker=marker)
    if repel:
        texts =list()
        for i,lab in enumerate(self.labels_):
            texts.append(ax.text(xs[i],ys[i],lab,color=color,ha=ha,va=va))
        adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
    else:
        for i,lab in enumerate(self.labels_):
            ax.text(xs[i],ys[i],lab,color=color,ha="center",va="center")

    if title is None:
        title = "Multidimensional scaling"

    # Add elements
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   


####################################################################################"
#               PRINCIPAL COMPONENTS ANALYSIS
# #################################################################################"

def plotPCA(self,choice ="ind",axis=[0,1],xlim=(None,None),ylim=(None,None),title =None,color="blue",marker="o",
            add_grid =True,ind_sup=False,color_sup = "red",marker_sup ="^",hotelling_ellipse=False,
            habillage = None,short_labels=True,color_map ="jet",add_hline = True,add_vline=True,ha="center",va="center",
            add_circle=True,quanti_sup=True,hline_color="black",hline_style="dashed",vline_color="black",
            vline_style ="dashed",patch_color = "black",
            random_state=None,repel=False,ax=None,**kwargs) -> plt:
    
    """ Plot the Factor map for individuals and variables

    Parameters
    ----------
    self : an instance of class PCA
    choice : str 
    axis : tuple or list of two elements
    xlim : tuple or list of two elements
    ylim : tuple or list of two elements
    title : str
    color : str
    marker : str
             The marker style for active points
    add_grid : bool
    ind_sup : bool
    color_sup : str : 
                The markers colors
    marker_sup : str
                 The marker style for supplementary points
    color_map : str
    add_hline : bool
    add_vline : bool
    ha : horizontalalignment : {'left','center','right'}
    va : verticalalignment {"bottom","baseline","center","center_baseline","top"}
    hline_color :
    hline_style :
    vline_color :
    vline_style :
    ax :
    **kwargs : Collection properties

    Returns
    -------
    None
    """

    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an instance of class PCA.")
    
    if choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' ")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    
    if choice == "ind":
        coord = self.row_coord_[:,axis]
        cos2 = self.row_cos2_[:,axis]
        contrib = self.row_contrib_[:,axis]
        labels = self.row_labels_
        if title is None:
            title = "Individuals factor map - PCA"
    else:
        coord = self.col_coord_[:,axis]
        cos2 = self.col_cos2_[:,axis]
        contrib = self.col_contrib_[:,axis]
        labels = self.col_labels_
        if title is None:
            title = "Variables factor map - PCA"
            
    # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    if color == "cos2":
        c = np.sum(cos2,axis=1)
    elif color == "contrib":
        c = np.sum(contrib,axis=1)
    
    if color in ["cos2","contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))
    
    if choice == "ind":
        if habillage is None:
            if color in ["cos2","contrib"]:
                p = ax.scatter(xs,ys,c=c,s=len(c),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
                plt.colorbar(p).ax.set_title(label=color,weight='bold')
                # Add labels
                if repel:
                    texts = list()
                    for i, lab in enumerate(labels):
                        colorVal = scalarMap.to_rgba(c[i])
                        texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
                else:
                    for i, lab in enumerate(labels):
                        colorVal = scalarMap.to_rgba(c[i])
                        ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)

            else:
                ax.scatter(xs,ys,c=color,marker=marker,**kwargs)
                # Add labels
                if repel:
                    texts = list()
                    for i, lab in enumerate(labels):
                        texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
                else:
                    for i, lab in enumerate(labels):
                        ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
        else:
            # Add Categorical variable
            if self.quali_sup_labels_ is not None:
                color_list=list([x[4:] for x in list(mcolors.TABLEAU_COLORS.keys())])
                marker_list = list(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
                vsQual = self.data_[habillage]
                modality_list = list(np.unique(vsQual))
                random.seed(random_state)
                color_dict = dict(zip(modality_list,random.sample(color_list,len(modality_list))))
                marker_dict = dict(zip(modality_list,random.sample(marker_list,len(modality_list))))
                for group in modality_list:
                    idx = np.where(vsQual==group)
                    ax.scatter(xs[idx[0]],ys[idx[0]],label=group,c= color_dict[group],marker = marker_dict[group])
                    if repel:
                        texts=list()
                        for i in idx[0]:
                            texts.append(ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va))
                        adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_dict[group],lw=1.0),ax=ax)
                    else:
                        for i in idx[0]:
                            ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(title=habillage, bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)

        if ind_sup:
            if self.row_sup_labels_ is not None:
                # Reset xlim and ylim
                xxs = self.row_sup_coord_[:,axis[0]]
                yys = self.row_sup_coord_[:,axis[1]]
                # Add supplementary row coordinates
                ax.scatter(xxs,yys,c=color_sup,marker=marker_sup)
                if repel:
                    texts = list()
                    for i,lab in enumerate(self.row_sup_labels_):
                        texts.append(ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for i,lab in enumerate(self.row_sup_labels_):
                        ax.text(xxs[i],yys[i],lab,ha=ha,va=va,color=color_sup)
                # Add Hotelling Ellipse 
                if hotelling_ellipse:
                    num = len(axis)*(len(xs)**2-1)*st.f.ppf(0.95,len(axis),len(xs)-len(axis))
                    denum = len(xs)*(len(xs)-len(axis))
                    c = num/denum
                    e1 = 2*math.sqrt(self.eig_[0][axis[0]]*c)
                    e2 = 2*math.sqrt(self.eig_[0][axis[1]]*c)
                    # Add Epplipse
                    ellipse = Ellipse((0,0),width=e1,height=e2,facecolor="none",edgecolor="tomato",linestyle="--")
                    ax.add_patch(ellipse)
        if self.quali_sup_labels_ is not None:
            if habillage is None:
                xxs = np.array(self.mod_sup_coord_[:,axis[0]])
                yys = np.array(self.mod_sup_coord_[:,axis[1]])
                ax.scatter(xxs,yys,color="red")
                if short_labels:
                    mod_sup_labels = self.short_sup_labels_
                else:
                    mod_sup_labels = self.mod_sup_labels_
                if repel:
                    texts =list()
                    for i,lab in enumerate(mod_sup_labels):
                        texts.append(ax.text(xxs[i],yys[i],lab,color="red"))
                    adjust_text(texts,x=xxs,y=yys,arrowprops=dict(arrowstyle="->",color="red",lw=1.0),ax=ax)
                else:
                    for i,lab in enumerate(mod_sup_labels):
                        ax.text(xxs[i],yys[i],lab,color="red")
    else:
        if color in ["cos2","contrib"]:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal)
        else:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color)  
        
        if quanti_sup:
            if self.quanti_sup_labels_ is not None:
                xxs = self.col_sup_coord_[:,axis[0]]
                yys = self.col_sup_coord_[:,axis[1]]
                # Add labels
                if repel:
                    texts=list()
                    for j, lab in enumerate(self.quanti_sup_labels_):
                        ax.arrow(0,0,xxs[j],yys[j],head_width=0.02,length_includes_head=True,color=color_sup)
                        texts.append(ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color=color_sup))
                    adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_sup,lw=1.0),ax=ax)
                else:
                    for j, lab in enumerate(self.quanti_sup_labels_):
                        ax.arrow(0,0,xxs[j],yys[j],head_width=0.02,length_includes_head=True,color=color_sup)
                        ax.text(xxs[j],yys[j],lab,ha=ha,va=va,color=color_sup)
        if add_circle:
             ax.add_patch(plt.Circle((0,0),1,color=patch_color,fill=False))
    
    if choice == "var":
        xlim = ylim = (-1.1,1.1)

    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   


########################################################################################################
##          PARTIAL PRINCIPAL COMPONENTS ANALYSIS
########################################################################################################

def plotPPCA(self,
             choice ="ind",
             axis=[0,1],
             xlim=(None,None),
             ylim=(None,None),
             title =None,
             color="blue",
             marker="o",
            add_grid =True,
            color_map ="jet",
            add_hline = True,
            add_vline=True,
            ha="center",
            va="center",
            add_circle=True,
            hline_color="black",
            hline_style="dashed",
            vline_color="black",
            vline_style ="dashed",
            patch_color = "black",
            repel=False,
            ax=None,**kwargs) -> plt:
    
    """ Plot the Factor map for individuals and variables

    Parameters
    ----------
    self : aninstance of class PCA
    choice : {'ind', 'var'}, default = 'ind'
    axis : tuple or list of two elements
    xlim : tuple or list of two elements
    ylim : tuple or list of two elements
    title : str
    color : str
    marker : str
             The marker style for active points
    add_grid : bool
    color_map : str
    add_hline : bool
    add_vline : bool
    ha : horizontalalignment : {'left','center','right'}
    va : verticalalignment {"bottom","baseline","center","center_baseline","top"}
    hline_color :
    hline_style :
    vline_color :
    vline_style :
    ax :
    **kwargs : Collection properties

    Returns
    -------
    None
    """

    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an instance of class PPCA.")
    
    if choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' ")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    
    if choice == "ind":
        coord = self.row_coord_[:,axis]
        cos2 = self.row_cos2_[:,axis]
        contrib = self.row_contrib_[:,axis]
        labels = self.row_labels_
        if title is None:
            title = "Individuals factor map - Partial PCA"
    else:
        coord = self.col_coord_[:,axis]
        cos2 = self.col_cos2_[:,axis]
        contrib = self.col_contrib_[:,axis]
        labels = self.col_labels_
        if title is None:
            title = "Variables factor map - Partial PCA"
            
    # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    if color == "cos2":
        c = np.sum(cos2,axis=1)
    elif color == "contrib":
        c = np.sum(contrib,axis=1)
    
    if color in ["cos2","contrib"]:
        cNorm  = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=plt.get_cmap(color_map))
    
    if choice == "ind":
        if color in ["cos2","contrib"]:
            p = ax.scatter(xs,ys,c=c,s=len(c),marker=marker,cmap=plt.get_cmap(color_map),**kwargs)
            plt.colorbar(p).ax.set_title(label=color,weight='bold')
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[i])
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[i])
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=colorVal)

        else:
            ax.scatter(xs,ys,c=color,marker=marker,**kwargs)
            # Add labels
            if repel:
                texts = list()
                for i, lab in enumerate(labels):
                    texts.append(ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for i, lab in enumerate(labels):
                    ax.text(xs[i],ys[i],lab,ha=ha,va=va,color=color)
    else:
        if color in ["cos2","contrib"]:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=colorVal,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    colorVal = scalarMap.to_rgba(c[j])
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=colorVal)
                    #plt.colorbar(p).ax.set_title(label=color,weight='bold')
                    #cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap(color_map),norm=cNorm,orientation='vertical')
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=colorVal)
        else:
            if repel:
                texts = list()
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    texts.append(ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color,lw=1.0),ax=ax)
            else:
                for j, lab in enumerate(labels):
                    ax.arrow(0,0,xs[j],ys[j],head_width=0.02,length_includes_head=True,color=color)
                    ax.text(xs[j],ys[j],lab,ha=ha,va=va,color=color)  
        
        if add_circle:
             ax.add_patch(plt.Circle((0,0),1,color=patch_color,fill=False))
    
    if choice == "var":
        xlim = ylim = (-1.1,1.1)

    # Add elements
    proportion = self.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   

    
########################################################################################3
#               
###########################################################################################

def plot_shepard(self,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 add_grid=True,
                 ax=None) -> plt:
    """Computes the Shepard plot

    Parameter:
    ---------
    self: An instance of class CMDS/MDS
    title : title
    xlabel : x-axis labels
    ylabel : y-axis labels
    add_grid : boolean. default = True.
    ax : default = None

    Return
    ------
    None
        
    """

    if self.model_ not in ["cmds","mds"]:
        raise ValueError("Error : 'Method' is allowed only for multidimensional scaling.")
    if ax is None:
        ax =plt.gca()

    # Scatter plot
    ax.scatter(self.dist_,self.dist_,color="steelblue")
    ax.scatter(self.dist_,self.res_dist_,color = "steelblue")

    if title == None:
        title = "Shepard Diagram"
    if xlabel is None:
        xlabel =  "input distance"
    if ylabel is None:
        ylabel =  "output distance"
    
    ax.set(xlabel = xlabel, ylabel =ylabel,title= title)
    ax.grid(visible=add_grid)


######################################################################################
#           Canonical Discriminant Analysis (CANDISC)
######################################################################################

def plotCANDISC(self,
                axis = [0,1],
                xlabel = None,
                ylabel = None,
                title = None,
                add_grid = True,
                add_hline = True,
                add_vline=True,
                marker = None,
                color = None,
                repel = False,
                show_text = False, 
                legend_title = None,
                hline_color="black",
                hline_style="dashed",
                vline_color="black",
                vline_style ="dashed",
                ha = "center",
                va = "center",
                random_state= 0,
                ax=None) -> plt:
    
    """
    
    """
    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class 'candisc'.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()
    

    coord = self.row_coord_[:,axis]
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    labels = self.row_labels_
    color_list=list([x[4:] for x in list(mcolors.TABLEAU_COLORS.keys())])
    marker_list = list(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
    classe = self.data_[self.target_]
    modality_list = list(np.unique(classe))

    if color is None:
        random.seed(random_state)
        color_dict = dict(zip(modality_list,random.sample(color_list,len(modality_list))))
    else:
        color_dict = dict(zip(modality_list,color))
    
    if marker is None:
        random.seed(random_state)
        marker_dict = dict(zip(modality_list,random.sample(marker_list,len(modality_list))))
    else:
        marker_dict = dict(zip(modality_list,marker))

    for group in modality_list:
        idx = np.where(classe==group)
        ax.scatter(xs[idx[0]],ys[idx[0]],label=group,c= color_dict[group],marker = marker_dict[group])
        if show_text:
            if repel:
                texts=list()
                for i in idx[0]:
                    texts.append(ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va))
                adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_dict[group],lw=1.0),ax=ax)
            else:
                for i in idx[0]:
                    ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    if legend_title is None:
        legend_title = "Classe"

    ax.legend(title=legend_title, bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)

    if xlabel is None:
        xlabel = f"Canonical {axis[0]+1}"
    
    if ylabel is None:
        ylabel = f"Canonical {axis[1]+1}"
    
    if title is None:
        title = "Canonical Discriminant Analysis"

    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.grid(visible=add_grid)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   


#######################################################################################
#               Hierarchical Clustering on Principal Components
#######################################################################################

def plot_dendrogram(self,**kwargs):
    """
    
    
    """
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(self.linkage_matrix_,labels=self.labels_,**kwargs)
    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'],ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y,'o',c=c);
                plt.annotate("%.3g" % y,(x,y),xytext=(0, -5),textcoords='offset points',va='top', ha='center');
        if max_d:
          plt.axhline(y=max_d, c = "k");

def rgb_hex(color):
    '''converts a (r,g,b) color (either 0-1 or 0-255) to its hex representation.
    for ambiguous pure combinations of 0s and 1s e,g, (0,0,1), (1/1/1) is assumed.'''
    message='color must be an iterable of length 3.'
    assert hasattr(color, '__iter__'), message
    assert len(color)==3, message
    if all([(c<=1)&(c>=0) for c in color]): color=[int(round(c*255)) for c in color] # in case provided rgb is 0-1
    color=tuple(color)
    return '#%02x%02x%02x' % color

def get_cluster_colors(n_clusters, my_set_of_20_rgb_colors, alpha=0.8, alpha_outliers=0.05):
    cluster_colors = my_set_of_20_rgb_colors
    cluster_colors = [c+[alpha] for c in cluster_colors]
    outlier_color = [0,0,0,alpha_outliers]
    return [cluster_colors[i%19] for i in range(n_clusters)] + [outlier_color]

def cluster_and_plot_dendrogram(self, threshold,default_color='black'):

    # get cluster labels
    Z = self.linkage_matrix_
    labels         = fcluster(Z, threshold, criterion='distance') - 1
    labels_str     = [f"cluster #{l}: n={c}\n" for (l,c) in zip(*np.unique(labels, return_counts=True))]
    n_clusters     = len(labels_str)

    cluster_colors = [rgb_hex(c[:-1]) for c in get_cluster_colors(n_clusters, alpha=0.8, alpha_outliers=0.05)]
    cluster_colors_array = [cluster_colors[l] for l in labels]
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else cluster_colors_array[x] for x in i12)
        link_cols[i+1+len(Z)] = c1 if c1 == c2 else 'k'

    # plot dendrogram with colored clusters
    fig = plt.figure(figsize=(12, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data points')
    plt.ylabel('Distance')

    # plot dendrogram based on clustering results
    dendrogram(
        Z,
        labels = self.row_labels_,
        color_threshold=threshold,
        truncate_mode = 'level',
        p = 5,
        show_leaf_counts = True,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=False,
        link_color_func=lambda x: link_cols[x],
        above_threshold_color=default_color,
        distance_sort='descending',
        ax=plt.gca()
    )
    plt.axhline(threshold, color='k')
    for i, s in enumerate(labels_str):
        plt.text(0.8, 0.95-i*0.04, s,
                transform=plt.gca().transAxes,
                va='top', color=cluster_colors[i])
    
    fig.patch.set_facecolor('white')

    return labels 



def plotHCPC(self,
             axis=(0,1),
             xlabel=None,
             ylabel=None,
             title=None,
             legend_title = None,
             random_state=None,
             xlim=None,
             ylim=None,
             show_clust_cent = False, 
             center_marker_size=200,
             center_text_size = 20,
             marker = None,
             color = None,
             repel=True,
             ha = "center",
             va = "center",
             add_grid=True,
             add_hline=True,
             add_vline=True,
             hline_color="black",
             vline_color="black",
             hline_style = "dashed",
             vline_style = "dashed",
             ax=None):
    
    if self.model_ != "hcpc":
        raise ValueError("Error : 'self' must be an object of class HCPC.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.factor_model_.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()

    if legend_title is None:
        legend_title = "cluster"
    
    coord = self.factor_model_.row_coord_[:,axis]

     # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]
    labels = self.labels_

    color_list=list([x[4:] for x in list(mcolors.TABLEAU_COLORS.keys())])
    marker_list = list(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
    cluster = self.cluster_
    modality_list = list(np.unique(cluster))
    if color is None:
        random.seed(random_state)
        color_dict = dict(zip(modality_list,random.sample(color_list,len(modality_list))))
    else:
        color_dict = dict(zip(modality_list,color))
    
    if marker is None:
        random.seed(random_state)
        marker_dict = dict(zip(modality_list,random.sample(marker_list,len(modality_list))))
    else:
        marker_dict = dict(zip(modality_list,marker))
    
    for group in modality_list:
        idx = np.where(cluster==group)
        ax.scatter(xs[idx[0]],ys[idx[0]],label=group,c= color_dict[group],marker = marker_dict[group])
        if repel:
            texts=list()
            for i in idx[0]:
                texts.append(ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va))
            adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_dict[group],lw=1.0),ax=ax)
        else:
            for i in idx[0]:
                ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(title=legend_title,bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)

    # Add cluster center
    if show_clust_cent:
        cluster_center = self.cluster_centers_.values[:,axis]
        xxs = cluster_center[:,axis[0]]
        yys = cluster_center[:,axis[1]]
        # For overlap text alebl
        texts=list()
        for i,name in enumerate(self.cluster_centers_.index):
            ax.scatter(xxs[i],yys[i],c=list(color_dict.values())[i],marker=list(marker_dict.values())[i],s=center_marker_size)
            if repel:
                texts.append(ax.text(xxs[i],yys[i],name,c=color_dict[name],ha=ha,va=va,fontsize=center_text_size))
                adjust_text(texts,x=xxs,y=yys,arrowprops=dict(arrowstyle="->",color=color_dict[name],lw=1.0),ax=ax)
            else:
                ax.text(xxs[i],yys[i],name,c=color_dict[name],ha=ha,va=va,fontsize=center_text_size)

    # Add elements
    proportion = self.factor_model_.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   
    

#######################################################################################################################
#           VARHCPC
#######################################################################################################################


def plotVARHCPC(self,
                axis=(0,1),
                xlabel=None,
                ylabel=None,
                title=None,
                legend_title = None,
                random_state=None,
                xlim=None,
                ylim=None,
                show_clust_cent = False, 
                center_marker_size=200,
                center_text_size = 20,
                marker = None,
                color = None,
                repel=True,
                ha = "center",
                va = "center",
                add_grid=True,
                add_hline=True,
                add_vline=True,
                hline_color="black",
                vline_color="black",
                hline_style = "dashed",
                vline_style = "dashed",
                ax=None):
    
    if self.model_ != "varhcpc":
        raise ValueError("Error : 'self' must be an object of class VARHCPC.")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.factor_model_.n_components_-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("Error : You must pass a valid 'axis'.")
    
    if ax is None:
        ax = plt.gca()

    if legend_title is None:
        legend_title = "cluster"
    
    if self.factor_model_.model_ == "pca":
        coord = self.factor_model_.col_coord_[:,axis]
    elif self.factor_model_.model_ == "mca":
        coord = self.factor_model_.mod_coord_[:,axis]

     # Extract coordinates
    xs = coord[:,axis[0]]
    ys = coord[:,axis[1]]

    # Extract labels
    labels = self.labels_

    color_list=list([x[4:] for x in list(mcolors.TABLEAU_COLORS.keys())])
    marker_list = list(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
    cluster = self.cluster_
    modality_list = list(np.unique(cluster))
    if color is None:
        random.seed(random_state)
        color_dict = dict(zip(modality_list,random.sample(color_list,len(modality_list))))
    else:
        color_dict = dict(zip(modality_list,color))
    
    if marker is None:
        random.seed(random_state)
        marker_dict = dict(zip(modality_list,random.sample(marker_list,len(modality_list))))
    else:
        marker_dict = dict(zip(modality_list,marker))
    
    for group in modality_list:
        idx = np.where(cluster==group)
        ax.scatter(xs[idx[0]],ys[idx[0]],label=group,c= color_dict[group],marker = marker_dict[group])
        if repel:
            texts=list()
            for i in idx[0]:
                texts.append(ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va))
            adjust_text(texts,x=xs,y=ys,arrowprops=dict(arrowstyle="->",color=color_dict[group],lw=1.0),ax=ax)
        else:
            for i in idx[0]:
                ax.text(xs[i],ys[i],labels[i],c=color_dict[group],ha=ha,va=va)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(title=legend_title,bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)

    # Add cluster center
    if show_clust_cent:
        cluster_center = self.cluster_centers_.values[:,axis]
        xxs = cluster_center[:,axis[0]]
        yys = cluster_center[:,axis[1]]
        # For overlap text alebl
        texts=list()
        for i,name in enumerate(self.cluster_centers_.index):
            ax.scatter(xxs[i],yys[i],c=list(color_dict.values())[i],marker=list(marker_dict.values())[i],s=center_marker_size)
            if repel:
                texts.append(ax.text(xxs[i],yys[i],name,c=color_dict[name],ha=ha,va=va,fontsize=center_text_size))
                adjust_text(texts,x=xxs,y=yys,arrowprops=dict(arrowstyle="->",color=color_dict[name],lw=1.0),ax=ax)
            else:
                ax.text(xxs[i],yys[i],name,c=color_dict[name],ha=ha,va=va,fontsize=center_text_size)

    # Add elements
    proportion = self.factor_model_.eig_[2]
    xlabel = "Dim."+str(axis[0]+1)+" ("+str(round(proportion[axis[0]],2))+"%)"
    ylabel = "Dim."+str(axis[1]+1)+" ("+str(round(proportion[axis[1]],2))+"%)"
    ax.grid(visible=add_grid)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
    if add_hline:
        ax.axhline(y=0,color=hline_color,linestyle=hline_style)
    if add_vline:
        ax.axvline(x=0,color=vline_color,linestyle=vline_style)   
