# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text

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
                sup_coord = self.col_sup_coord[:,axis]
        

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

