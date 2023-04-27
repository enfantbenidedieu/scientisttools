# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text

def plotPPCA(self,choice ="ind",axis=[0,1],xlim=(None,None),ylim=(None,None),title =None,color="blue",marker="o",
            add_grid =True,color_map ="jet",add_hline = True,add_vline=True,ha="center",va="center",
            add_circle=True,hline_color="black",hline_style="dashed",vline_color="black",
            vline_style ="dashed",patch_color = "black",repel=False,ax=None,**kwargs) -> plt:
    
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

    
