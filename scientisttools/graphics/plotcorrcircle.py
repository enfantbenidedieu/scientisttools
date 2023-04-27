# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from adjustText import adjust_text

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
    
    
    
    
    