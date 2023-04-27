# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from adjustText import adjust_text

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
        raise ValueError("Error : 'self' must be an instance of class Classic MDS.")
     
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