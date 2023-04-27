# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import math
import random
import matplotlib.colors as mcolors
from adjustText import adjust_text

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

    
