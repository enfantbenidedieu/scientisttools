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
