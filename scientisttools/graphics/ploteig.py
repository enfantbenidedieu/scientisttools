# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_eigenvalues(self,choice ="proportion",n_components=10,title=None,xlabel=None,ylabel=None,bar_fill="steelblue",
                     bar_color = "steelblue",line_color="black",line_style="dashed",bar_width=None,
                     add_kaiser=False, add_kss = False, add_broken_stick = False,add_grid=True,
                     add_labels=False, ha = "center",va = "bottom",ax=None):
        
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
        if self.model_ != "cmds":
            kaiser = self.kaiser_threshold_
        if self.model_ in ["pca","ppca","efa"]:
            kss = self.kss_threshold_
            bst = self.broken_stick_threshold_[:ncp]
        if ylabel is None:
            ylabel = "Eigenvalue"
    elif choice == "proportion":
        eig = self.eig_[2][:ncp]
        text_labels = list([str(np.around(x,1))+"%" for x in eig])
        if self.model_ != "cmds":
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