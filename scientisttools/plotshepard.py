# -*- coding: utf-8 -*-

##########
# Shepard plot for Muldimensionnal Scaling

import matplotlib.pyplot as plt

def plot_shepard(self,title=None,xlabel=None,ylabel=None,add_grid=True,ax=None) -> plt:
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
        raise ValueError("Error : 'Method' is allowed only for multidimensional scalling.")
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