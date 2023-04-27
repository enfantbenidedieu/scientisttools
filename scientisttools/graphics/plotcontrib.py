# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            'var' :   continues variables
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
    cos2_sorted = np.sort(contrib)[limit:n]
    labels_sort = pd.Series(labels)[np.argsort(contrib)][limit:n]
    r = np.arange(n_labels)
    ax.barh(r,cos2_sorted,height=bar_width,color=color,align="edge")
    ax.set_yticks([x + bar_width/2 for x in r], labels_sort)
    ax.set(title=f"Contribution of {name} to Dim-{axis+1}",xlabel=xlabel,ylabel=name,xlim=(0,100))
    ax.grid(visible=add_grid)
