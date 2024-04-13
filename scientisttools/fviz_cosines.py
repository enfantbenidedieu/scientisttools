# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
import numpy as np

def fviz_cosines(self,
                 choice="ind",
                 axis=None,
                 x_label=None,
                 top_cos2=10,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 short_labels=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    
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
        
        Returns
        -------
        None
        """

    if choice not in ["row","col","ind","var","mod","quanti_var","quali_var","ind_sup"]:
        raise ValueError("Error : 'choice' not allowed.")
    
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise ValueError("Error : 'axis' must be an integer.")
    elif axis < 0 or axis > self.n_components_:
        raise ValueError(f"Error : 'axis' must be an integer between 0 and {self.n_components_ - 1}")

    if xlabel is None:
        xlabel = "Cos2 - Quality of representation"
    if bar_width is None:
        bar_width = 0.5
    if top_cos2 is None:
        top_cos2 = 10
        
    if choice == "ind":
        name = "individuals"
        if self.model_ == "ca":
            name = "rows"
        cos2 = self.row_cos2_[:,axis]
        labels = self.row_labels_
    elif choice == "var" :
        if self.model_ != "mca":
            name = "continues variables"
            cos2 = self.col_cos2_[:,axis]
            labels  = self.col_labels_
            if self.model_ == "ca":
                name = "columns"
        else:
            name = "categorical variables"
            cos2 = self.var_cos2_[:,axis]
            labels  = self.var_labels_
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
            raise ValueError("Error : Factor Model must have at least two supplementary continuous variables.")
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

    df = pd.DataFrame({"labels" : labels_sort, "cos2" : cos2_sorted})

    p = pn.ggplot(df,pn.aes(x = "reorder(labels,cos2)", y = "cos2"))+pn.geom_bar(stat="identity",fill=color,width=bar_width)

    title = f"Cosinus of {name} to Dim-{axis+1}"
    p = p + pn.ggtitle(title)+pn.xlab(name)+pn.ylab(xlabel)
    p = p + pn.coord_flip()

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"),
                         axis_text_x = pn.element_text(angle = 60, ha = "center", va = "center"))

    return p+ggtheme