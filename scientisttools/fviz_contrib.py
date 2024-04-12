# -*- coding: utf-8 -*-
import plotnine as pn

def fviz_contrib(self,
                 choice="ind",
                 axis=None,
                 y_label=None,
                 top_contrib=None,
                 bar_width=None,
                 add_grid=True,
                 fill_color = "steelblue",
                 color = "steelblue",
                 palette = "Set2",
                 sort_contrib = "desc",
                 xtickslab_rotation = 45,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize the contributions of row/columns elements
    ---------------------------------------------------

    Description
    -----------
    This function can be used to visualize the contribution of rows/columns from the results of Principal Component Analysis (PCA), 
    Correspondence Analysis (CA), Multiple Correspondence Analysis (MCA), Factor Analysis of Mixed Data (FAMD), and Multiple Factor Analysis (MFA) functions.
    

    For the selected axis, the graph represents the row or column
    cosines sorted in descending order.            
        
    Parameters
    ----------
    choice : {'ind','var','mod'}.
            'ind' :   individuals
            'var' :   continues/categorical variables
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
        
    Returns
    -------
    None
    """    
        
    if choice not in ["row","col","var","ind","quanti_var","quali_var","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'row', 'col', 'var', 'ind', 'quanti_var', 'quali_var', 'group' or 'partial_axes'.")

    ncp = self.call_["n_components"]
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise TypeError("Error : 'axis' must be an integer.")
    elif axis not in list(range(0,ncp)):
        raise TypeError(f"'axis' must be an integer between 0 and {ncp-1}.")
            
    if bar_width is None:
        bar_width = 0.8

    ################## set
    if self.model_ in ["pca","mca","partialpca"] and choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'var', 'ind'.")
    
    if self.model_ == "ca" and choice not in ["row","col"]:
        raise ValueError("Error : 'choice' should be one of 'row', 'col'.")
    
    if self.model_ == "famd" and choice not in ["ind","var","quali_var"]:
        raise ValueError("'choice' should be one of 'var', 'ind', 'quali_var'")

    if sort_contrib not in ["desc","asc","none"]:
        raise ValueError("'sort_contrib' should be one of 'desc', 'asc' or 'none'.")

    #### Set names
    if choice == "ind":
        name = "individuals"
    elif choice == "var":
        name = "variables"
    elif choice == "quali_var":
        name = "qualitative variables"
    elif choice == "row":
        name = "rows"
    elif choice == "col":
        name = "columns"
    
    # Extract contribution
    if choice == "ind":
        contrib = self.ind_["contrib"]
    elif choice == "var":
        contrib = self.var_["contrib"]
    elif choice == "quali_var":
        contrib = self.quali_var_["contrib"]
    elif choice == "row":
        contrib = self.row_["contrib"]
    elif choice == "col":
        contrib = self.col_["contrib"]
    
    ####
    contrib = contrib.iloc[:,[axis]].reset_index()
    contrib.columns = ["name","contrib"]

    # Add hline
    hvalue = 100/contrib.shape[0]
    
    #####
    if top_contrib is not None:
            contrib = contrib.sort_values(by="contrib",ascending=False).head(top_contrib)
    
    p = pn.ggplot()
    if choice == "quanti_var" and self.model_ == "mfa":
        pass
    else:
        if sort_contrib == "desc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,-contrib)",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
        elif sort_contrib == "asc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,contrib)",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
        else:
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="contrib",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
    
    if y_label is None:
        y_label = "Contributions (%)"
    title = f"Contribution of {name} to Dim-{axis+1}"
    p = p + pn.labs(title=title,y=y_label,x="")
    
    # if not (choice == "var" and self.model_ =="mca"):
    p = p + pn.geom_hline(yintercept=hvalue,linetype="dashed",color="red")

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    p = p + ggtheme

    if xtickslab_rotation > 5:
        ha = "right"
    if xtickslab_rotation == 90:
        ha = "center"

    # Rotation
    p = p + pn.theme(axis_text_x = pn.element_text(rotation = xtickslab_rotation,ha=ha))

    return p