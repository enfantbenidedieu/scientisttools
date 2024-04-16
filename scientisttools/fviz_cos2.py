# -*- coding: utf-8 -*-
import pandas as pd
import plotnine as pn

def fviz_cos2(self,
              choice="ind",
              axis=None,
              y_label=None,
              top_cos2=None,
              bar_width=None,
              add_grid=True,
              fill_color = "steelblue",
              color = "steelblue",
              sort_cos2 = "desc",
              xtickslab_rotation = 45,
              ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize the quality of representation of row/columns elements
    ---------------------------------------------------------------

    Description
    -----------
    This function can be used to visualize the quality of representation of rows/columns from the results of Principal Component Analysis (PCA), 
    Correspondence Analysis (CA), Multiple Correspondence Analysis (MCA), Factor Analysis of Mixed Data (FAMD), and Multiple Factor Analysis (MFA) functions.       
        
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
        
    top_cos2 : None or int.
        Set the maximum number of values to plot.
        If top_cos2 is None : all the values are plotted.
            
    bar_width : None, float or array-like.
        The width(s) of the bars.

    add_grid : bool or None, default = True.
        Whether to show the grid lines.

    color : color or list of color, default = "steelblue".
        The colors of the bar faces.
        
    Returns
    -------
    None
    """
    if self.model_ not in ["pca","ca","mca","famd","mfa","mfaqual","mfamix","mfact","partialpca"]:
        raise TypeError("'self' must be an object of class PCA, CA, MCA, FAMD, MFA, MFAQUAL, MFAMIX, MFACT, PartialPCA")
        
    if choice not in ["row","col","var","ind","quanti_var","quali_var","freq","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'row', 'col', 'var', 'ind', 'quanti_var', 'quali_var',  'freq','group' 'partial_axes'.")

    ncp = self.call_["n_components"]
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise TypeError("'axis' must be an integer.")
    elif axis not in list(range(0,ncp)):
        raise TypeError(f"'axis' must be an integer between 0 and {ncp-1}.")
            
    if bar_width is None:
        bar_width = 0.8

    ################## set
    if self.model_ in ["pca","mca","partialpca"] and choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'var', 'ind'")
    
    if self.model_ == "ca" and choice not in ["row","col"]:
        raise ValueError("'choice' should be one of 'row', 'col'.")
    
    if self.model_ == "famd" and choice not in ["ind","quanti_var","quali_var"]:
        raise ValueError("'choice' should be one of 'ind', 'quanti_var','quali_var'")
    
    if self.model_ == "mfa" and choice not in ["ind","quanti_var","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'ind', 'quanti_var', 'group', 'partial_axes'")
    
    if self.model_ == "mfaqual" and choice not in ["ind","quali_var","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'ind', 'quali_var', 'group', 'partial_axes'")
    
    if self.model_ == "mfamix" and choice not in ["ind","quanti_var","quali_var","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'ind', 'quanti_var', 'quali_var','group', 'partial_axes'")
    
    if self.model_ == "mfact" and choice not in ["ind","freq","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'ind', 'freq', 'group', 'partial_axes'")

    if sort_cos2 not in ["desc","asc","none"]:
        raise ValueError("'sort_cos2' should be one of 'desc', 'asc' or 'none'.")

    #### Set names
    if choice == "ind":
        name = "individuals"
    elif choice == "var":
        name = "variables"
    elif choice == "quali_var":
        name = "qualitative variables"
    elif choice == "quanti_var":
        name = "quantitative variables"
    elif choice == "freq":
        name = "frequencies"
    elif choice == "group":
        name = "groups"
    elif choice == "partial_axes":
        name = "partial axes"
    elif choice == "row":
        name = "rows"
    elif choice == "col":
        name = "columns"
    
    # Extract contribution
    if choice == "ind":
        cos2 = self.ind_["cos2"]
    elif choice == "var":
        cos2 = self.var_["cos2"]
    elif choice == "quali_var":
        cos2 = self.quali_var_["cos2"]
    elif choice == "quanti_var":
        cos2 = self.quanti_var_["cos2"]
    elif choice == "freq":
        cos2 = self.freq_["cos2"]
    elif choice == "group":
        cos2 = self.group_["cos2"]
    elif choice == "partial_axes":
        cos2 = pd.DataFrame().astype("float")
        for grp in self.partial_axes_["cos2"].columns.get_level_values(0).unique().tolist():
            data = self.partial_axes_["cos2"][grp].T
            data.index = [x+"."+str(grp) for x in data.index]
            cos2 = pd.concat([cos2,data],axis=0)
    elif choice == "row":
        cos2 = self.row_["cos2"]
    elif choice == "col":
        cos2 = self.col_["cos2"]
    
    ####
    cos2 = cos2.iloc[:,[axis]].reset_index()
    cos2.columns = ["name","cos2"]
    
    #####
    if top_cos2 is not None:
            cos2 = cos2.sort_values(by="cos2",ascending=False).head(top_cos2)
    
    p = pn.ggplot()
    if sort_cos2 == "desc":
        p = p + pn.geom_bar(data=cos2,mapping=pn.aes(x="reorder(name,-cos2)",y="cos2",group = 1),
                            fill=fill_color,color=color,width=bar_width,stat="identity")
    elif sort_cos2 == "asc":
        p = p + pn.geom_bar(data=cos2,mapping=pn.aes(x="reorder(name,cos2)",y="cos2",group = 1),
                            fill=fill_color,color=color,width=bar_width,stat="identity")
    else:
        p = p + pn.geom_bar(data=cos2,mapping=pn.aes(x="name",y="cos2",group = 1),
                            fill=fill_color,color=color,width=bar_width,stat="identity")
    if y_label is None:
        y_label = "Cos2 - Quality of representation"
    title = f"Cos2 of {name} to Dim-{axis+1}"
    p = p + pn.labs(title=title,y=y_label,x="")

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