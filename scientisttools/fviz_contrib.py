# -*- coding: utf-8 -*-
import pandas as pd
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
        
    Parameters
    ----------
    choice : allowed values are :
            - 'row' for CA
            - 'col' for CA
            - 'var' for PCA or MCA
            - 'ind' for PCA
            - 'quanti_var' for FAMD, MFA, MFAMIX
            - 'quali_var' for FAMD, MFAQUAL
            - 'freq' for MFACT
            - 'group' for MFA, MFAQUAL, MFAMIX, MFACT
            - 'partial_axes' for MFA, MFAQUAL, MFAMIX, MFACT
        
    axis : None or int.
        Select the axis for which the row/col contributions are plotted. If None, axis = 0.
        
    y_label : None or str (default).
        The label text.
        
    top_contrib : an integer value specifying the number of top elements to be shown.
            
    bar_width : None, float or array-like.
        The width(s) of the bars.

    add_grid : bool or None, default = True.
        Whether to show the grid lines.
    
    fill_color : a fill color for the bar plot, default = "steelblue"
    
    color : color or list of color, default = "steelblue".
        The colors of the bar faces.
    
    sort_contrib : a string specifying whether the value should be sorted. Allowed values are :
                    - "none" for no sorting
                    - 'asc' for ascending
                    - 'desc' for descending

    xtickslab_rotation : x text ange

    ggtheme : function, plotnine theme name. Default value is theme_pubr(). 
               Allowed values include plotnine official themes: 
               theme_gray(), theme_bw(), theme_minimal(), theme_classic(), theme_void(),
        
    Returns
    -------
    a plotnine figure
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """    
        
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
    
    if self.model_ == "efa" and choice != "var":
        raise ValueError("'choice' should be 'var'")
    
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

    if sort_contrib not in ["desc","asc","none"]:
        raise ValueError("'sort_contrib' should be one of 'desc', 'asc' or 'none'.")

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
        contrib = self.ind_["contrib"]
    elif choice == "var":
        contrib = self.var_["contrib"]
    elif choice == "quali_var":
        contrib = self.quali_var_["contrib"]
    elif choice == "quanti_var":
        contrib = self.quanti_var_["contrib"]
    elif choice == "freq":
        contrib = self.freq_["contrib"]
    elif choice == "group":
        contrib = self.group_["contrib"]
    elif choice == "partial_axes":
        contrib = pd.DataFrame().astype("float")
        for grp in self.partial_axes_["contrib"].columns.get_level_values(0).unique().tolist():
            data = self.partial_axes_["contrib"][grp].T
            data.index = [x+"."+str(grp) for x in data.index]
            contrib = pd.concat([contrib,data],axis=0)
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
    if choice == "quanti_var":
        if self.model_ in ["mfa","mfamix"]:
            contrib = contrib.rename(columns={"name" : "variable"})
            contrib = pd.merge(contrib,self.group_label_,on=["variable"]).rename(columns={"variable" : "name"})
            
            if sort_contrib == "desc":
                p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,-contrib)",y="contrib",group = 1,color="group",fill="group"),
                                    width=bar_width,stat="identity")
            elif sort_contrib == "asc":
                p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,contrib)",y="contrib",group = 1,color="group",fill="group"),
                                    width=bar_width,stat="identity")
            else:
                p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="name",y="contrib",group = 1,color="group",fill="group"),
                                    width=bar_width,stat="identity")
        else:
            if sort_contrib == "desc":
                p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,-contrib)",y="contrib",group = 1),
                                    fill=fill_color,color=color,width=bar_width,stat="identity")
            elif sort_contrib == "asc":
                p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,contrib)",y="contrib",group = 1),
                                    fill=fill_color,color=color,width=bar_width,stat="identity")
            else:
                p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="name",y="contrib",group = 1),
                                    fill=fill_color,color=color,width=bar_width,stat="identity")
    elif choice == "freq":
        if self.model_ != "mfact":
             raise TypeError("'self' must be an object of class MFACT")
        
        contrib = contrib.rename(columns={"name" : "variable"})
        contrib = pd.merge(contrib,self.group_label_,on=["variable"]).rename(columns={"variable" : "name"})
        
        if sort_contrib == "desc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,-contrib)",y="contrib",group = 1,color="group",fill="group"),
                                width=bar_width,stat="identity")
        elif sort_contrib == "asc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,contrib)",y="contrib",group = 1,color="group",fill="group"),
                                width=bar_width,stat="identity")
        else:
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="name",y="contrib",group = 1,color="group",fill="group"),
                                width=bar_width,stat="identity")
           

    else:
        if sort_contrib == "desc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,-contrib)",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
        elif sort_contrib == "asc":
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="reorder(name,contrib)",y="contrib",group = 1),
                                fill=fill_color,color=color,width=bar_width,stat="identity")
        else:
            p = p + pn.geom_bar(data=contrib,mapping=pn.aes(x="name",y="contrib",group = 1),
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