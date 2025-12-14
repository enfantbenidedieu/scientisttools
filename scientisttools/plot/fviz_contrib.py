# -*- coding: utf-8 -*-
from pandas import DataFrame, concat, merge
from plotnine import ggplot, geom_bar, geom_hline, aes,labs,theme,theme_minimal,element_line,element_text

def fviz_contrib(self,
                 element = "ind",
                 axis = None,
                 y_label = None,
                 top_contrib = None,
                 col_bar = "steelblue",
                 fill_bar = "steelblue",
                 width_bar = None,
                 add_grid = True,
                 sort = "desc",
                 xtickslab_rotation = 45,
                 ggtheme=theme_minimal()):
    
    """
    Visualize the contributions of row/columns elements
    ---------------------------------------------------

    Description
    -----------
    This function can be used to visualize the contribution of rows/columns from the results of Principal Component Analysis (PCA), Correspondence Analysis (CA), Multiple Correspondence Analysis (MCA), Factor Analysis of Mixed Data (FAMD), Mixed PCA(MPCA), and Multiple Factor Analysis (MFA) functions.     
        
    Parameters
    ----------
    `self`: an object of class PCA, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT, PartialPCA

    `element`: the element o subset. Allowed values are :
        - 'row' for CA
        - 'col' for CA
        - 'var' for PCA, PartialPCA, MCA or SpecificMCA
        - 'ind' for PCA, PartialPCA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT
        - 'quanti_var' for FAMD, MPCA, PCAMIX, MFA, MFAMIX
        - 'quali_var' for FAMD, MPCA, PCAMIX, MFAQUAL
        - 'freq' for MFACT
        - 'group' for MFA, MFAQUAL, MFAMIX, MFACT
        - 'partial_axes' for MFA, MFAQUAL, MFAMIX, MFACT
        
    `axis`: None or int. Select the axis for which the row/col contributions are plotted. If None, axis = 0.
        
    `y_label`: None or str (default).
        The y-label text.
        
    `top_contrib`: an integer value specifying the number of top elements to be shown.
            
    `width_bar`: None, float or array-like.
        The width(s) of the bars.

    `add_grid`: bool or None, default = True.
        Whether to show the grid lines.

    `col_bar`: color or list of color, default = "steelblue".
        The colors of the bar faces.

    `fill_bar`: a fill color for the bar plot, default = "steelblue"

   `sort`: None or a string specifying whether the value should be sorted. Allowed values are:
        - `None`: no sorting
        - `"asc"`: for ascending
        - `"desc"`: for dsecending

    `xtickslab_rotation`: Same as x_text_angle and y_text_angle, respectively

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes: theme_gray(), theme_bw(), theme_classic(), theme_void(), ....
    
    Returns
    -------
    a plotnine figure
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ not in ["pca","ca","mca","specificmca","famd","mpca","pcamix","mfa","mfaqual","mfamix","mfact","partialpca","fa","dmfa"]:
        raise TypeError("'self' must be an object of class PCA, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT, PartialPCA, FactorAnalysis, DMFA")    
        
    if element not in ["row","col","var","ind","quanti_var","quali_var","freq","group","partial_axes"]:
        raise ValueError("'choice' should be one of 'row', 'col', 'var', 'ind', 'quanti_var', 'quali_var',  'freq','group' 'partial_axes'.")

    ncp = self.call_.n_components
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise TypeError("'axis' must be an integer.")
    elif axis not in list(range(0,ncp)):
        raise TypeError(f"'axis' must be an integer between 0 and {ncp-1}.")
            
    if width_bar is None:
        width_bar = 0.8

    if self.model_ in ["pca","mca","specificmca","partialpca","efa","dmfa"] and element not in ["ind","var"]:
        raise ValueError("'element' should be one of 'var', 'ind'")
    
    if self.model_ == "fa" and element != "var":
        raise ValueError("'element' should be 'var'")
    
    if self.model_ == "ca" and element not in ["row","col"]:
        raise ValueError("'element' should be one of 'row', 'col'.")
    
    if self.model_ in ["famd","mpca","pcamix"] and element not in ["ind","quanti_var","quali_var"]:
        raise ValueError("'element' should be one of 'ind', 'quanti_var','quali_var'")
    
    if self.model_ == "mfa" and element not in ["ind","quanti_var","group","partial_axes"]:
        raise ValueError("'element' should be one of 'ind', 'quanti_var', 'group', 'partial_axes'")
    
    if self.model_ == "mfaqual" and element not in ["ind","quali_var","group","partial_axes"]:
        raise ValueError("'element' should be one of 'ind', 'quali_var', 'group', 'partial_axes'")
    
    if self.model_ == "mfamix" and element not in ["ind","quanti_var","quali_var","group","partial_axes"]:
        raise ValueError("'element' should be one of 'ind', 'quanti_var', 'quali_var','group', 'partial_axes'")
    
    if self.model_ == "mfact" and element not in ["ind","freq","group","partial_axes"]:
        raise ValueError("'element' should be one of 'ind', 'freq', 'group', 'partial_axes'")

    if element == "ind":
        name, contrib = "individuals", self.ind_.contrib
    elif element == "var":
        name, contrib = "variables", self.var_.contrib
    elif element == "quali_var":
        name, contrib = "qualitative variables", self.quali_var_.contrib
    elif element == "quanti_var":
        name, contrib = "quantitative variables", self.quanti_var_.contrib
    elif element == "freq":
        name, contrib = "frequencies", self.freq_.contrib
    elif element == "group":
        name, contrib = "groups", self.group_.contrib
    elif element == "partial_axes":
        name, contrib = "partial axes", DataFrame().astype("float")
        for grp in self.partial_axes_.contrib.columns.get_level_values(0).unique().tolist():
            data = self.partial_axes_.contrib[grp].T
            data.index = [x+"."+str(grp) for x in data.index]
            contrib = concat([contrib,data],axis=0)
    elif element == "row":
        name, contrib = "rows", self.row_.contrib
    elif element == "col":
        name, contrib = "columns", self.col_.contrib
    
    contrib = contrib.iloc[:,axis].to_frame().reset_index()
    contrib.columns = ["name","contrib"]

    # Add hline
    hvalue = 100/contrib.shape[0]

    if top_contrib is not None:
        contrib = contrib.sort_values(by="contrib",ascending=False).head(top_contrib)
 
    def sort_contrib(sort):
        if sort == "desc":
            return "reorder(name,-contrib)"
        elif sort == "asc":
            return"reorder(name,contrib)"
        elif sort is None:
            return "name"
        else:
            raise ValueError("'sort' must be one of 'desc','asc', None")
    
    p = ggplot()
    if element == "quanti_var":
        if self.model_ in ["mfa","mfamix"]:
            contrib = contrib.rename(columns={"name" : "variable"})
            contrib = merge(contrib,self.group_label_,on=["variable"]).rename(columns={"variable" : "name"})
            p = p + geom_bar(data=contrib,mapping=aes(x=sort_contrib(sort),y="contrib",group = 1,color="group name",fill="group name"),width=width_bar,stat="identity") + labs(fill='Groups', color="Groups")
        else:
            p = p + geom_bar(data=contrib,mapping=aes(x=sort_contrib(sort),y="contrib",group = 1),color=col_bar,fill=fill_bar,width=width_bar,stat="identity")
    elif element == "freq":
        if self.model_ != "mfact":
             raise TypeError("'self' must be an object of class MFACT")
        
        contrib = contrib.rename(columns={"name" : "variable"})
        contrib = merge(contrib,self.group_label_,on=["variable"]).rename(columns={"variable" : "name"})
        p = p + geom_bar(data=contrib,mapping=aes(x=sort_contrib(sort),y="contrib",group = 1,color="group name",fill="group name"),width=width_bar,stat="identity") + labs(fill='Groups', color="Groups")
    else:
        p = p + geom_bar(data=contrib,mapping=aes(x=sort_contrib(sort),y="contrib",group = 1),color=col_bar,fill=fill_bar,width=width_bar,stat="identity")
    
    if y_label is None:
        y_label = "Contributions (%)"
    title = f"Contribution of {name} to Dim-{axis+1}"
    p = p + labs(title=title,y=y_label,x="")
    
    p = p + geom_hline(yintercept=hvalue,linetype="dashed",color="red")
    if add_grid:
        p = p + theme(panel_grid_major = element_line(color = "black",size = 0.5,linetype = "dashed"))
    p = p + ggtheme

    if xtickslab_rotation > 5:
        ha = "right"
    if xtickslab_rotation == 90:
        ha = "center"
    p = p + theme(axis_text_x = element_text(rotation = xtickslab_rotation,ha=ha))

    return p