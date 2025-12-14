# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from plotnine import ggplot, aes,geom_bar, theme, theme_minimal, labs, element_line, element_text

def fviz_cos2(self,
              element="ind",
              axis=None,
              y_label=None,
              top_cos2=None,
              width_bar=None,
              add_grid=True,
              col_bar = "steelblue",
              fill_bar = "steelblue",
              sort = "desc",
              xtickslab_rotation = 45,
              ggtheme=theme_minimal()):
    
    """
    Visualize the quality of representation of row/columns elements
    ---------------------------------------------------------------

    Description
    -----------
    This function can be used to visualize the quality of representation of rows/columns from the results of Principal Component Analysis (PCA), 
    Correspondence Analysis (CA), (specific) Multiple Correspondence Analysis (MCA/SpecificMCA), Factor Analysis of Mixed Data (FAMD), Mixed PCA (MPCA)
    and Multiple Factor Analysis (MFA) functions.       
        
    Parameters
    ----------
    `self`: an object of class PCA, CA, MCA, SpecificMCA, FAMD, MPCA, MFA, MFAQUAL, MFAMIX, MFACT, PartialPCA

    `element` : the element to subset. Allowed values are :
            - 'row' for CA
            - 'col' for CA
            - 'var' for PCA, PartialPCA, MCA or SpecificMCA
            - 'ind' for PCA, PartialPCA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT
            - 'quanti_var' for FAMD, MPCA, PCAMIX, MFA, MFAMIX
            - 'quali_var' for FAMD, MPCA, PCAMIX, MFAQUAL
            - 'freq' for MFACT
            - 'group' for MFA, MFAQUAL, MFAMIX, MFACT
            - 'partial_axes' for MFA, MFAQUAL, MFAMIX, MFACT
        
    `axis`: None or int.
        Select the axis for which the row/col contributions are plotted. If None, axis = 0.
        
    `y_label`: None or str (default).
        The y-label text.
        
    `top_cos2`: None or int.
        Set the maximum number of values to plot.
        If top_cos2 is None : all the values are plotted.
            
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
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```
    >>> #principal component analysis
    >>> from scientisttools import decathlon, PCA, fviz_cos2
    >>> res_pca = res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None).fit(decathlon)
    >>> #variable cos2 on axis 1
    >>> fviz_cos2(res_pca, element = "var", axis = 0, top_cos2 = 10 )
    >>> #change color
    >>> fviz_cos2(res_pca, element = "var", axis = 0, fill_bar = "lightgray", col_bar = "black") 
    >>> #cos2 of individuals on axis 1
    >>> fviz_cos2(res_pca, element = "ind", axis = 0)
    >>> #
    >>> #correspondence analysis (CA)
    >>> from scientisttools import children, CA
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8).fit(children)
    >>> #visualize row cos2 on axes 1
    >>> fviz_cos2(res_ca, element ="row", axis = 0)
    >>> #visualize column cos2 on axes 1
    >>> fviz_cos2(res_ca, element ="col", axis = 0)
    >>> #
    >>> #multiple correspondence analysis (MCA)
    >>> from scientisttools import poison, MCA
    >>> res_mca <- MCA(quanti_sup = (0,1), quali_sup = (2,3)).fit(poison)
    >>> #visualize individual cos2 on axes 1
    >>> fviz_cos2(res_mca, element ="ind", axis = 0, top_cos2 = 20)
    >>> #visualize variable categorie cos2 on axes 1
    >>> fviz_cos2(res_mca, element ="var", axes = 0)
    >>> #
    >>> #factor analysis of mixed data (FAMD)
    >>> #
    >>> #
    >>> #multiple factor analysis (MFA)
    >>> from scientisttools import poison, MFAQUAL
    >>> res_mfaqual <- MFAQUAL(group=c(2,2,5,6), type=c("s","n","n","n"),name_group=c("desc","desc2","symptom","eat"),num_group_sup=(0,1)).fit(poison)
    >>> #visualize individual cos2 on axes 1
    >>> #select the top 20
    >>> fviz_cos2(res_mfaqual, element ="ind", axis = 0, top_cos2 = 20)
    >>> #visualize catecorical variable categorie cos2 on axes 1
    >>> fviz_cos2(res_mfaqual, element ="quali_var", axis = 0)
    ```
    """
    if self.model_ not in ["pca","ca","mca","specificmca","famd","mpca","pcamix","mfa","mfaqual","mfamix","mfact","partialpca","dmfa"]:
        raise TypeError("'self' must be an object of class PCA, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX, MFACT, PartialPCA, DMFA")
        
    if element not in ["row","col","var","ind","quanti_var","quali_var","freq","group","partial_axes"]:
        raise ValueError("'element' should be one of 'row', 'col', 'var', 'ind', 'quanti_var', 'quali_var',  'freq','group' 'partial_axes'.")

    ncp = self.call_.n_components
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise TypeError("'axis' must be an integer.")
    elif axis not in list(range(0,ncp)):
        raise TypeError(f"'axis' must be an integer between 0 and {ncp-1}.")
            
    if width_bar is None:
        width_bar = 0.8

    if self.model_ in ["pca","mca","specificmca","partialpca","dmfa"] and element not in ["ind","var"]:
        raise ValueError("'element' should be one of 'var', 'ind'")
    
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

    #set name and extract DataFrame
    if element == "ind":
        name, cos2 ="individuals", self.ind_.cos2
    elif element == "var":
        name, cos2 = "variables", self.var_.cos2
    elif element == "quali_var":
        name, cos2 = "qualitative variables", self.quali_var_.cos2
    elif element == "quanti_var":
        name, cos2 = "quantitative variables", self.quanti_var_.cos2
    elif element == "freq":
        name, cos2 = "frequencies", self.freq_.cos2
    elif element == "group":
        name, cos2 ="groups", self.group_.cos2
    elif element == "partial_axes":
        name, cos2 = "partial axes", DataFrame().astype("float")
        for grp in self.partial_axes_.cos2.columns.get_level_values(0).unique().tolist():
            data = self.partial_axes_.cos2[grp].T
            data.index = [x+"."+str(grp) for x in data.index]
            cos2 = concat([cos2,data],axis=0)
    elif element == "row":
        name, cos2 ="rows", self.row_.cos2
    elif element == "col":
        name, cos2 = "columns", self.col_.cos2
    
    cos2 = cos2.iloc[:,axis].to_frame().reset_index()
    cos2.columns = ["name","cos2"]
    
    if top_cos2 is not None:
            cos2 = cos2.sort_values(by="cos2",ascending=False).head(top_cos2)
    
    def sort_cos2(sort):
        if sort == "desc":
            return "reorder(name,-cos2)"
        elif sort == "asc":
            return"reorder(name,cos2)"
        elif sort is None:
            return "name"
        else:
            raise ValueError("'sort' must be one of 'desc','asc', None")
        
    p = ggplot(data=cos2,mapping=aes(x=sort_cos2(sort),y="cos2",group = 1)) + geom_bar(fill=fill_bar,color=col_bar,width=width_bar,stat="identity")
    if y_label is None:
        y_label = "Cos2 - Quality of representation"
    title = f"Cos2 of {name} to Dim-{axis+1}"
    p = p + labs(title=title,y=y_label,x="")

    if add_grid:
        p = p + theme(panel_grid_major = element_line(color = "black",size = 0.5,linetype = "dashed"))
    p = p + ggtheme

    if xtickslab_rotation > 5:
        ha = "right"
    if xtickslab_rotation == 90:
        ha = "center"

    # Rotation
    p = p + theme(axis_text_x = element_text(rotation = xtickslab_rotation,ha=ha))

    return p 