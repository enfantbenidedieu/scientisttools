# -*- coding: utf-8 -*-
from numpy import linspace, sin, cos, pi, issubdtype,number,asarray,ndarray,sqrt,arctan2
from pandas import DataFrame, Series, Categorical, concat
from plotnine import (
    ggplot,
    aes,
    geom_point, geom_segment,
    scale_color_gradient2,
    guides,
    guide_legend,
    stat_ellipse,
    scale_color_manual, scale_fill_manual,
    arrow,
    annotate,
    theme_minimal,
    labs, 
    xlim,ylim,
    geom_hline, geom_vline,
    theme,
    element_line, element_text,
    geom_text, geom_label,
    annotate)

#list of colors
list_colors = ["black", "red", "green", "blue", "cyan", "magenta","darkgray", "darkgoldenrod", "darkgreen", "violet",
                "turquoise", "orange", "lightpink", "lavender", "yellow","lightgreen", "lightgrey", "lightblue", "darkkhaki",
                "darkmagenta", "darkolivegreen", "lightcyan", "darkorange","darkorchid", "darkred", "darksalmon", 
                "darkseagreen","darkslateblue", "darkslategray", "darkslategrey","darkturquoise", 
                "darkviolet", "lightgray", "lightsalmon","lightyellow", "maroon"]
            
def coord_adjust(coord:DataFrame,
                 cos2:DataFrame,
                 contrib:DataFrame,
                 color,
                 legend_title,
                 habillage,
                 palette,
                 lim_cos2,
                 lim_contrib):
    """
    Coordinates Adjustment
    ----------------------

    Description
    -----------

    Parameters
    ----------
    `coord`: a pandas DataFrame containing factor coordinates

    `cos2`: a pandas DataFrame containing squared cosinus

    `contrib`: a pandas DataFrame conatining contributions

    `color`: color for individuals and variables, respectively. Can be a continuous variable or a factor variable. Possible values include also : "cos2", "contrib", "coord", "x" or "y". In this case, the colors for individuals/variables are automatically controlled by their qualities of representation ("cos2"), contributions ("contrib"), coordinates (x**2+y**2, "coord"), x values ("x") or y values ("y"). To use automatic coloring (by cos2, contrib, ....), make sure that habillage = None.

    `habillage`: an optional factor variable for coloring the observations by groups. Default value is None. 

    `palette`: the color palette to be used for coloring or filling by groups. 

    `lim_cos2`: a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib`: a numeric specifying the relative contribution limit (by default = None)

    Return
    ------
    `coord`: a pandas DataFrame containing factor coordinates


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #set color
    if isinstance(color,str):
        if color == "cos2":
            if legend_title is None:
                legend_title = "Cos2"
            coord.insert(coord.shape[1],legend_title,cos2.sum(axis=1))
        elif color == "contrib":
            if legend_title is None:
                legend_title = "Contrib"
            coord.insert(coord.shape[1],legend_title,contrib.sum(axis=1))
        elif color == "coord":
            if legend_title is None:
                legend_title = "Coord"
            coord.insert(coord.shape[1],legend_title,coord.iloc[:,:2].pow(2).sum(axis=1))
        elif color in coord.columns.tolist():
            if not issubdtype(coord[color].dtype,number):
                raise TypeError("'color' must me a numeric variable.")
            if legend_title is None:
                legend_title = color
            coord = coord.rename(columns={color : legend_title})
        index = None
    elif isinstance(color,ndarray):
        if legend_title is None:
            legend_title = "Cont_Var"
        coord.insert(coord.shape[1],legend_title,asarray(color))
        index = None
    elif hasattr(color,"labels_"):
        if legend_title is None:
            legend_title = "Cluster"
        coord.insert(coord.shape[1],legend_title,[str(x+1) for x in color.labels_])
        index = coord[legend_title].unique().tolist()
        coord[legend_title] = Categorical(coord[legend_title],categories=sorted(index),ordered=True)
    elif isinstance(color,(list,tuple,Series)) and len(color) == coord.shape[0] and all(isinstance(x,str) for x in color):
        if legend_title is None:
            legend_title = "Group"
        coord.insert(coord.shape[1],legend_title,[str(x+1) for x in color.labels_])
        index = coord[legend_title].unique().tolist()
        coord[legend_title] = Categorical(coord[legend_title],categories=sorted(index),ordered=True)

    #set palette
    if habillage is not None or hasattr(color,"labels_") or (isinstance(color,(list,tuple,Series)) and len(color) == coord.shape[0] and all(isinstance(x,str) for x in color)):    
        if habillage is not None:
            index = coord[habillage].unique().tolist()
        #set palette
        if palette is None:
            palette = list_colors[:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
    
    #using lim cos2
    if lim_cos2 is not None:
        if isinstance(lim_cos2,(int,float)):
            lim_cos2 = float(lim_cos2)
            cos2 = cos2.sum(axis=1).to_frame("cosinus").sort_values(by="cosinus",ascending=False).query("cosinus > @lim_cos2")
            if cos2.shape[0] != 0:
                coord = coord.loc[cos2.index,:]
        else:
            raise TypeError("'lim_cos2' must be a float or an integer")
    
    #using lim contrib
    if lim_contrib is not None:
        if isinstance(lim_contrib,(int,float)):
            lim_contrib = float(lim_contrib)
            contrib = contrib.sum(axis=1).to_frame("contrib").sort_values(by="contrib",ascending=False).query("contrib > @lim_contrib")
            if contrib.shape[0] != 0:
                coord = coord.loc[contrib.index,:]
        else:
            raise TypeError("'lim_contrib' must be a float or an integer")

    return coord, legend_title, index, palette

def fviz_scatter(self,
                 element = "ind",
                 axis = [0,1],
                 geom = ("point","text"),
                 repel = False,
                 lim_cos2 = None,
                 lim_contrib = None,
                 color = "black",
                 point_args = dict(shape="o"),
                 text_args = dict(size=8),
                 gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                 legend_title = None,
                 habillage = None,
                 palette = None,
                 add_ellipses = False, 
                 ellipse_level = 0.95,
                 ellipse_type = "norm",
                 ellipse_alpha = 0.1):
    """
    Creat a scatter plot with text
    ------------------------------
    
    Usage
    -----
    ```python
    >>> fviz_scatter(self,**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCA, PartialPCA, FactorAnalysis, CA, MCA, FAMD, PCAMIX,

    `element`: a string specifying the name of the active elements (by default = "ind")

    `axis`: a numeric list of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `geom`: a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. 
        * "point" to show only points; 
        * "text" to show only labels; 
        * ("point","text") to show both types.
    
    `repel`: a boolean, whether to avoid overplotting text labels or not (by default == False)

    `lim_cos2`: a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib`: a numeric specifying the relative contribution limit (by default = None),

    `color`: a color for the active rows points (by default = "black"). Can be a continuous variable or a labels from scikit-learn KMeans functions. 
        Possible values include also: "cos2", "contrib". In this case, the colors for active individuals are automatically controlled by their qualities of representation ("cos2") or contributions ("contrib"). 
        To use automatic coloring (by cos2, contrib), make sure that habillage = None.

    `point_args`: a dictionary containing others keyword arguments for active rows points (see https://plotnine.org/reference/geom_point.html).

    `text_args`: a dictionary containing keyword arguments for `geom_text` (see https://plotnine.org/reference/geom_text.html)

    `gradient_cols`:  a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `legend_title`: a string corresponding to the title of the legend (by default = None).

    `habillage`: a string or an integer specifying the variables of indexe for coloring the observations by groups. Default value is None.

    `palette`:  a list or tuple specifying the color palette to be used for coloring or filling by groups.

    `add_ellipses`: a boolean to either add or not ellipses (by default = False). see https://plotnine.org/reference/stat_ellipse.html

    `ellipse_level`: a numeric specifying the size of the concentration ellipse in normal probability (by default = 0.95)
    
    `ellipse_type`: a character specifying frame type. Possible values are : "norm" (default), "t", or "euclid".
        * `norm`: assumes a multivariate normal distribution.
        * `t`: assumes a multivariate t-distribution.
        * `eulclid`: draws a circle with the radius equal to level, representing the euclidean distance from the center.
    
    `ellipse_alpha`: alpha for ellipse specifying the transparency level of fill color. Use alpha = 0 for no fill color (by default = 0.1)

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA
    >>> #instanciation
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #graph of individuals
    >>> from scientisttools import fviz_scatter
    >>> p = fviz_scatter(res_pca,element="ind",repel=True)
    >>> print(p)
    ```
    """
    #check if axis is an instance of list
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")

    #check if valid axis
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])):
        raise ValueError("You must pass a valid 'axis'.")
    
    #check if geom is valid
    if isinstance(geom,str):
        if geom not in ["point","text"]:
            raise ValueError("The specified value for the argument 'geom' are not allowed")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in ["point","text"]]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed")
    
    #for individuals points (PCA, PartialPCA, FactorAnalysis)
    if element == "ind":
        coord, cos2, contrib = concat((self.ind_.coord.iloc[:,axis],self.call_.Xtot),axis=1), self.ind_.cos2.iloc[:,axis], self.ind_.contrib.iloc[:,axis]
        if hasattr(self,"ind_sup_"):
            coord = coord.drop(index=self.call_.ind_sup)
    #for rows points (CA)
    if element == "row" and self.model_ == "ca":
        coord, cos2, contrib = concat((self.row_.coord.iloc[:,axis],self.call_.Xtot),axis=1), self.row_.cos2.iloc[:,axis], self.row_.contrib.iloc[:,axis]
        if hasattr(self,"row_sup_"):
            coord = coord.drop(index=self.call_.row_sup)
    #for columns points (CA)
    if element == "col" and self.model_ == "ca":
        coord, cos2, contrib = self.col_.coord.iloc[:,axis], self.col_.cos2.iloc[:,axis], self.col_.contrib.iloc[:,axis]
    #for variables categories in multiple correspondence analysis (MCA)
    if element == "var" and self.model_ == "mca":
        coord, cos2, contrib = self.var_.coord.iloc[:,axis], self.var_.cos2.iloc[:,axis], self.var_.contrib.iloc[:,axis]
    #for qualitative variables in FAMD, PCAMIX, MPCA, MFAMIX, MFAQUAL
    if element == "quali_var" and self.model_ in ("famd","pcamix","mpca","mfamix","mfaqual"):
        coord, cos2, contrib = self.var_.coord.iloc[:,axis], self.var_.cos2.iloc[:,axis], self.var_.contrib.iloc[:,axis]

    #set text arguments
    if repel:
        text_args = dict(**text_args,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))
    
    #update coordinates
    coord, legend_title, index, palette = coord_adjust(coord=coord,cos2=cos2,contrib=contrib,color=color,legend_title=legend_title,habillage=habillage,palette=palette,lim_cos2=lim_cos2,lim_contrib=lim_contrib)
    
    #iitialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if habillage is None : 
        if (isinstance(color,str) and color in [*["cos2","contrib","coord"],*coord.columns]) or (isinstance(color,ndarray)):
            if "point" in geom:
                p = p + geom_point(aes(color=legend_title),**point_args) + scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title)
            if "text" in geom:
                p = p + geom_text(aes(color=legend_title),**text_args)
        elif hasattr(color,"labels_") or (isinstance(color,(list,tuple,Series)) and len(color) == coord.shape[0] and all(isinstance(x,str) for x in color)):
            if "point" in geom:
                p = p + geom_point(aes(color=legend_title,fill=legend_title),**point_args) + guides(color=guide_legend(title=legend_title))
            if "text" in geom:
                p = p + geom_text(aes(color=legend_title),**text_args)
        else:
            if "point" in geom:
                p = p + geom_point(color=color,**point_args)
            if "text" in geom:
                p = p + geom_text(color=color,**text_args)
    else:
        if habillage not in coord.columns:
            raise ValueError(f"{habillage} not in DataFrame.")
        if "point" in geom:
            p = p + geom_point(aes(color=habillage,fill=habillage),**point_args)
        if "text" in geom:
            p = p + geom_text(aes(color=habillage),**text_args)
        if add_ellipses:
            p = p + stat_ellipse(mapping=aes(color=habillage,fill=habillage),type=ellipse_type,level=ellipse_level,alpha=ellipse_alpha)
    #set color manual
    if habillage is not None or hasattr(color,"labels_") or (isinstance(color,(list,tuple,Series)) and len(color) == coord.shape[0] and all(isinstance(x,str) for x in color)):
        p = p + scale_color_manual(values=palette,labels=index) + scale_fill_manual(values=palette,labels=index)
    
    return p

##https://stackoverflow.com/questions/64935396/how-do-i-shift-the-geom-text-labels-to-after-a-geom-segment-arrow-in-ggplot2
def overlap_coord(coord,axis,repel):
    def rshift(r,theta,a=0.03,b=0.07):
        return r + a + b*abs(cos(theta))
    if repel:
        coord = (coord
                    .assign(
                        r = lambda x : sqrt(x[f"Dim.{axis[0]+1}"]**2+x[f"Dim.{axis[1]+1}"]**2),
                        theta = lambda x : arctan2(x[f"Dim.{axis[1]+1}"],x[f"Dim.{axis[0]+1}"]),
                        rnew = lambda x : rshift(r=x["r"],theta=x["theta"]),
                        xnew = lambda x : x["rnew"]*cos(x["theta"]),
                        ynew = lambda x : x["rnew"]*sin(x["theta"])
                        )
                    .drop(columns=["r","theta","rnew"]))
    else:
        coord = coord.assign(xnew = lambda x : x[f"Dim.{axis[0]+1}"], ynew = lambda x : x[f"Dim.{axis[1]+1}"])
    return coord 

def fviz_arrow(self,
               element = "var",
               axis = [0,1],
               geom = ("arrow","text"),
               repel = False,
               lim_cos2 = None,
               lim_contrib = None,
               color = "black",
               segment_args = dict(linetype="solid",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
               text_args = dict(size=8),
               gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
               legend_title = None,
               palette = None,
               scale = 1):
    """
    Create an arrow plot with text
    ------------------------------
    
    Usage
    -----
    ```python
    >>> fviz_arrow(self, element = "var", **kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCA, FAMD, PCAMIX,

    `element`: a string specifying the name of the active elements (by default = "var")

    `axis`: a numeric list of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `geom`: a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["arrow","text"]. 
        * "arrow" to show only segments and arrows; 
        * "text" to show only labels; 
        * ("arrow","text") to show both types.
    
    `repel`: a boolean, whether to avoid overplotting text labels or not (by default == False)

    `lim_cos2`: a numeric specifying the square cosinus limit (by default = None).

    `lim_contrib`: a numeric specifying the relative contribution limit (by default = None),

    `color`: a color for the active rows points (by default = "black"). Can be a continuous variable or a labels from scikit-learn KMeans functions. 
        Possible values include also: "cos2", "contrib". In this case, the colors for active individuals are automatically controlled by their qualities of representation ("cos2") or contributions ("contrib"). 
        To use automatic coloring (by cos2, contrib), make sure that habillage = None.

    `segment_args`: a dictionary containing others keyword arguments for active rows segments (see https://plotnine.org/reference/geom_segment.html).

    `text_args`: a dictionary containing keyword arguments for `geom_text` (see https://plotnine.org/reference/geom_text.html)

    `gradient_cols`: a list/tuple of 3 colors for low, mid and high correlation values (by default = ("#00AFBB", "#E7B800", "#FC4E07")).

    `legend_title`: a string corresponding to the title of the legend (by default = None).

    `palette`: a list or tuple specifying the color palette to be used for coloring or filling by groups.

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA
    >>> #instanciation
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #graph of individuals
    >>> from scientisttools import fviz_arrow
    >>> p = fviz_arrow(res_pca,element="var",repel=True)
    >>> print(p)
    ```
    """
    #check if axis is an instance of list
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")

    #check if valid axis
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    #check if geom is valid
    if isinstance(geom,str):
        if geom not in ["arrow","text"]:
            raise ValueError("The specified value for the argument 'geom' are not allowed")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in ["arrow","text"]]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed")
    
    #variables in PCA
    if element == "var" and self.model_ == "pca":
        coord, cos2, contrib = self.var_.coord.iloc[:,axis].mul(scale), self.var_.cos2.iloc[:,axis], self.var_.contrib.iloc[:,axis]
    #variables in FAMD, PCAMIX, MPCA, MFA or MFAMIX
    if element == "quanti_var" and self.model_ in ("famd","pcamix","mpca","mfa","mfamix"):
        coord, cos2, contrib = self.quanti_var_.coord.iloc[:,axis].mul(scale), self.quanti_var_.cos2.iloc[:,axis], self.quanti_var_.contrib.iloc[:,axis]

    #define text coordinates
    coord = overlap_coord(coord=coord,axis=axis,repel=repel)

    #update coordinates
    coord, legend_title, index, palette = coord_adjust(coord=coord,cos2=cos2,contrib=contrib,color=color,legend_title=legend_title,habillage=None,palette=palette,lim_cos2=lim_cos2,lim_contrib=lim_contrib)

    #set x, y 
    if repel:
        x_text, y_text = "xnew", "ynew"
    else:
        x_text, y_text = f"Dim.{axis[0]+1}", f"Dim.{axis[1]+1}"
    
    #iitialize
    p = ggplot(data=coord,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if (isinstance(color,str) and color in [*["cos2","contrib","coord"],*coord.columns]) or (isinstance(color,ndarray)):
        if "arrow" in geom:
            p = p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=legend_title),**segment_args)+scale_color_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title)
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text,color=legend_title),**text_args)
    elif hasattr(color,"labels_") or (isinstance(color,(list,tuple,Series)) and len(color) == coord.shape[0] and all(isinstance(x,str) for x in color)):
        if "arrow" in geom:
            p = p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}",color=legend_title),**segment_args) + guides(color=guide_legend(title=legend_title))
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text,color=legend_title),**text_args)
    else:
        if "arrow" in geom:
            p = p + geom_segment(aes(x=0,y=0,xend=f"Dim.{axis[0]+1}",yend=f"Dim.{axis[1]+1}"),**segment_args)
        if "text" in geom:
            p = p + geom_text(aes(x=x_text,y=y_text),color=color,**text_args)

    #set color manual and fill manual
    if hasattr(color,"labels_") or (isinstance(color,(list,tuple,Series)) and len(color) == coord.shape[0] and all(isinstance(x,str) for x in color)):
        p = p + scale_color_manual(values=palette,labels=index) + scale_fill_manual(values=palette,labels=index)
    
    return p

#add elements (point & text) to plotnine graph
def add_scatter(p,
                data:DataFrame,
                axis = [0,1],
                geom = ("point","text"),
                repel = False,
                color = "blue",
                points_args = dict(size=1.5),
                text_args = dict(size=8)):
    """
    Add elements (point & text) to plotnine graph
    ---------------------------------------------

    Usage
    -----
    ```python
    >>> add_scatter(p, data, **kwargs) 
    ```

    Parameters
    ----------
    `p`: a plotnine graph

    `data`: a pandas DataFrame containing factor coordinates

    `axis`: a numeric list of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `geom`: a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ("arrow","point","text").
        * "point" to show only points; 
        * "text" to show only labels;

    `repel`: a boolean, whether to avoid overplotting text labels or not (by default = False).

    `color`: a color for the points, texts or segments (by default = "blue").

    `point_args`: a dictionary containing others keyword arguments for `geom_point` (see https://plotnine.org/reference/geom_point.html).

    `text_args`: a dictionary containing others keyword arguments for `geom_text` (see https://plotnine.org/reference/geom_text.html)

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> from scientisttools import fviz_arrow, add_arrow
    >>> p = fviz_arrow(res_pca,element="var",repel=True)
    >>> p = add_arrow(p=p,data=res_pca.quanti_sup_.coord,geom=("arrow","text"),text_args=dict(size=8),repel=False)
    >>> print(p)
    ```
    """
    #check if data is an instance of pd.DataFrame class
    if not isinstance(data,DataFrame):
        raise TypeError(f"{type(data)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #check if geom is valid
    if isinstance(geom,str):
        if geom not in ["point","text"]:
            raise ValueError("The specified value for the argument 'geom' are not allowed")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in ["point","text"]]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed")
    
    #set text arguments
    if repel:
        text_args = dict(**text_args,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))
    
    #add points
    if "point" in geom:
        p = p + annotate("point",x=asarray(data.iloc[:,axis[0]]),y=asarray(data.iloc[:,axis[1]]),color=color,**points_args)
    #add texts
    if "text" in geom:
        p = p + geom_text(data=data,mapping=aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=data.index),color=color,**text_args)
    return p

#add elements (arrow & text) to plotnine graph
def add_arrow(p,
              data:DataFrame,
              axis = [0,1],
              geom = ("arrow","text"),
              repel = False,
              color = "blue",
              segment_args = dict(linetype="dashed",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
              text_args = dict(size=8)):
    """
    Add elements (arrow & text) to plotnine graph
    ---------------------------------------------

    Usage
    -----
    ```python
    >>> add_arrow(p, data, **kwargs) 
    ```

    Parameters
    ----------
    `p`: a plotnine graph

    `data`: a pandas DataFrame containing factor coordinates

    `axis`: a numeric list of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `geom`: a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ("arrow","text").
        * "arrow" (to show only segments with arrows)
        * "text" to show only texts;

    `repel`: a boolean, whether to avoid overplotting text labels or not (by default = False).

    `color`: a color for the points, texts or segments (by default = "blue").

    `segment_args`: a dictionary containing others keyword arguments for `geom_segment` (see https://plotnine.org/reference/geom_segment.html).

    `text_args`: a dictionary containing others keyword arguments for `geom_text` (see https://plotnine.org/reference/geom_text.html) or `geom_label` (see https://plotnine.org/reference/geom_label.html)

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_arrow, add_arrow
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> p = fviz_arrow(res_pca,element="var",repel=True)
    >>> p = add_arrow(p=p,data=res_pca.quanti_sup_.coord,geom=("arrow","text"),text_args=dict(size=8),repel=False)
    >>> print(p)
    ```
    """
    #check if data is an instance of pd.DataFrame class
    if not isinstance(data,DataFrame):
        raise TypeError(f"{type(data)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #check if geom is valid
    if isinstance(geom,str):
        if geom not in ["arrow","text"]:
            raise ValueError("The specified value for the argument 'geom' are not allowed")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in ["arrow","text"]]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed")
    
    #define text coordinates
    data = overlap_coord(coord=data,axis=axis,repel=repel)

    #set x, y for geom_text
    if repel:
        x_text, y_text = "xnew", "ynew"
    else:
        x_text, y_text = f"Dim.{axis[0]+1}", f"Dim.{axis[1]+1}"
    #add segment with arrow
    if "arrow" in geom:
        p  = p + annotate("segment",x=0,y=0,xend=asarray(data.iloc[:,axis[0]]),yend=asarray(data.iloc[:,axis[1]]),color=color,**segment_args)
    #add texts
    if "text" in geom:
        p = p + geom_text(data=data,mapping=aes(x=x_text,y=y_text,label=data.index),color=color,**text_args)
    return p

def set_axis(p,
             self,
             axis = [0,1],
             x_lim = None,
             y_lim = None,
             x_label = None,
             y_label = None,
             title = None,
             add_hline = True,
             add_vline = True,
             add_grid = True,
             ggtheme = theme_minimal()):
    """
    Add elememts to plotnine graph
    ------------------------------

    Usage
    -----
    ```python
    >>> set_axis(p, self, **kwargs) 
    ```

    Parameters
    ----------
    `p`: a plotnine graph

    `self`: an instance of class PCA, FactorAnalysis, CA, MCA, FAMD, PCAMIX, MPCA

    `axis`: a numeric list of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_lim`: a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim`: a numeric list of length 2 specifying the range of the plotted 'y' values (by default = None).

    `x_label`: a string specifying the label text of x (by default = None and a x_label is chosen).
    
    `y_label`: a string specifying the label text of y (by default = None and a y_label is chosen).
    
    `title`: a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `add_hline`: a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline`: a boolean to either add or not a vertical ligne (by default = True).

    `add_grid`: a boolean to either add or not a grid customization (by default = True).

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Returns
    -------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #set x label
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(self.eig_.iloc[axis[0],2],2))+"%)"
    #set y label
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(self.eig_.iloc[axis[1],2],2))+"%)"
    #set title
    if title is None:
        title = "Map"
    p = p + labs(title=title,x=x_label,y=y_label)
    #set x limits
    if x_lim is not None:
        p = p + xlim(x_lim)
    #set y limits
    if y_lim is not None:
        p = p + ylim(y_lim)
    #add horizontal line
    if add_hline:
        p = p + geom_hline(yintercept=0,alpha=0.5,color="black",size=0.5,linetype="dashed")
    #add vertical line
    if add_vline:
        p = p+ geom_vline(xintercept=0,alpha=0.5,color="black",size=0.5,linetype="dashed")
    #add grid
    if add_grid:
        p = p + theme(panel_grid_major=element_line(alpha=None,color="black",size=0.5,linetype="dashed"))
    
    return p + ggtheme

#drwa circle to plot
def fviz_circle(p,
                r = 1.0,
                x0 = 0.0, 
                y0 = 0.0,
                color = "black"):
    """
    Draw (Add) a circle with plotnine
    ---------------------------------

    Usage
    -----
    ```python
    >>> gg_circle(p, r, x0, y0, color) 
    ```

    Description
    -----------
    Draw (add) a circle with plotnine based on center and radius.

    Parameters
    ----------
    `p`: a plotnine graph

    `r`: a numeric value specifying the radius (by default = 1.0).

    `x0`: a numeric value specifying the `x` center (by default = 0.0).

    `y0`: a numeric value specifying the `y` center (by default = 0.0).

    `color`: a string specifying the color of the circle (by default = "black")

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Reference
    ---------
    See https://stackoverflow.com/questions/6862742/draw-a-circle-with-ggplot2
    """
    x = x0 + r*cos(linspace(0,pi,num=100))
    ymin = y0 + r*sin(linspace(0,-pi,num=100))
    ymax = y0 + r*sin(linspace(0,pi,num=100))
    return p + annotate("ribbon", x=x, ymin=ymin, ymax=ymax,color=color,fill=None)