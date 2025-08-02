# -*- coding: utf-8 -*-
from pandas import DataFrame, Categorical
from plotnine import ggplot, aes,geom_point,guides,coord_flip,scale_fill_gradient2,labs,theme, theme_minimal

#intern function
from scientisttools.methods.functions.pivot_longer import pivot_longer

def fviz_corrplot(X,
                  x_label=None,
                  y_label=None,
                  title=None,
                  col_outline = "gray",
                  gradient_cols = ["blue","white","red"],
                  legend_title = "Corr",
                  show_legend = True,
                  ggtheme = theme_minimal()):
    """
    Corrplot
    --------
    
    Description
    -----------

    Usage
    -----
    ```python
    >>> fviz_corrplot(X,x_label=None,y_label=None,title=None,col_outline = "gray",gradient_cols = ["blue","white","red"],legend_title = "Corr",show_legend = True,ggtheme = theme_minimal())
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with `n` rows and `p` columns

    `x_label`: a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label`: a string specifying the label text of y (by default = None and a x_label is chosen).

    `title`: a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `col_outline`: a string specifying the point outline color

    `gradient_cols`:  a list/tuple of 3 colors for low, mid and high values (by default = ("blue", "white", "red")).

    `legend_title`: a string corresponding to the title of the legend (by default = "Corr").

    `show_legend`: a boolean to either add or not legend (by default = True).

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes: theme_gray(), theme_bw(), theme_classic(), theme_void(), ....

    Return
    ------

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_corrplot
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #variables squared cosinus
    >>> print(fviz_corrplot(res_pca.var_.cos2,title="Variables cos2",legend_title="Cos2"))
    >>> #variables contributions
    >>> print(fviz_corrplot(res_pca.var_.contrib,title="Variables contributions",legend_title="Contrib"))
    ```
    """
    #check if X is an instance of pandas DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    if not isinstance(gradient_cols,(list,tuple)):
        raise TypeError("'gradient_cols' must be a list or tuple of colors")
    elif len(gradient_cols) != 3:
        raise ValueError("'gradient_cols' must be a list or tuple with length 3.")
    
    X.columns, X.index = Categorical(X.columns,categories=X.columns.tolist()), Categorical(X.index,categories=X.index.tolist())
    melt = pivot_longer(X)

    p = (ggplot(melt,aes(x="Var1",y="Var2",fill="value")) 
         + geom_point(aes(size="value"),color=col_outline,shape="o")
         + guides(size=None)
         + coord_flip()
         + scale_fill_gradient2(low = gradient_cols[0],high = gradient_cols[2],mid = gradient_cols[1],name = legend_title))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##add others informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if title is None:
        title = "Correlation"
    if y_label is None:
        y_label = "Dimensions"
    if x_label is None:
        x_label = "Variables"
    p = p + labs(title=title,x=x_label,y=y_label)
        
    # Removing legend
    if not show_legend:
        p =p + theme(legend_position=None)
    
    return p + ggtheme