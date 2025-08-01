# -*- coding: utf-8 -*-
from plotnine import ggplot, aes,geom_point,guides,coord_flip,scale_fill_gradient2,labs, theme, theme_minimal
from pandas import DataFrame, Categorical

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
    >>> fviz_coorplot(X,**kwargs)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with `n` rows and `p` columns


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
    >>> print(fviz_corrplot(res_pca.var_.coord))
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