# -*- coding: utf-8 -*-
from pandas import DataFrame, Categorical
from plotnine import (
    ggplot, 
    aes,
    geom_point,
    guides,
    coord_flip,
    scale_fill_gradient2,
    guide_legend,
    labs,
    theme, 
    theme_minimal
)

def fviz_corrplot(X,
                  x_label = None,
                  y_label = None,
                  title = None,
                  subtitle = None,
                  col_outline = "gray",
                  gradient_cols = ("#00AFBB", "#E7B800", "#FC4E07"),
                  legend_title = "Corr",
                  show_legend = True,
                  pntheme = theme_minimal(),
                  **kwargs):
    """
    Visualization of a correlation matrix

    A graphical display of a correlation matrix.
    
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    x_label : str, default = None
        The label text of x. If None, then x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then y_label is chosen.

    title : str, default = None
        The title of the graph you draw. If None, then a title is chosen.

    subtitle : str, default = None
        The subtitle of the graph you draw.

    col_outline : str, default = "gray"
        The point outline color.

    gradient_cols:  list, tuple, default = ("#00AFBB", "#E7B800", "#FC4E07")
        Three colors for low, mid and high correlation values.

    legend_title : str, defaut = "Corr"
        The title of the legend.

    show_legend : bool, default = True
        If True, then add legend.

    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`).
    
    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`.

    Returns
    -------
    A plotnine object

    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, fviz_corrplot
    >>> clf = PCA(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    >>> # variables squared cosinus
    >>> p = fviz_corrplot(clf.quanti_var_.cos2,title="Variables cos2",legend_title="Cos2")
    >>> print(p.show())
    >>> # variables contributions
    >>> p = fviz_corrplot(clf.quanti_var_.contrib,title="Variables contributions",legend_title="Contrib")
    >>> print(p.show())
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pandas DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check valid gradient_cols
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(gradient_cols,(list,tuple)):
        raise TypeError("'gradient_cols' must be a list or tuple of colors")
    elif len(gradient_cols) != 3:
        raise ValueError("'gradient_cols' must be a list or tuple with length 3.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data preparation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert to longer data
    df_long = X.stack(level=-1, dropna=False,future_stack=False).rename_axis(('Var1', 'Var2')).reset_index(name="value")
    # convert to categorical
    df_long["Var1"] = Categorical(df_long["Var1"],categories=X.index,ordered=True)
    df_long["Var2"] = Categorical(df_long["Var2"],categories=X.columns,ordered=True)

    # plot
    p = (
        ggplot(df_long,aes(x="Var1",y="Var2",fill="value")) 
        + geom_point(aes(size="value"),color=col_outline,shape="o")
        + guides(size=guide_legend(title=legend_title))
        + coord_flip()
        + scale_fill_gradient2(
            low = gradient_cols[0],
            high = gradient_cols[2],
            mid = gradient_cols[1],
            name = legend_title
        )
    )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x label, y label and title
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set x label
    if y_label is None:
        y_label = "Dimensions"
    # set y label
    if x_label is None:
        x_label = "Variables"
    # set title
    if title is None:
        title = "Correlation"
    # set subtitle
    if subtitle is None:
        subtitle = ""
    p = p + labs(x=x_label,y=y_label,title=title,subtitle=subtitle)

    # add plotnine theme
    p = p + pntheme
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # removing legend
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not show_legend:
        kwargs["legend_position"] = None
    p = p + theme(**kwargs)
    return p