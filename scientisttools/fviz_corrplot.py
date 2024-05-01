# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd

from .get_melt import get_melt

def fviz_corrplot(X,
                  x_label=None,
                  y_label=None,
                  title=None,
                  outline_color = "gray",
                  colors = ["blue","white","red"],
                  legend_title = "Corr",
                  show_legend = True,
                  ggtheme = pn.theme_minimal()):
    
    if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.columns = pd.Categorical(X.columns,categories=X.columns.tolist())
    X.index = pd.Categorical(X.index,categories=X.index.tolist())
    melt = get_melt(X)

    p = (pn.ggplot(melt,pn.aes(x="Var1",y="Var2",fill="value")) + 
            pn.geom_point(pn.aes(size="value"),color=outline_color,shape="o")+pn.guides(size=None)+pn.coord_flip())

    # Adding colors
    p =p + pn.scale_fill_gradient2(low = colors[0],high = colors[2],mid = colors[1],name = legend_title) 

    # Add theme
    p = p + ggtheme

    # Add axis elements and title
    if title is None:
        title = "Correlation"
    if y_label is None:
        y_label = "Dimensions"
    if x_label is None:
        x_label = "Variables"
    p = p + pn.labs(title=title,x=x_label,y=y_label)
        
    # Removing legend
    if not show_legend:
        p =p+pn.theme(legend_position=None)
    
    return p
