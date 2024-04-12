# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd
from ggcorrplot import ggcorrplot, get_melt

def fviz_corrplot(X,
                  method = "square",
                  type = "full",
                  x_label=None,
                  y_label=None,
                  title=None,
                  outline_color = "gray",
                  colors = ["blue","white","red"],
                  legend_title = "Corr",
                  is_corr = False,
                  show_legend = True,
                  ggtheme = pn.theme_minimal(),
                  show_diag = None,
                  hc_order = False,
                  hc_method = "complete",
                  lab = False,
                  lab_col = "black",
                  lab_size = 11,
                  p_mat = None,
                  sig_level=0.05,
                  insig = "pch",
                  pch = 4,
                  pch_col = "black",
                  pch_cex = 5,
                  tl_cex = 12,
                  tl_col = "black",
                  tl_srt = 45,
                  digits = 2):
    
    if not isinstance(X,pd.DataFrame):
        raise ValueError("'X' must be a DataFrame.")
    
    if is_corr:
        p = ggcorrplot(x=X,
                       method=method,
                       type=type,
                       ggtheme = ggtheme,
                       title = title,
                       show_legend = show_legend,
                       legend_title = legend_title,
                       show_diag = show_diag,
                       colors = colors,
                       outline_color = outline_color,
                       hc_order = hc_order,
                       hc_method = hc_method,
                       lab = lab,
                       lab_col = lab_col,
                       lab_size = lab_size,
                       p_mat = p_mat,
                       sig_level=sig_level,
                       insig = insig,
                       pch = pch,
                       pch_col = pch_col,
                       pch_cex = pch_cex,
                       tl_cex = tl_cex,
                       tl_col = tl_col,
                        tl_srt = tl_srt,
                        digits = digits)
    else:
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
