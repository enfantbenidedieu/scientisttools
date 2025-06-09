# -*- coding: utf-8 -*-
from numpy import linspace, sin, cos, pi
from plotnine import labs, xlim,ylim,geom_hline,geom_vline,theme,element_line,geom_text,geom_label,annotate

def fviz_add(p,
             self,
             axis,
             x_label,
             y_label,
             title,
             x_lim,
             y_lim,
             add_hline,
             alpha_hline,
             col_hline,
             linetype_hline,
             size_hline,
             add_vline,
             alpha_vline,
             col_vline,
             linetype_vline,
             size_vline,
             add_grid,
             ggtheme):
    
    """
    Add eleme
    
    """

    pct = self.eig_.iloc[:,2]
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(pct[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(pct[axis[1]],2))+"%)"
    p = p + labs(title=title,x=x_label,y=y_label)
    if x_lim is not None:
        p = p + xlim(x_lim)
    if y_lim is not None:
        p = p + ylim(y_lim)
    if add_hline:
        p = p + geom_hline(yintercept=0,alpha=alpha_hline,color=col_hline,linetype =linetype_hline,size=size_hline)
    if add_vline:
        p = p+ geom_vline(xintercept=0,alpha=alpha_vline,color=col_vline,linetype=linetype_vline,size=size_vline)
    if add_grid:
        p = p + theme(panel_grid_major=element_line(color="black",size=0.5,linetype="dashed"))
    p = p + ggtheme
    return p

def text_label(text_type,repel,**kwargs):
    """
    Function to choose between `geom_text` and `geom_label`
    -------------------------------------------------------

    Parameters
    ----------
    `text_type` : {"text", "label"}, default = "text".

    `repel` : a boolean, whether to avoid overplotting text labels or not (by default == False)

    `**kwargs` : geom parameters

    return
    ------
    a plotnine geom

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    """
    if text_type not in ["text","label"]:
        raise TypeError("'text_type' should be one of 'text', 'label'")
    
    if not isinstance(repel,bool):
        raise TypeError("'repel' must be a boolean")
    
    if repel:
        kwargs = dict(kwargs,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
    
    if text_type == "text":
        return geom_text(**kwargs)
    elif text_type == "label":
        return geom_label(**kwargs)
    
def gg_circle(r, xc, yc, color="black",fill=None,**kwargs):
    x = xc + r*cos(linspace(0,pi,num=100))
    ymin, ymax = yc + r*sin(linspace(0,-pi,num=100)), yc + r*sin(linspace(0,pi,num=100))
    return annotate("ribbon", x=x, ymin=ymin, ymax=ymax, color=color, fill=fill,**kwargs)
    
# list of colors
list_colors = ["black", "red", "green", "blue", "cyan", "magenta","darkgray", "darkgoldenrod", "darkgreen", "violet",
                "turquoise", "orange", "lightpink", "lavender", "yellow","lightgreen", "lightgrey", "lightblue", "darkkhaki",
                "darkmagenta", "darkolivegreen", "lightcyan", "darkorange","darkorchid", "darkred", "darksalmon", 
                "darkseagreen","darkslateblue", "darkslategray", "darkslategrey","darkturquoise", 
                "darkviolet", "lightgray", "lightsalmon","lightyellow", "maroon"]