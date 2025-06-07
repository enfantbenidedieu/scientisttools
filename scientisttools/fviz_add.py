# -*- coding: utf-8 -*-
import numpy as np
import plotnine as pn
import plotnine3d as pn3d

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
             linestyle_hline,
             size_hline,
             add_vline,
             alpha_vline,
             col_vline,
             linestyle_vline,
             size_vline,
             add_grid,
             ggtheme):
    
    """
    
    """

    pct = self.eig_.iloc[:,2]
    if x_label is None:
        x_label = "Dim."+str(axis[0]+1)+" ("+str(round(pct[axis[0]],2))+"%)"
    if y_label is None:
        y_label = "Dim."+str(axis[1]+1)+" ("+str(round(pct[axis[1]],2))+"%)"
    p = p + pn.labs(title=title,x=x_label,y=y_label)
    if x_lim is not None:
        p = p + pn.xlim(x_lim)
    if y_lim is not None:
        p = p + pn.ylim(y_lim)
    if add_hline:
        p = p + pn.geom_hline(yintercept=0,alpha=alpha_hline,color=col_hline,linetype =linestyle_hline,size=size_hline)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0,alpha=alpha_vline,color=col_vline,linetype=linestyle_vline,size=size_vline)
    if add_grid:
        p = p + pn.theme(panel_grid_major=pn.element_line(color="black",size=0.5,linetype="dashed"))
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
        return pn.geom_text(**kwargs)
    elif text_type == "label":
        return pn.geom_label(**kwargs)

def text3d_label(text_type,repel,**kwargs):
    """
    Function to choose between `geom_text_3d` and `geom_label_3d`
    ------------------------------------------------------------

    Parameters
    ----------
    text_type : {"text", "label"}, default = "text"

    **kwargs : geom parameters

    return
    ------

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if text_type not in ["text","label"]:
        raise TypeError("'text_type' should be one of 'text', 'label'")
    
    if not isinstance(repel,bool):
        raise TypeError("'repel' must be a boolean")
    
    if repel:
        kwargs = dict(kwargs,adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
    
    if text_type == "text":
        return pn3d.geom_text_3d(**kwargs)
    elif text_type == "label":
        return pn3d.geom_label_3d(**kwargs)
    
def gg_circle(r, xc, yc, color="black",fill=None,**kwargs):
    seq1 = np.linspace(0,np.pi,num=100)
    seq2 = np.linspace(0,-np.pi,num=100)
    x = xc + r*np.cos(seq1)
    ymax = yc + r*np.sin(seq1)
    ymin = yc + r*np.sin(seq2)
    return pn.annotate("ribbon", x=x, ymin=ymin, ymax=ymax, color=color, fill=fill,**kwargs)
    
# list of colors
list_colors = ["black", "red", "green", "blue", "cyan", "magenta","darkgray", "darkgoldenrod", "darkgreen", "violet",
                "turquoise", "orange", "lightpink", "lavender", "yellow","lightgreen", "lightgrey", "lightblue", "darkkhaki",
                "darkmagenta", "darkolivegreen", "lightcyan", "darkorange","darkorchid", "darkred", "darksalmon", 
                "darkseagreen","darkslateblue", "darkslategray", "darkslategrey","darkturquoise", 
                "darkviolet", "lightgray", "lightsalmon","lightyellow", "maroon"]