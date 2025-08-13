# -*- coding: utf-8 -*-
from plotnine import ggplot, arrow, theme_minimal

#intern functions
from .fviz import add_arrow, set_axis, fviz_circle

def fviz_corrcircle(self,
                    axis = [0,1],
                    geom = ("arrow","text"),
                    repel = False,
                    col_var = "black",
                    segment_args_var = dict(linetype="solid",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                    text_args_var = dict(size=8),
                    col_quanti_sup = "blue",
                    segment_args_quanti_sup = dict(linetype="dashed",size=0.5,arrow = arrow(angle=10,length=0.1,type="closed")),
                    text_args_quanti_sup = dict(size=8),
                    scale = 1,
                    add_circle = True,
                    col_circle = "gray",
                    x_lim = (-1.1,1.1),
                    y_lim = (-1.1,1.1),
                    x_label = None,
                    y_label = None,
                    title = None,
                    add_hline = True,
                    add_vline = True,
                    add_grid = True,
                    ggtheme = theme_minimal()):
    """
    Draw correlation circle
    -----------------------

    Description
    -----------

    Usage
    -----
    ```python
    >>> fviz_corrcircle(self,**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCA, PartiaLPCA, CA, MCA, FAMD

    `**kwargs`: additionals informations. For more see `fviz_arrow`.

    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import decathlon, PCA, fviz_corrcircle
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotate=None)
    >>> res_pca.fit(decathlon)
    >>> #graph of variables
    >>> print(fviz_corrcircle(res_pca))

    """
    if self.model_ not in ["pca","partialpca","fa","ca","mca","specificmca","famd","mpca","pcamix","mfa","mfaqual","mfamix"]:
        raise TypeError("'self' must be an object of class PCA, PartialPCA, FactorAnalysis, CA, MCA, SpecificMCA, FAMD, MPCA, PCAMIX, MFA, MFAQUAL, MFAMIX")
    
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > self.call_.n_components-1)  or (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    if self.model_ in ["pca","partialpca","fa"]:
        coord = self.var_.coord
    elif self.model_ in ["famd","mpca","pcamix","mfa","mfamix"]:
        coord = self.quanti_var_.coord
    else:
        if hasattr(self, "quanti_sup_"):
            coord = self.quanti_sup_.coord
        if hasattr(self, "quanti_var_sup_"):
            coord = self.quanti_var_sup_.coord

    #initialize
    p = add_arrow(p=ggplot(),data=coord.mul(scale),axis=axis,geom=geom,repel=repel,color=col_var,segment_args=segment_args_var,text_args=text_args_var)
            
    if self.model_ in ["pca","famd","mpca","pcamix","mfa","mfamix"]:
        if hasattr(self, "quanti_sup_"):
            sup_coord = self.quanti_sup_.coord
        elif hasattr(self, "quanti_var_sup_"):
            sup_coord = self.quanti_var_sup_.coord
        if hasattr(self, "quanti_sup_") or hasattr(self, "quanti_var_sup_"):
            p = add_arrow(p=p,data=sup_coord.mul(scale),axis=axis,geom=geom,repel=repel,color=col_quanti_sup,segment_args=segment_args_quanti_sup,text_args=text_args_quanti_sup)
    
    #add correlation circle
    if add_circle:
        p = fviz_circle(p=p,color=col_circle)

    #add others elements
    if title is None:
        title = "Correlation circle"
    p = set_axis(p=p,self=self,axis=axis,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)
      
    return p