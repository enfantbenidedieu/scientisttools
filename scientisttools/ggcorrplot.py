# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotnine as pn
import scipy.stats as st
import plydata as ply
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scientisttools.extractfactor import get_melt

def hc_cormat_order(cormat, method='complete'):
    if not isinstance(cormat,pd.DataFrame):
        raise ValueError("Error : 'cormat' must be a DataFrame.")
    X = (1-cormat)/2
    Z = hierarchy.linkage(squareform(X),method=method, metric="euclidean")
    order = hierarchy.leaves_list(Z)
    return dict({"order":order,"height":Z[:,2],"method":method,
                "merge":Z[:,:2],"n_obs":Z[:,3],"data":cormat})

def match_arg(x, lst):
    return [el for el in lst if x in el][0]

def no_panel():
    return pn.theme(
        axis_title_x=pn.element_blank(),
        axis_title_y=pn.element_blank()
    )

def remove_diag(cormat):
    if cormat is None:
        return cormat
    if not isinstance(cormat,pd.DataFrame):
        raise ValueError("Error : 'cormat' must be a DataFrame.")
    np.fill_diagonal(cormat.values, np.nan)
    return cormat

def get_upper_tri(cormat,show_diag=False):
    if cormat is None:
        return cormat
    if not isinstance(cormat,pd.DataFrame):
        raise ValueError("Error : 'cormat' must be a DataFrame.")
    cormat = pd.DataFrame(np.triu(cormat),index=cormat.index,columns=cormat.columns)
    cormat.values[np.tril_indices(cormat.shape[0], -1)] = np.nan
    if not show_diag:
        np.fill_diagonal(cormat.values,np.nan)
    return cormat

def get_lower_tri(cormat,show_diag=False):
    if cormat is None:
        return cormat
    if not isinstance(cormat,pd.DataFrame):
        raise ValueError("Error : 'cormat' must be a DataFrame.")
    cormat = pd.DataFrame(np.tril(cormat),index=cormat.index,columns=cormat.columns)
    cormat.values[np.triu_indices(cormat.shape[0], 1)] = np.nan
    if not show_diag:
        np.fill_diagonal(cormat.values,np.nan)
    return cormat

def cor_pmat(x,**kwargs):
    if not isinstance(x,pd.DataFrame):
        raise ValueError("Error : 'x' must be a DataFrame.")
    y = np.array(x)
    n = y.shape[1]
    p_mat = np.zeros((n,n))
    np.fill_diagonal(p_mat,0)
    for i in np.arange(0,n-1):
        for j in np.arange(i+1,n):
            tmps = st.pearsonr(y[:,i],y[:,j],**kwargs)
            p_mat[i,j] = p_mat[j,i] = tmps[1]
    p_mat = pd.DataFrame(p_mat,index=x.columns,columns=x.columns)
    return p_mat

def ggcorrplot(x,
               method = "square",
               type = "full",
               ggtheme = pn.theme_minimal(),
               title = None,
               show_legend = True,
               legend_title = "Corr",
               show_diag = None,
               colors = ["blue","white","red"],
               outline_color = "gray",
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
    
    if not isinstance(x,pd.DataFrame):
        raise ValueError("Error : 'x' must be a DataFrame.")
    
    if p_mat is not None:
        if not isinstance(p_mat,pd.DataFrame):
            raise ValueError("Error : 'p_mat' must be a DataFrame.")

    type = match_arg(type, ["full","lower","upper"])
    method = match_arg(method,["square",'circle'])
    insig = match_arg(insig,["pch","blank"])

    if show_diag is None:
        if type == "full":
            show_diag = True
        else:
            show_diag = False

    corr = x.corr().round(decimals=digits)

    if hc_order:
        ord = hc_cormat_order(corr,method=hc_method)["order"]
        corr = corr.iloc[ord,ord]
        if p_mat is not None:
            p_mat = p_mat.iloc[ord,ord]
            p_mat = p_mat.round(decimals=digits)

    if not show_diag:
        corr = remove_diag(corr)
        if p_mat is not None:
            p_mat = remove_diag(p_mat)
    
    # Get lower or upper triangle
    if type == "lower":
        corr = get_lower_tri(corr,show_diag)
        if p_mat is not None:
            p_mat = get_lower_tri(p_mat,show_diag)
    elif type == "upper":
        corr = get_upper_tri(corr,show_diag)
        if p_mat is not None:
            p_mat = get_upper_tri(corr,show_diag)
    
    # Melt corr and p_mat
    corr.columns = pd.Categorical(corr.columns,categories=corr.columns)
    corr.index = pd.Categorical(corr.columns,categories=corr.columns)
    corr = get_melt(corr)
    
    corr = corr >> ply.define(pvalue=np.nan)
    corr = corr >> ply.define(signif=np.nan)

    if p_mat is not None:
        p_mat = get_melt(p_mat)
        corr = corr >> ply.define(coef="value")
        corr = corr >> ply.mutate(pvalue=p_mat.value)
        corr["signif"] = np.where(p_mat.value <= sig_level,1,0)
        p_mat = p_mat.query(f'value > {sig_level}')
        if insig == "blank":
            corr = corr >> ply.mutate(value="value*signif")
    
    corr = corr >> ply.define(abs_corr="abs(value)*10")

    p = pn.ggplot(corr,pn.aes(x="Var1",y="Var2",fill="value"))
    
    # Modification based on method
    if method == "square":
        p = p + pn.geom_tile(color=outline_color)
    elif method == "circle":
        p = p+pn.geom_point(pn.aes(size="abs_corr"),
                            color=outline_color,
                            shape="o")+pn.scale_size_continuous(range=(4,10))+pn.guides(size=None)
    
    # Adding colors
    p =p + pn.scale_fill_gradient2(
        low = colors[0],
        high = colors[2],
        mid = colors[1],
        midpoint = 0,
        limits = [-1,1],
        name = legend_title
    )

    # depending on the class of the object, add the specified theme
    p = p + ggtheme

    p =p+pn.theme(
        axis_text_x=pn.element_text(angle=tl_srt,
                                    va="center",
                                    size=tl_cex,
                                    ha="center",
                                    color=tl_col),
        axis_text_y=pn.element_text(size=tl_cex)
    ) + pn.coord_fixed()

    label = corr["value"].round(digits)

    if p_mat is not None and insig == "blank":
        ns = corr["pvalue"] > sig_level
        if sum(ns) > 0:
            label[ns] = " "
    
    # matrix cell labels
    if lab:
        p = p + pn.geom_text(mapping=pn.aes(x="Var1",y="Var2"),
                             label = label,
                             color=lab_col,
                             size=lab_size)
    
    # matrix cell 
    if p_mat is not None and insig == "pch":
        p = p + pn.geom_point(data = p_mat,
                              mapping = pn.aes(x = "Var1",y = "Var2"),
                              shape = pch,
                              size=pch_cex,
                              color= pch_col)
    
    if title is not None:
        p = p + pn.ggtitle(title=title)
    
    # Removing legend
    if not show_legend:
        p =p+pn.theme(legend_position=None)
    
    # Removing panel
    p = p + no_panel()

    return p




    
    
    
    






