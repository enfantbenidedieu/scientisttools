# -*- coding: utf-8 -*-

import plotnine as pn
import numpy as np
import pandas as pd
from scientisttools.extractfactor import get_eigenvalue

def fviz_screeplot(self,choice="proportion",geom_type=["bar","line"],bar_fill = "steelblue",
                   bar_color="steelblue",line_color="black",line_type="solid",bar_width=None,
                   add_kaiser=False,add_kss = False, add_broken_stick = False,n_components=10,
                   add_labels=False,h_align= "center",v_align = "bottom",title=None,x_label=None,
                   y_label=None,ggtheme=pn.theme_minimal(),**kwargs):
        
    if self.model_ not in ["pca","ca","mca","famd","mfa","cmds"]:
        raise ValueError("'res' must be an object of class PCA, CA, MCA, FAMD, MFA, CMDS")

    eig = get_eigenvalue(self)
    eig = eig.iloc[:min(n_components,self.n_components_),:]

    if choice == "eigenvalue":
        eig = eig["eigenvalue"]
        text_labels = list([str(np.around(x,3)) for x in eig.values])
        if self.model_ != "mds":
            kaiser = self.kaiser_threshold_
        if y_label is None:
            y_label = "Eigenvalue"
    elif choice == "proportion":
        eig = eig["proportion"]
        text_labels = list([str(np.around(x,1))+"%" for x in eig.values])
        if self.model_ !=  "pca":
            kaiser = self.kaiser_proportion_threshold_
    else:
        raise ValueError("Allowed values for the argument choice are : 'proportion' or 'eigenvalue'")
    
    df_eig = pd.DataFrame({"dim" : pd.Categorical(np.arange(1,len(eig)+1)),"eig" : eig.values})
    
    p = pn.ggplot(df_eig,pn.aes(x = "dim",y="eig",group = 1))
    if "bar" in geom_type :
        p = p   +   pn.geom_bar(stat="identity",fill=bar_fill,color=bar_color,width=bar_width)
    if "line" in geom_type :
        p = (p  +   pn.geom_line(color=line_color,linetype=line_type)+\
                    pn.geom_point(shape="o",color=line_color))
    if add_labels:
        p = p + pn.geom_text(label=text_labels,ha = h_align,va = v_align)
    if add_kaiser :
        p = (p +  pn.geom_hline(yintercept=kaiser,linetype="--", color="red")+\
                  pn.annotate("text", x=int(np.median(np.arange(1,len(eig)+1))), y=kaiser, 
                              label="Kaiser threshold"))

    if add_kss:
        if self.model_ in ["pca","ppca"]:
            if choice == "eigenvalue":
                p = (p  +   pn.geom_hline(yintercept=self.kss_threshold_,linetype="--", color="yellow")+ \
                            pn.annotate("text", x=int(np.mean(np.arange(1,len(eig)+1))), y=self.kss_threshold_, 
                                        label="Karlis - Saporta - Spinaki threshold",colour = "yellow"))
            else:
                raise ValueError("'add_kss' is only with 'choice=eigenvalue'.")
        else:
            raise ValueError("'add_kss' is only for class PCA.")
    if add_broken_stick:
        if choice == "eigenvalue":
            if self.model_ == ["pca","ppca"]:
                bst = self.broken_stick_threshold_[:min(n_components,self.n_components_)]
                p = (p  +   pn.geom_line(pn.aes(x="dim",y=bst),color="green",linetype="--")+\
                            pn.geom_point(pn.aes(x="dim",y=bst),colour="green")+\
                            pn.annotate("text", x=int(np.mean(np.arange(1,len(eig)+1))), y=np.median(bst), 
                                        label="Broken stick threshold",colour = "green"))
        else:
            raise ValueError("'add_broken_stick' is only for class PCA.")

    if title is None:
        title = "Scree plot"
    if x_label is None:
        x_label = "Dimensions"
    if y_label is None:
        y_label = "Percentage of explained variances"
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    p = p + ggtheme
    return p

def fviz_eigenvalue(self,**kwargs):
    return fviz_screeplot(self,**kwargs)

def fviz_eig(self,**kwargs):
    return fviz_screeplot(self,**kwargs)