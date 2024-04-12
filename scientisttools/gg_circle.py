# -*- coding: utf-8 -*-
import plotnine as pn
import numpy as np

def gg_circle(r, xc, yc, color="black",fill=None,**kwargs):
    seq1 = np.linspace(0,np.pi,num=100)
    seq2 = np.linspace(0,-np.pi,num=100)
    x = xc + r*np.cos(seq1)
    ymax = yc + r*np.sin(seq1)
    ymin = yc + r*np.sin(seq2)
    return pn.annotate("ribbon", x=x, ymin=ymin, ymax=ymax, color=color, fill=fill,**kwargs)