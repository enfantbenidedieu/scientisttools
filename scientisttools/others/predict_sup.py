# -*- coding: utf-8 -*-
from numpy import ones, dot
from mapply.mapply import mapply
from collections import OrderedDict

#----------------------------------------------------------------------------------------------------------------------------------------
##predict supplementary individuals
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_ind_sup(X,V,col_weights,n_workers):
    # Factor coordinates
    coord = mapply(X,lambda x : x*col_weights,axis=1,progressbar=False,n_workers=n_workers).dot(V)
    coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]

    # Square distance to origin
    sqdisto = mapply(X,lambda  x : (x**2)*col_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."

    # Square cosine
    cos2 = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)
    #convert to dict
    return OrderedDict(coord=coord,cos2=cos2,dist=sqdisto)

#----------------------------------------------------------------------------------------------------------------------------------------
##predict supplementary quantitative variables
#----------------------------------------------------------------------------------------------------------------------------------------
def predict_quanti_sup(X,U,row_weights,n_workers):
    # Factor coordinates
    coord = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(U)
    coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]

    # Square distance to origin
    sqdisto = dot(ones(X.shape[0]),mapply(X,lambda x : (x**2)*row_weights,axis=0,progressbar=False,n_workers=n_workers))

    # Square cosine
    cos2 = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)  
    
    return OrderedDict(coord=coord,cos2=cos2)