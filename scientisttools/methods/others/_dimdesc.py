# -*- coding: utf-8 -*-
from pandas import Series,concat
from collections import OrderedDict

#intern functions
from ._condes import condes

def dimdesc(obj,
            axis=[0,1,2],
            proba=0.05):
    """
    Dimension description
    
    This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    Parameters
    ----------
    obj : class
        An object of class :class:`scientisttools.PCA`, :class:`scientisttools.CA`, :class:`scientisttools.MCA`, :class:`scientisttools.FAMD`, :class:`scientisttools.PCAmix`, :class:`scientisttools.MPCA`, :class:`scientisttools.MFA`.

    axis : int, list, tuple, default = [0,1,2]
        The dimensions to describe.

    proba : float, default = 0.05
        The significance threshold considered to characterized the dimension.

    Returns
    -------
    dimdesc : dimdescResult:
        An object with the following attributes:

        quanti : OrderedDict
            The description of the dimensions by the continuous variables. The variables are sorted.
        quali : OrderedDict
            The description of the dimensions by the categorical variables

    References
    ----------
    [1] F. Bertrand, M. Maumy-Bertrand, Initiation à la Statistique avec R, Dunod, 4ème édition, 2023.

    [2] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Example
    -------
    >>> from scientisttools.dataset import wine
    >>> from scientisttools import PCA, dimdesc
    >>> clf = PCA(sup_var=(0,1,29,30))
    >>> clf.fit(wine.data)
    >>> dimdesc(clf)
    """
    #check if model is an instance of class
    if not (obj.__class__.__name__ in ("PCA","CA","MCA","FAMD","PCAmix","MPCA","MFA")):
        raise TypeError("obj must be an object of class PCA, CA, MCA, FAMD, PCAmix, MPCA, MFA")
    
    if obj.__class__.__name__ == "CA":
        # Extract rows and columns factor coordinates
        row_coord, col_coord = obj.row_.coord, obj.col_.coord

        # Add Supplementary row
        if obj.row_sup is not None:
            row_coord = concat((row_coord,obj.row_sup_.coord),axis=0)
        
        #add supplmentary columns
        if obj.col_sup is not None:
            col_coord = concat((col_coord,obj.col_sup_.coord),axis=0)

        # Select axis
        if axis is not None:
            row_coord, col_coord = row_coord.iloc[:,axis], col_coord.iloc[:,axis]
            if isinstance(row_coord,Series):
                row_coord, col_coord = row_coord.to_frame(), col_coord.to_frame()
        
        corrdim = {}
        for idx in row_coord.columns:
            corrdim[idx] = {"row" : (row_coord[idx].to_frame().sort_values(by=idx,ascending=True).rename(columns={idx:"coord"})),
                            "col" : (col_coord[idx].to_frame().sort_values(by=idx,ascending=True).rename(columns={idx:"coord"}))}
    else:
        data = obj.call_.Xtot
        if hasattr(obj,"ind_sup_"):
            data = data.drop(index=obj.call_.ind_sup)
       
        ind_coord = obj.ind_.coord

        #select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,Series):
                ind_coord = ind_coord.to_frame()
        corrdim = OrderedDict()
        for idx in ind_coord.columns:
            X = concat((data,ind_coord[idx]),axis=1)
            corrdim[idx] = condes(X=X,num_var=idx,w=obj.call_.row_w,proba=proba)
    return corrdim