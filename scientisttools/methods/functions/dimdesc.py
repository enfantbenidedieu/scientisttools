# -*- coding: utf-8 -*-
from pandas import Series,concat, DataFrame

#intern functions
from .catdesc import catdesc
from .contdesc import contdesc
from .splitmix import splitmix

def dimdesc(self,axis=None,proba=0.05):
    """
    Dimension description
    ----------------------

    Description
    -----------
    This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    Usage
    -----
    ```python
    >>> dimdesc(self,axis=None,proba=0.05)
    ```

    Parameters
    ----------
    `self` : an object of class PCA, CA, MCA, SpecificMCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX, MFACT

    `axis` : an integer or a list/tuple specifying the axis (by default = None)

    `axis` : the significance threshold considered to characterized the dimension (by default 0.05)

    Returns
    -------
    dictionary of dataframes including :

    `quanti` :  the description of the dimensions by the quantitative variables. The variables are sorted.
    
    `quali`	: the description of the dimensions by the categorical variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import 
    >>> gironde = load_gironde()
    >>> X_quanti = splitmix(X=gironde).quanti
    >>> X_
    ```

    References
    ----------
    F. Bertrand, M. Maumy-Bertrand, Initiation à la Statistique avec R, Dunod, 4ème édition, 2023.

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    """
    def desc(x,y,weights,proba):
        # Split data
        x_quanti, x_quali = splitmix(x).quanti, splitmix(x).quali

        res = {}
        if x_quanti is not None:
            quanti = contdesc(x=x_quanti,y=y,weights=weights,proba=proba)
            if quanti.shape[0]!=0:
                res["quanti"] = quanti
        
        if x_quali is not None:
            cat_desc =  catdesc(data=x_quali,coord=y,proba=proba)
            if cat_desc.quali.shape[0] != 0 :
                res["quali"] = cat_desc.quali
            if cat_desc.category.shape[0] != 0:
                res["category"] = cat_desc.category
        return res

    # Check if model is an instance of class PCA, CA, MCA, SpecificMCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX, MFACT
    if self.model_ not in ["pca","ca","mca","specificmca","famd","pcamix","mpca","mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("`self` must be an instance of class PCA, CA, MCA, SpecificMCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX, MFACT")
    
    if self.model_ == "ca":
        # Extract rows and columns factor coordinates
        row_coord, col_coord = self.row_.coord, self.col_.coord

        # Add Supplementary row
        if self.row_sup is not None:
            row_coord = concat([row_coord,self.row_sup_.coord],axis=0)
        
        # Add supplmentary columns
        if self.col_sup is not None:
            col_coord = concat([col_coord,self.col_sup_.coord],axis=0)

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
        if self.model_ != "mfact":
            data = self.call_.Xtot
            if self.ind_sup is not None:
                data = data.drop(index=self.call_.ind_sup)
        else:
            data = self.global_pca_.call_.Z

        ind_coord, ind_weights = self.ind_.coord, self.call_.ind_weights

        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,Series):
                ind_coord = ind_coord.to_frame()
        corrdim = {}
        for idx in ind_coord.columns:
            corrdim[idx] = desc(x=data,y=ind_coord[idx],weights=ind_weights,proba=proba)
    return corrdim