# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_partialpca_ind(self) -> dict:

    """
    self : an object of class PPCA

    Returns
    -------
    Partial Principal Component Analysis - Results for individuals
    ===============================================================
        Names       Description
    1   "coord"     "coordinates for the individuals"
    2   "cos2"      "cos2 for the individuals"
    3   "contrib"   "contributions of the individuals"
    4   "infos"     "additionnal informations for the individuals :"
                        - distance between individuals and inertia
                        - weight for the individuals
                        - inertia for the individuals
    """
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA.")
    return self.ind_

def get_partialpca_var(self) -> dict:

    """
    self : an instance of class PPCA

    Returns
    -------
    Partial Principal Component Analysis - Results for variables
    ==============================================================
        Names       Description
    1   "coord"     "coordinates for the variables"
    2   "cos2"      "cos2 for the variables"
    3   "contrib"   "contributions of the variables"
    4   "cor"       "correlations between variables and dimensions"
    """
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    
    # Store informations
    return self.var_

def get_partialpca(self,choice = "ind")-> dict:

    """
    self : an object of class PPCA

    choice : {"ind", "var"}, default= "ind"

    Returns
    -------
    if choice == "ind":
        Partial Principal Component Analysis - Results for individuals
        ===================================================
            Names       Description
        1   "coord"     "coordinates for the individuals"
        2   "cos2"      "cos2 for the individuals"
        3   "contrib"   "contributions of the individuals"
        4   "infos"     "additionnal informations for the individuals :"
                            - distance between individuals and inertia
                            - weight for the individuals
                            - inertia for the individuals
    
    if choice == "var":
        Partial rincipal Component Analysis - Results for variables
        ===================================================
            Names       Description
        1   "coord"     "coordinates for the variables"
        2   "cos2"      "cos2 for the variables"
        3   "contrib"   "contributions of the variables"
        4   "cor"       "correlations between variables and dimensions"
    """
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    if choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'ind', 'var'")
    
    if choice == "row":
        return get_partialpca_ind(self)
    elif choice == "var":
        return get_partialpca_var(self)

########## Partial PCA

def summaryPartialPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Partial Principal Component Analysis model
    ----------------------------------------------------------------

    Parameters
    ----------
    self        :   an obect of class PartialPCA

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["resid"].shape[0])

    # Partial Principal Components Analysis Results
    print("                     Partial Principal Component Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in np.arange(0,ncp,1):
        ind_coord = ind["coord"].iloc[:,i]
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_ctr = ind["contrib"].iloc[:,i]
        ind_ctr.name = "ctr"
        ind_infos = pd.concat([ind_infos,ind_coord,ind_ctr,ind_cos2],axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add variables informations
    print(f"\nContinues variables\n")
    var = self.var_
    var_infos = var["infos"]
    for i in np.arange(0,ncp,1):
        var_coord = var["coord"].iloc[:,i]
        var_cos2 = var["cos2"].iloc[:,i]
        var_cos2.name = "cos2"
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr"
        var_infos = pd.concat([var_infos,var_coord,var_ctr,var_cos2],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)