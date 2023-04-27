# -*- coding: utf-8 -*-
##########################################################################################
#                                                                                        #
#                 PARTIAL PRINCIPAL COMPONENTS ANALYSIS                                  #
#                                                                                        #
##########################################################################################

import pandas as pd

def get_ppca_ind(self) -> dict:

    """
    self : an instance of class PPCA

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
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an object of class PPCA.")

    # Store informations
    df = dict({
        "coord"     :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.row_infos_,columns= ["d(i,G)","p(i)","I(i,G)"],index=self.row_labels_)
        })   
    return df

def get_ppca_var(self) -> dict:

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
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an object of class PPCA")
    
    # Store informations
    df = dict({
        "coord"     :   pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_),
        "cor"       :   pd.DataFrame(self.col_cor_,index=self.col_labels_,columns=self.dim_index_)
    })
    return df

def get_ppca(self,choice = "row")-> dict:

    """
    self : an instance of class PPCA

    choice : {"row", "var"}, default= "row"

    Returns
    -------
    if choice == "row":
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
    if self.model_ != "ppca":
        raise ValueError("Error : 'self' must be an object of class PPCA.")
    if choice == "row":
        return get_ppca_ind(self)
    elif choice == "var":
        return get_ppca_var(self)
    else:
        raise ValueError("Allowed values for the argument choice are : 'row' or 'var'.")
