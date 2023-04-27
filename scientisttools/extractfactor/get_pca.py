# -*- coding: utf-8 -*-
##########################################################################################
#                                                                                        #
#                         PRINCIPAL COMPONENTS ANALYSIS                                  #
#                                                                                        #
##########################################################################################

import pandas as pd

def get_pca_ind(self) -> dict:

    """
    self : an instance of class PCA

    Returns
    -------
    Principal Component Analysis - Results for individuals
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
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")

    # Store informations
    df = dict({
        "coord"     :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.row_infos_,columns= ["d(i,G)","p(i)","I(i,G)"],index=self.row_labels_)
        })
    if self.row_sup_labels_ is not None:
        df["ind_sup"] = dict({
            "coord" :   pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.row_sup_cos2_,index=self.row_sup_labels_,columns=self.dim_index_)
            })    
    return df

def get_pca_var(self) -> dict:

    """
    self : an instance of class PCA

    Returns
    -------
    Principal Component Analysis - Results for variables
    ==============================================================
        Names       Description
    1   "corr"      "Pearson correlation between continuous variables"
    2   "pcorr"     "Partial correlation between continuous variables"
    3   "coord"     "Coordinates for the continuous variables"
    4   "cos2"      "Cos2 for the continuous variables"
    5   "contrib"   "Contributions of the continuous variables"
    6   "ftest"     "Fisher test of the continuous variables"
    7   "cor"       "correlations between variables and dimensions"
    """
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA")
    
    # Store informations
    df = dict({
        "corr"      :   pd.DataFrame(self.col_corr_,index=self.col_labels_,columns=self.col_labels_),
        "pcorr"     :   pd.DataFrame(self.col_pcorr_,index=self.col_labels_,columns=self.col_labels_),
        "coord"     :   pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_),
        "ftest"     :   pd.DataFrame(self.col_ftest_,index = self.col_labels_,columns=self.dim_index_),
        "cor"       :   pd.DataFrame(self.col_cor_,index=self.col_labels_,columns=self.dim_index_)
    })
    
    if ((self.quanti_sup_labels_ is not None) and (self.quali_sup_labels_ is not None)):
        # Add supplementary quantitatives informations
        df["quanti_sup"] = dict({
            "corr"  :  pd.DataFrame(self.col_sup_corr_,index=self.quanti_sup_labels_,columns=self.col_labels_), 
            "coord" :  pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
            "cos2"  :  pd.DataFrame(self.col_sup_cos2_,index=self.col_sup_labels_,columns=self.dim_index_),
            "ftest" :  pd.DataFrame(self.col_sup_ftest_,index=self.col_sup_labels_,columns=self.dim_index_)
        })
        # Add supplementary categories informations
        df["quali_sup"] = dict({
            "stats"    :   pd.DataFrame(self.mod_sup_stats_,columns=["n(k)","p(k)"],index=self.mod_sup_labels_),
            "coord"     :  pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "cos2"     :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_), 
            "dist"     :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=["dist"]),
            "eta2"     :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
            "vtest"    :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
            })
    elif self.quanti_sup_labels_ is not None:
        # Add supplementary quantitatives informations
        df["quanti_sup"] = dict({
            "corr"  :  pd.DataFrame(self.col_sup_corr_,index=self.quanti_sup_labels_,columns=self.col_labels_), 
            "coord" :  pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
            "cos2"  :  pd.DataFrame(self.col_sup_cos2_,index=self.col_sup_labels_,columns=self.dim_index_),
            "ftest" :  pd.DataFrame(self.col_sup_ftest_,index=self.col_sup_labels_,columns=self.dim_index_)
        })
    elif self.quali_sup_labels_ is not None:
        # Add supplementary categories informations
        df["quali_sup"] = dict({
            "stats"    :   pd.DataFrame(self.mod_sup_stats_,columns=["n(k)","p(k)"],index=self.mod_sup_labels_),
            "coord"     :  pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "cos2"     :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_), 
            "dist"     :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=["dist"]),
            "eta2"     :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
            "vtest"    :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
        })
    return df

def get_pca(self,choice = "row")-> dict:

    """
    self : an instance of class PCA

    choice : {"row", "var"}, default= "row"

    Returns
    -------
    if choice == "row":
        Principal Component Analysis - Results for individuals
        ===================================================
            Names       Description
        1   "coord"     "coordinates for the individuals"
        2   "cos2"      "cos2 for the individuals"
        3   "contrib"   "contributions of the individuals"
        4   "infos"     "additionnal informations for the individuals :"
                            - distance between individuals and inertia
                            - weight for the individuals
                            - inertia for the individuals
    elif choice == "var":
        Principal Component Analysis - Results for variables
        ===================================================
            Names       Description
        1   "corr"      "Pearson correlation between continuous variables"
        2   "pcorr"     "Partial correlation between continuous variables"
        3   "coord"     "Coordinates for the continuous variables"
        4   "cos2"      "Cos2 for the continuous variables"
        5   "contrib"   "Contributions of the continuous variables"
        6   "ftest"     "Fisher test of the continuous variables"
        7   "cor"       "correlations between variables and dimensions"
    """
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")
    if choice == "row":
        return get_pca_ind(self)
    elif choice == "var":
        return get_pca_var(self)
    else:
        raise ValueError("Allowed values for the argument choice are : 'row' or 'var'.")
