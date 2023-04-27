# -*- coding: utf-8 -*-

import pandas as pd

def get_mca_ind(self) -> dict:
    """
    self : an instance of class MCA

    Returns
    -------
    Multiple Correspondence Analysis - Results for individuals
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
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")

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

def get_mca_mod(self) -> dict:

    """
    self : an instance of class MCA

    Returns
    -------
    Multiple Correspondence Analysis - Results for categories
    =====================================================================
        Names               Description
    1   "coord"             "coordinates for the categories"
    2   "corrected_coord"   "Coorected coordinates for the categories"
    3   "cos2"              "cos2 for the categories"
    4   "contrib"           "contributions of the categories"
    5   "infos"             "additionnal informations for the categories :"
                                - distance between categories and inertia
                                - weight for the categories
                                - inertia for the categories
    """
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")

    # Store informations
    df = dict({
        "coord"             :   pd.DataFrame(self.mod_coord_,index=self.mod_labels_,columns=self.dim_index_), 
        "corrected_coord"   :   pd.DataFrame(self.corrected_mod_coord_,index=self.mod_labels_,columns=self.dim_index_),
        "cos2"              :   pd.DataFrame(self.mod_cos2_,index=self.mod_labels_,columns=self.dim_index_),
        "contrib"           :   pd.DataFrame(self.mod_contrib_,index=self.mod_labels_,columns=self.dim_index_),
        "infos"             :   pd.DataFrame(self.mod_infos_,columns= ["d(k,G)","p(k)","I(k,G)"],index=self.mod_labels_)
        })
    if self.quali_sup_labels_ is not None:
        df["sup"] = dict({
            "stats"     :   pd.DataFrame(self.mod_sup_stats_, index = self.mod_sup_labels_,columns = ["n(k)","p(k)"]),
            "coord"     :   pd.DataFrame(self.mod_sup_coord_, index =self.mod_sup_labels_,columns=self.dim_index_),
            "cos2"      :   pd.DataFrame(self.mod_sup_cos2_,  index =self.mod_sup_labels_,columns=self.dim_index_),
            "dist"      :   pd.DataFrame(self.mod_sup_disto_, index = self.mod_sup_labels_,columns=["Dist"]),
            "vtest"     :   pd.DataFrame(self.mod_sup_vtest_, index =self.mod_sup_coord_,columns=self.dim_index_)
            })    
    return df

def get_mca_var(self) -> dict:
    """
    self : an instance of class MCA

    Returns
    -------
    Multiple Correspondence Analysis - Results for categories variables
    =====================================================================
        Names           Description
    1   "chi2"          "chi-squared tests and p-values"
    2   "inertia"       "Categories variables inertia"
    3   "eta2"          "Correlation ratio"
    4   "cos2"          "cosines of the categories variables"
    5   "contrib"       "contributions of the categories variables"
    """

    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")

    df = dict({
        "chi2"      :   self.chi2_test_,
        "inertia"   :   pd.DataFrame(self.var_inertia_,index=self.var_labels_,columns=["I(j,G)"]),
        "eta2"      :   pd.DataFrame(self.var_eta2_,index=self.var_labels_,columns=self.dim_index_),
        "cos2"      :   pd.DataFrame(self.var_cos2_,index=self.var_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.var_contrib_,index=self.var_labels_,columns=self.dim_index_)
    })
    if ((self.quanti_sup_labels_ is not None) & (self.quali_sup_labels_ is not None)):
        df["quanti_sup"] = dict({
            "coord" :   pd.DataFrame(self.quanti_sup_coord_,index=self.quanti_sup_labels_,columns=self.dim_index_)
        })
        df["quali_sup"] = dict({
            "eta2" :    pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
        })
    elif self.quanti_sup_labels_ is not None:
        df["quanti_sup"] = dict({
            "coord" :   pd.DataFrame(self.quanti_sup_coord_,index=self.quanti_sup_labels_,columns=self.dim_index_)
        })
    elif self.quali_sup_labels_ is not None:
        df["quali_sup"] = dict({
            "eta2" :    pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
        })

    return df

def get_mca(self,choice="ind") -> dict:
    """

    Parameters
    ---------
    self    : an instance of class MCA
    choice  : {'ind','mod','var'}

    if choice == "ind":
        -------
        Multiple Correspondence Analysis - Results for individuals
        ===============================================================
            Names       Description
        1   "coord"     "coordinates for the individuals"
        2   "cos2"      "cos2 for the individuals"
        3   "contrib"   "contributions of the individuals"
        4   "infos"     "additionnal informations for the individuals :"
                            - distance between individuals and inertia
                            - weight for the individuals
                            - inertia for the individuals
    elif choice == "mod":
         Multiple Correspondence Analysis - Results for categories
        =====================================================================
            Names               Description
        1   "coord"             "coordinates for the categories"
        2   "corrected_coord"   "Coorected coordinates for the categories"
        3   "cos2"              "cos2 for the categories"
        4   "contrib"           "contributions of the categories"
        5   "infos"             "additionnal informations for the categories :"
                                    - distance between categories and inertia
                                    - weight for the categories
                                    - inertia for the categories
    elif choice == "var":
        Multiple Correspondence Analysis - Results for categories variables
        =====================================================================
            Names           Description
        1   "chi2"          "chi-squared tests and p-values"
        2   "inertia"       "Categories variables inertia"
        3   "eta2"          "Correlation ratio"
        4   "cos2"          "cosines of the categories variables"
        5   "contrib"       "contributions of the categories variables"
    """
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if choice == "ind":
        return get_mca_ind(self)
    elif choice == "mod":
        return get_mca_mod(self)
    elif choice == "var":
        return get_mca_var(self)
    else:
        raise ValueError("Error : Allowed values for the argument 'choice' are : 'ind','var' and 'mod'.")
