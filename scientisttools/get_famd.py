# -*- coding: utf-8 -*-
##########################################################################################
#                                                                                        #
#                        FACTOR ANALYSIS OF MIXED DATA                                   #
#                                                                                        #
##########################################################################################

import pandas as pd

def get_famd_ind(self) -> dict:
    """Extract individuals informations

    Parameters
    ----------
    self : an instance of class FAMD

    Returns
    -------
    Factor Analysis of Mixed Data - Results for individuals
    =======================================================
        Names       Description
    1   "coord"     "Coordinates for the individuals"
    2   "cos2"      "Cos2 for the individuals"
    3   "contrib"   "Contributions of the individuals"
    4   "infos"     "Additionnal informations for the individuals :"
                        - distance between individuals and inertia
                        - weight for the individuals
                        - inertia for the individuals
    """
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")

    # Store informations
    df = dict({
        "coord"     :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.row_infos_,columns= ["d(i,G)","p(i)","I(i,G)"],index=self.row_labels_)
        })
    if self.row_sup_labels_ is not None:
        df["ind_sup"] = dict({
            "coord" :   pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_)
            })
    return df

def get_famd_col(self) -> dict:
    """Extract continuous variables informations

    Parameters
    ----------
    self : an instance of class FAMD

    Returns
    -------
    Factor Analysis of Mixed Data - Results for continuous variables
    ================================================================
        Names       Description
    1   "corr"      "Pearson correlation between continuous variables"
    2   "pcorr"     "Partial correlation between continuous variables"
    3   "coord"     "Coordinates for the continuous variables"
    4   "cos2"      "Cos2 for the continuous variables"
    5   "contrib"   "Contributions of the continuous variables"
    6   "ftest"     "Fisher test of the continuous variables"
    """
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD")
    
    # Store informations
    df = dict({
        "corr"      :   pd.DataFrame(self.col_corr_,index=self.col_labels_,columns=self.col_labels_),
        "pcorr"      :   pd.DataFrame(self.col_pcorr_,index=self.col_labels_,columns=self.col_labels_),
        "coord"     :   pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_),
        "ftest"     :   pd.DataFrame(self.col_ftest_,index = self.col_labels_,columns=self.dim_index_)
    })
    if self.quanti_sup_labels_ is not None:
        # Add supplementary continuous variables informations
        df["quanti_sup"] = dict({
            "corr"  :   pd.DataFrame(self.col_sup_corr_,index=self.col_sup_labels_,columns=self.col_labels_),
            "coord" :   pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.col_sup_cos2_,index=self.col_sup_labels_,columns=self.dim_index_),
            "ftest" :   pd.DataFrame(self.col_sup_ftest_,index=self.col_sup_labels_,columns=self.dim_index_)
        })
    return df

def get_famd_mod(self) -> dict:
    """Extract categories informations

    Parameters
    ----------
    self  : an instance of class FAMD

    Returns
    -------
    Factor Analysis of Mixed Data - Results for categories
    ======================================================
        Names       Description
    1   "stats"     "Count and percentage of categories"
    2   "coord"     "coordinates for the categories"
    3   "cos2"      "cos2 for the categories"
    4   "contrib"   "contributions of the categories"
    5   "vtest"     "value test of the categories"
    6   "infos"     "additionnal informations for the categories :"
                        - distance between categories and inertia
                        - weight for the categories
                        - inertia for the categories
    """
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")

    # Store informations
    df = dict({
        "stats"     :   pd.DataFrame(self.mod_stats_,columns=["n(k)","p(k)"],index=self.mod_labels_),
        "coord"     :   pd.DataFrame(self.mod_coord_,index=self.mod_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.mod_cos2_,index=self.mod_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.mod_contrib_,index=self.mod_labels_,columns=self.dim_index_),
        "vtest"     :   pd.DataFrame(self.mod_vtest_,index=self.mod_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.mod_infos_,columns= ["d(k,G)","p(k)","I(k,G)"],index=self.mod_labels_)
        })
    if self.quali_sup_labels_ is not None:
        df["quali_sup"] = dict({
            "stats" :   pd.DataFrame(self.mod_sup_stats_,columns=["n(k)","p(k)"],index=self.mod_sup_labels_),
            "coord" :   pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "dist"  :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "vtest" :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
        })
    return df

def get_famd_var(self):
    """Extract categorical variables informations

    Parameters
    ----------
    self  : an instance of class FAMD

    Returns
    -------
    Factor Analysis of Mixed Data - Results for categorical variables
    =================================================================
        Names       Description
    1   "chi2"     "chi-squared statistics and p-values"
    2   "eta2"      "Correlation ratio"
    3   "cos2"      "cos2 for categorical variables"
    4   "contrib"   "contributions of categorical variables"
    """

    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")

    df = dict({
        "chi2"      :   self.chi2_test_,
        "eta2"      :   pd.DataFrame(self.var_eta2_,index=self.quali_labels_,columns=self.dim_index_),
        "cos2"      :   pd.DataFrame(self.var_cos2_,index=self.quali_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.var_contrib_,index=self.quali_labels_,columns=self.dim_index_)
    })
    if self.quali_sup_labels_ is not None:
        df["quali_sup"] = dict({
            "chi2"  :   self.chi2_sup_stats_,
            "eta2"  :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
        })
    return df


def get_famd(self,choice = "ind")-> dict:
    """Extract Factor Analysis oif Mixed Data informations

    Parameters
    ----------
    self : an instance of class FAMD

    choice : {"ind","var","mod","col"}, default= "ind"

    Returns
    -------
    if choice == "ind":
        Factor Analysis of Mixed Data - Results for individuals
        ===================================================
            Names       Description
        1   "coord"     "Coordinates for the individuals"
        2   "cos2"      "Cos2 for the individuals"
        3   "contrib"   "Contributions of the individuals"
        4   "infos"     "Additionnal informations for the individuals :"
                            - distance between individuals and inertia
                            - weight for the individuals
                            - inertia for the individuals
    
    if choice == "col":
        Factor Analysis of Mixed Data - Results for continuous variables
        ==================================================================
            Names       Description
        1   "corr"      "Pearson correlation between continuous variables"
        2   "pcorr"     "Partial correlation between continuous variables"
        3   "coord"     "Coordinates for the continuous variables"
        4   "cos2"      "Cos2 for the continuous variables"
        5   "contrib"   "Contributions of the continuous variables"
        6   "ftest"     "Fisher test of the continuous variables"
    if choice == "mod":
        Factor Analysis of Mixed Data - Results for modality of qualitatives variables
        ===============================================================================
            Names       Description
        1   "stats"     "Count and percentage of categories"
        2   "coord"     "coordinates for the categories"
        3   "cos2"      "cos2 for the categories"
        4   "contrib"   "contributions of the categories"
        5   "vtest"     "value test of the categories"
        6   "infos"     "additionnal informations for the categories :"
                            - distance between categories and inertia
                            - weight for the categories
                            - inertia for the categories
    if choice == "var"
        Factor Analysis of Mixed Data - Results for variables
        =====================================================
            Names       Description
        1   "chi2"     "chi-squared statistics and p-values"
        2   "eta2"      "Correlation ratio"
        3   "cos2"      "cos2 for categorical variables"
        4   "contrib"   "contributions of categorical variables"
    """
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    if choice == "ind":
        return get_famd_ind(self)
    elif choice == "col":
        return get_famd_col(self)
    elif choice == "mod":
        return get_famd_mod(self)
    elif choice == "var":
        return get_famd_var(self)
    else:
        raise ValueError("Allowed values for the argument choice are : 'ind','var','mod' and 'col'.")
