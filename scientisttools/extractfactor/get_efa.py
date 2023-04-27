# -*- coding: utf-8 -*-
##########################################################################################
#                                                                                        #
#                       EXPLORATORY FACTOR ANALYSIS                                      #
#                                                                                        #
##########################################################################################

import numpy as np
import pandas as pd

def get_efa_ind(self) -> dict:

    """
    self : an instance of class EFA

    Returns
    -------
    Exploratoty Factor Analysis - Results for individuals
    ===============================================================
        Names       Description
    1   "coord"     "coordinates for the individuals"
    """
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")

    # Store informations
    df = dict({
        "coord"     :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_)
        })   
    return df

def get_efa_var(self) -> dict:

    """
    self : an instance of class EFA

    Returns
    -------
    Exploratory Factor Analysis - Results for variables
    ==============================================================
        Names           Description
    1   "coord"         "coordinates for the variables"
    2   "contrib"       "contributions of the variables"
    3   "communality"   "Communality of the variables"
    4   "variance"      "Percentage of variance"
    5   "fscore"        "Factor score"
    """
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")
    
    # Store informations
    df = dict({
        "coord"         :   pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_), 
        "contrib"       :   pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_),
        "communality"   :   pd.DataFrame(np.c_[self.initial_communality_,self.estimated_communality_],columns=["initial","estimated"],index = self.col_labels_),
        "variance"      :   pd.DataFrame(self.percentage_variance_,index=self.col_labels_,columns=["% var."]),
        "fscore"        :   pd.DataFrame(self.factor_score_,index=self.col_labels_, columns=self.dim_index_)
    })
    return df

def get_efa(self,choice = "row")-> dict:

    """
    self : an instance of class EFA

    choice : {"row", "var"}, default= "row"

    Returns
    -------
    if choice == "row":
        Exploratory Factor Analysis - Results for individuals
        ===================================================
            Names       Description
        1   "coord"     "coordinates for the individuals"
    
    if choice == "var":
        Exploratory Factor Analysis - Results for variables
        ===================================================
            Names           Description
        1   "coord"         "coordinates for the variables"
        2   "contrib"       "contributions of the variables"
        3   "communality"   "Communality of the variables"
        4   "variance"      "Percentage of variance"
        5   "fscore"        "Factor score"
    """
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")
    if choice == "row":
        return get_efa_ind(self)
    elif choice == "var":
        return get_efa_var(self)
    else:
        raise ValueError("Allowed values for the argument choice are : 'row' or 'var'.")
