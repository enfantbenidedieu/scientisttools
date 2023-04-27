# -*- coding: utf-8 -*-

import pandas as pd

def get_ca_row(self)-> dict:

    """
    self. : an instance of class CA

    Returns
    -------
    Correspondence Analysis - Results for rows
    =========================================================
        Name        Description
    1   "coord"     "coordinates for the rows"
    2   "cos2"      "cos2 for the rows"
    3   "constrib"  "contributions of the rows"
    4   "dist"      "Rows distance"
    5   "res.dist"  "Restitued distance"
    6   "infos"     "additionnal informations for the rows:"
                        - distance between rows and inertia
                        - weight for the rows
                        - inertia for the rows
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an instance of class CA.")
    df = dict({"coord"      :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_), 
               "cos2"       :   pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
               "contrib"    :   pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
               "dist"       :   pd.DataFrame(self.row_dist_,index=self.row_labels_,columns=self.row_labels_),
               "res.dist"  :   pd.DataFrame(self.res_row_dist_,index=self.row_labels_,columns=self.row_labels_),
               "infos"      :   pd.DataFrame(self.row_infos_,columns= ["d(i,G)","p(i)","I(i,G)"],index=self.row_labels_)
            })
    if self.row_sup_labels_ is not None:
        df["row_sup"] = dict({
            "coord" : self.row_sup_coord_})
    
    return df

def get_ca_col(self)-> dict:

    """
    self : an instance of class CA

    Returns
    -------
    Correspondence Analysis - Results for columns
    =========================================================
        Name        Description
    1   "coord"     "coordinates for the columns"
    2   "cos2"      "cos2 for the columns"
    3   "constrib"  "contributions of the columns"
    4   "dist"      "Columns distance"
    5   "res.dist"  "Restitued distance"
    6   "infos"     "additionnal informations for the columns :"
                        - distance between columns and inertia
                        - weight for the columns
                        - inertia for the columns
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    df = dict({"coord"      :   pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_), 
               "cos2"       :   pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_),
               "contrib"    :   pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_),
               "dist"       :   pd.DataFrame(self.col_dist_,index=self.col_labels_,columns=self.col_labels_),
               "res.dist"   :   pd.DataFrame(self.res_col_dist_,index=self.col_labels_,columns=self.col_labels_),
               "infos"      :   pd.DataFrame(self.col_infos_,columns= ["d(k,G)","p(k)","I(k,G)"],index=self.col_labels_)
               })
    if self.col_sup_labels_ is not None:
        df["col_sup"] = dict({
            "coord" : self.col_sup_coord_
            })
    
    return df

def get_ca(self,choice = "row")-> dict:

    """
    self : an instance of class CA

    choice : {"row", "col"}, default= "row"

    Returns
    -------
    if choice == "row":
        Correspondence Analysis - Results for rows
        =========================================================
            Name        Description
        1   "coord"     "coordinates for the rows"
        2   "cos2"      "cos2 for the rows"
        3   "constrib"  "contributions of the rows"
        4   "dist"      "Rows distance"
        5   "res.dist"  "Restitued distance"
        6   "infos"     "additionnal informations for the rows:"
                            - distance between rows and inertia
                            - weight for the rows
                            - inertia for the rows
    if choice == "col":
        Correspondence Analysis - Results for columns
        =========================================================
            Name        Description
        1   "coord"     "coordinates for the columns"
        2   "cos2"      "cos2 for the columns"
        3   "constrib"  "contributions of the columns"
        4   "dist"      "Columns distance"
        5   "res.dist"  "Restitued distance"
        6   "infos"     "additionnal informations for the columns :"
                            - distance between columns and inertia
                            - weight for the columns
                            - inertia for the columns
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    if choice == "row":
        return get_ca_row(self)
    elif choice == "col":
        return get_ca_col(self)
    else:
        raise ValueError("Error : Allowed values for the argument choice are : 'row' or 'col'.")
