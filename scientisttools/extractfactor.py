# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy
from scientisttools.utils import eta2
import scipy.stats as st

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
            "coord" : pd.DataFrame(self.row_sup_coord_,columns=self.dim_index_,index=self.row_sup_labels_)
            })
    
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
            "coord" : pd.DataFrame(self.col_sup_coord_,columns=self.dim_index_,index=self.col_sup_labels_)
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

# -*- coding: utf-8 -*-

def StandardScaler(X):
    return (X - X.mean())/X.std(ddof=0)

def get_dist(X, method = "euclidean",normalize=False,**kwargs) -> dict:
    if isinstance(X,pd.DataFrame) is False:
        raise ValueError("Error : 'X' must be a DataFrame")
    if normalize:
        X = X.transform(StandardScaler)
    if method in ["pearson","spearman","kendall"]:
        corr = X.T.corr(method=method)
        dist = corr.apply(lambda cor :  1 - cor,axis=0).values.flatten('F')
    else:
        dist = pdist(X.values,metric=method,**kwargs)
    return dict({"dist" :dist,"labels":X.index})


################### Exploratory factor analysis

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

################## Eigenvalues

def get_eig(self) -> pd.DataFrame:

    """
    self : an instance of class PCA, PartialPCA, CA, MCA, FAMD, MFA,CMDS

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent
    """
    if self.model_ in ["pca","ppca","ca","mca","famd","mfa","cmds","candisc"]:
        eig = pd.DataFrame(self.eig_.T,columns=["eigenvalue","difference","proportion","cumulative"],index = self.dim_index_)
        return eig
    else:
        raise ValueError("Error : 'self' must be an instance of class PCA, PPCA, CA, MCA, FAMD, MFA, CMDS")

def get_eigenvalue(self) -> pd.DataFrame:

    """
    self : an instance of class PCA, PartialPCA, CA, MCA, FAMD, MFA, MDS

    Returns
    -------
    eigenvalue, variance percent and cumulative variance of percent
    """
    return get_eig(self)


############ Factor analysis of mixed data

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
    df = {
        "coord"     :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.row_infos_,columns= ["d(i,G)","p(i)","I(i,G)"],index=self.row_labels_)
        }
    if self.row_sup_labels_ is not None:
        df["ind_sup"] = {
            "dist"  :   pd.DataFrame(self.row_sup_disto_,index = self.row_sup_labels_,columns=["Dist"]),
            "coord" :   pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.row_sup_cos2_,index=self.row_sup_labels_,columns=self.dim_index_)
            }
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
    df = {
        "corr"      :   pd.DataFrame(self.col_corr_,index=self.col_labels_,columns=self.col_labels_),
        "pcorr"      :   pd.DataFrame(self.col_pcorr_,index=self.col_labels_,columns=self.col_labels_),
        "coord"     :   pd.DataFrame(self.col_coord_,index = self.col_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.col_cos2_,index = self.col_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.col_contrib_,index = self.col_labels_,columns=self.dim_index_),
        "ftest"     :   pd.DataFrame(self.col_ftest_,index = self.col_labels_,columns=self.dim_index_)
    }
    if self.quanti_sup_labels_ is not None:
        # Add supplementary continuous variables informations
        df["quanti_sup"] = {
            "corr"  :   pd.DataFrame(self.col_sup_corr_,index=self.col_sup_labels_,columns=self.col_labels_),
            "coord" :   pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.col_sup_cos2_,index=self.col_sup_labels_,columns=self.dim_index_),
            "ftest" :   pd.DataFrame(self.col_sup_ftest_,index=self.col_sup_labels_,columns=self.dim_index_)
        }
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
    df = {
        "stats"     :   pd.DataFrame(self.mod_stats_,columns=["n(k)","p(k)"],index=self.mod_labels_),
        "coord"     :   pd.DataFrame(self.mod_coord_,index=self.mod_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.mod_cos2_,index=self.mod_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.mod_contrib_,index=self.mod_labels_,columns=self.dim_index_),
        "vtest"     :   pd.DataFrame(self.mod_vtest_,index=self.mod_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.mod_infos_,columns= ["d(k,G)","p(k)","I(k,G)"],index=self.mod_labels_)
        }
    if self.quali_sup_labels_ is not None:
        df["quali_sup"] = {
            "stats" :   pd.DataFrame(self.mod_sup_stats_,columns=["n(k)","p(k)"],index=self.mod_sup_labels_),
            "coord" :   pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_),
            "dist"  :   pd.DataFrame(np.c_[self.mod_sup_disto_],index=self.mod_sup_labels_,columns=["dist"]),
            "vtest" :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
        }
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

    df = {
        "chi2"      :   self.chi2_test_,
        "eta2"      :   pd.DataFrame(self.var_eta2_,index=self.quali_labels_,columns=self.dim_index_),
        "cos2"      :   pd.DataFrame(self.var_cos2_,index=self.quali_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.var_contrib_,index=self.quali_labels_,columns=self.dim_index_)
    }
    if self.quali_sup_labels_ is not None:
        df["quali_sup"] = {
            "chi2"  :   self.chi2_sup_test_,
            "eta2"  :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.quali_sup_cos2_,index=self.quali_sup_labels_,columns=self.dim_index_)
        }
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


############# Hierarchical 

def get_hclust(X, method='single', metric='euclidean', optimal_ordering=False):
    Z = hierarchy.linkage(X,method=method, metric=metric)
    if optimal_ordering:
        order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z,X))
    else:
        order = hierarchy.leaves_list(Z)
    return dict({"order":order,"height":Z[:,2],"method":method,
                "merge":Z[:,:2],"n_obs":Z[:,3],"data":X})


########## Multiple Correspondence Analysis

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
    df = {
        "coord"     :   pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_), 
        "cos2"      :   pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
        "infos"     :   pd.DataFrame(self.row_infos_,columns= ["d(i,G)","p(i)","I(i,G)"],index=self.row_labels_)
        }
    if self.row_sup_labels_ is not None:
        df["ind_sup"] = {
            "coord" :   pd.DataFrame(self.row_sup_coord_,index=self.row_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.row_sup_cos2_,index=self.row_sup_labels_,columns=self.dim_index_)
            }   
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
    df = {
        "coord"             :   pd.DataFrame(self.mod_coord_,index=self.mod_labels_,columns=self.dim_index_), 
        "corrected_coord"   :   pd.DataFrame(self.corrected_mod_coord_,index=self.mod_labels_,columns=self.dim_index_),
        "cos2"              :   pd.DataFrame(self.mod_cos2_,index=self.mod_labels_,columns=self.dim_index_),
        "contrib"           :   pd.DataFrame(self.mod_contrib_,index=self.mod_labels_,columns=self.dim_index_),
        "vtest"             :   pd.DataFrame(self.mod_vtest_,index = self.mod_labels_,columns=self.dim_index_),
        "infos"             :   pd.DataFrame(self.mod_infos_,columns= ["d(k,G)","p(k)","I(k,G)"],index=self.mod_labels_)
        }
    if self.quali_sup_labels_ is not None:
        df["sup"] = {
            "stats"     :   pd.DataFrame(self.mod_sup_stats_, index = self.mod_sup_labels_,columns = ["n(k)","p(k)"]),
            "coord"     :   pd.DataFrame(self.mod_sup_coord_, index =self.mod_sup_labels_,columns=self.dim_index_),
            "cos2"      :   pd.DataFrame(self.mod_sup_cos2_,  index =self.mod_sup_labels_,columns=self.dim_index_),
            "dist"      :   pd.DataFrame(self.mod_sup_disto_, index = self.mod_sup_labels_,columns=["Dist"]),
            "vtest"     :   pd.DataFrame(self.mod_sup_vtest_, index =self.mod_sup_labels_,columns=self.dim_index_)
            }
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

    df = {
        "chi2"      :   self.chi2_test_,
        "inertia"   :   pd.DataFrame(self.var_inertia_,index=self.var_labels_,columns=["I(j,G)"]),
        "eta2"      :   pd.DataFrame(self.var_eta2_,index=self.var_labels_,columns=self.dim_index_),
        "cos2"      :   pd.DataFrame(self.var_cos2_,index=self.var_labels_,columns=self.dim_index_),
        "contrib"   :   pd.DataFrame(self.var_contrib_,index=self.var_labels_,columns=self.dim_index_)
    }

    if self.quanti_sup_labels_ is not None:
        df["quanti_sup"] = {
            "coord" :   pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
            "cos2"  :   pd.DataFrame(self.col_sup_cos2_,index=self.col_sup_labels_,columns=self.dim_index_)
        }
    if self.quali_sup_labels_ is not None:
        df["quali_sup"] = {
            "eta2"  :    self.quali_sup_eta2_,
            "cos2"  :    self.quali_sup_cos2_ 
        }

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
    
################## MDS

def get_mds(self) -> dict:

    """
    self : an object of class MDS

    Returns
    -------
    Multidimensional Scaling - Results 
    ===============================================================
        Names       Description
    1   "coord"     "coordinates"
    2   "res.dist"  "Restitues distances"
    """
    if self.model_ not in ["mds","cmds"]:
        raise ValueError("Error : 'res' must be an object of class MDS or CMDS.")

    # Store informations
    df = dict({
        "coord"     : pd.DataFrame(self.coord_,index=self.labels_,columns=self.dim_index_),
        "res.dist"  : pd.DataFrame(self.res_dist_,index=self.labels_,columns=self.labels_)
    })
    return df

############ Principal Components Analysis

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
            "cos2"  :   pd.DataFrame(self.row_sup_cos2_,index=self.row_sup_labels_,columns=self.dim_index_),
            "dist"  :   pd.DataFrame(np.c_[self.row_sup_disto_],index=self.row_sup_labels_,columns=["d(i,G)"])
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

########## Partial Principal Components Analysis
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


################## Summarize functions

def summaryCA(self,
              digits=3,
              nb_element=10,
              ncp=3,
              to_markdown=False,
              tablefmt="pipe",
              **kwargs):
    """Printing summaries of correspondence analysis model

    Parameters
    ----------
    self        :   an obect of class CA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_ca(self,choice="row")
    col = get_ca(self,choice="col")

    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Principal Components Analysis Results
    print("                     Correspondence Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nRows\n")
    row_infos = row["infos"]
    for i in np.arange(0,ncp,1):
        row_coord = row["coord"].iloc[:,i]
        row_cos2 = row["cos2"].iloc[:,i]
        row_cos2.name = "cos2"
        row_ctr = row["contrib"].iloc[:,i]
        row_ctr.name = "ctr"
        row_infos = pd.concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)

    # Add supplementary individuals
    if self.row_sup_labels_ is not None:
        print(f"\nSupplementary rows\n")
        # Save all informations
        row_sup_coord = row["row_sup"]["coord"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(row_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_coord)

    # Add variables informations
    print(f"\nColumns\n")
    col_infos = col["infos"]
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_cos2 = col["cos2"].iloc[:,i]
        col_cos2.name = "cos2"
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary columns informations
    if self.col_sup_labels_ is not None:
        print(f"\nSupplementary columns\n")
        col_sup_coord = col["col_sup"]["coord"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(col_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_coord)


def summaryEFA(self,
               digits=3,
               nb_element=10,
               ncp=3,
               to_markdown=False,
               tablefmt = "pipe",
               **kwargs):
    """Printing summaries of exploratory factor analysis model

    Parameters
    ----------
    self        :   an obect of class EFA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_efa(self,choice="row")
    col = get_efa(self,choice="var")


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_),len(self.col_labels_))

    # Exploratory Factor Analysis Results
    print("                     Exploratory Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first) \n")
    row_coord = row["coord"].iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(row_coord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_coord)

    # Add supplementary individuals
    if self.row_sup_labels_ is not None:
        nb_elt = min(nb_element,len(self.row_sup_labels_))
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        row_sup_infos = pd.DataFrame(index=self.row_sup_labels_).astype("float")
        row_sup = row["ind_sup"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2 = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.iloc[:nb_elt,:].round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    # Add variables informations
    print(f"\nContinues Variables\n")
    col_infos = pd.DataFrame(index=self.col_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr],axis=1)
    col_infos = col_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup_labels_ is not None:
        print(f"\nSupplementary continuous variable\n")
        col_sup_infos = pd.DataFrame(index=self.quanti_sup_labels_).astype("float")
        col_sup = col["quanti_sup"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos =pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.round(decimals=digits)

        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup_labels_ is not None:
        print("\nSupplementary categories\n")
        mod_sup = col["quali_sup"]
        mod_sup_infos = np.sqrt(mod_sup["dist"])
        for i in np.arange(0,ncp,1):
            mod_sup_coord = mod_sup["coord"].iloc[:,i]
            mod_sup_cos2 = mod_sup["cos2"].iloc[:,i]
            mod_sup_cos2.name = "cos2"
            mod_sup_vtest = mod_sup["vtest"].iloc[:,i]
            mod_sup_vtest.name = "v.test"
            mod_sup_infos = pd.concat([mod_sup_infos,mod_sup_coord,mod_sup_cos2,mod_sup_vtest],axis=1)
        mod_sup_infos = mod_sup_infos.round(decimals=digits)

        if to_markdown:
            print(mod_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(mod_sup_infos)
        
        # Add supplementary qualitatives - correlation ration
        print("\nSupplementatry categorical variable\n")
        corr_ratio = mod_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(corr_ratio.to_markdown(tablefmt=tablefmt))
        else:
            print(corr_ratio)

###### FAMD

def summaryFAMD(self,
               digits=3,
               nb_element=10,
               ncp=3,
               to_markdown=False,
               tablefmt = "pipe",
               **kwargs):
    """Printing summaries of factor analysis of miixed data model

    Parameters
    ----------
    self        :   an obect of class FAMD.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_famd_ind(self)
    mod = get_famd_mod(self)
    var = get_famd_var(self)
    col = get_famd_col(self)

    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Principal Components Analysis Results
    print("                     Factor Analysis of Mixed Data - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["infos"]
    for i in np.arange(0,ncp,1):
        row_coord = row["coord"].iloc[:,i]
        row_cos2 = row["cos2"].iloc[:,i]
        row_cos2.name = "cos2"
        row_ctr = row["contrib"].iloc[:,i]
        row_ctr.name = "ctr"
        row_infos = pd.concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)

    # Add supplementary individuals
    if self.row_sup_labels_ is not None:
        print(f"\nSupplementary individuals\n")
        row_sup = row["ind_sup"]
        row_sup_infos = row_sup["dist"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2 = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    # Add variables informations
    print(f"\nContinuous variables\n")
    col_infos = pd.DataFrame(index=self.col_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_cos2 = col["cos2"].iloc[:,i]
        col_cos2.name = "cos2"
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup_labels_ is not None:
        print(f"\nSupplementary continuous variable\n")
        col_sup_infos = pd.DataFrame(index=self.col_sup_labels_).astype("float")
        col_sup = col["quanti_sup"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos =pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.round(decimals=digits)

        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
    # Add variables informations
    print(f"\nCategories\n")
    mod_infos = mod["infos"]
    for i in np.arange(0,ncp,1):
        mod_coord = mod["coord"].iloc[:,i]
        mod_cos2 = mod["cos2"].iloc[:,i]
        mod_cos2.name = "cos2"
        mod_ctr = mod["contrib"].iloc[:,i]
        mod_ctr.name = "ctr"
        mod_vtest = mod["vtest"].iloc[:,i]
        mod_vtest.name = "vtest"
        mod_infos = pd.concat([mod_infos,mod_coord,mod_ctr,mod_cos2,mod_vtest],axis=1)
    mod_infos = mod_infos.round(decimals=digits)
    if to_markdown:
        print(mod_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(mod_infos)
    
    # Add variables
    print("\nCategorical variables\n")
    var_infos = pd.DataFrame(index=self.quali_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        var_eta2 = var["eta2"].iloc[:,i]
        var_eta2.name = "Dim."+str(i+1)
        var_contrib = var["contrib"].iloc[:,i]
        var_contrib.name = "ctr"
        var_cos2 = var["cos2"].iloc[:,i]
        var_cos2.name = "cos2"
        var_infos = pd.concat([var_infos,var_eta2,var_contrib,var_cos2],axis=1)
    var_infos = var_infos.round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup_labels_ is not None:
        print("\nSupplementary categories\n")
        mod_sup = mod["quali_sup"]
        mod_sup_infos = np.sqrt(mod_sup["dist"])
        for i in np.arange(0,ncp,1):
            mod_sup_coord = mod_sup["coord"].iloc[:,i]
            mod_sup_cos2 = mod_sup["cos2"].iloc[:,i]
            mod_sup_cos2.name = "cos2"
            mod_sup_vtest = mod_sup["vtest"].iloc[:,i]
            mod_sup_vtest.name = "v.test"
            mod_sup_infos = pd.concat([mod_sup_infos,mod_sup_coord,mod_sup_cos2,mod_sup_vtest],axis=1)
        mod_sup_infos = mod_sup_infos.round(decimals=digits)

        if to_markdown:
            print(mod_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(mod_sup_infos)
        
        # Add supplementary qualitatives - correlation ration
        print("\nSupplementary categorical variable\n")
        var_sup = var["quali_sup"]
        var_sup_infos = pd.DataFrame().astype("float")
        for i in np.arange(0,ncp,1):
            var_sup_eta2 = var_sup["eta2"].iloc[:,i]
            var_sup_eta2.name = "Dim."+str(i+1)
            var_sup_cos2 = var_sup["cos2"].iloc[:,i]
            var_sup_cos2.name = "cos2"
            var_sup_infos =  pd.concat([var_sup_infos,var_sup_eta2,var_sup_cos2],axis=1)
        var_sup_infos = var_sup_infos.round(decimals=digits)
        if to_markdown:
            print(var_sup_infos.to_markdown(tablefmt=tablefmt))
        else:
            print(var_sup_infos)

########" MCA"

def summaryMCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """Printing summaries of multiple correspondence analysis model
    Parameters
    ----------
    self        :   an obect of class MCA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_mca(self,choice="ind")
    mod = get_mca(self,choice="mod")
    var = get_mca(self,choice="var")


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_),len(self.mod_labels_))

    # Multiple correspondance Analysis - Results
    print("                     Multiple Correspondance Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["infos"]
    for i in np.arange(0,ncp,1):
        row_coord = row["coord"].iloc[:,i]
        row_cos2 = row["cos2"].iloc[:,i]
        row_cos2.name = "cos2"
        row_ctr = row["contrib"].iloc[:,i]
        row_ctr.name = "ctr"
        row_infos = pd.concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)

    # Add supplementary individuals
    if self.row_sup_labels_ is not None:
        nb_elt = min(nb_element,len(self.row_sup_labels_))
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        row_sup_infos = pd.DataFrame(index=self.row_sup_labels_).astype("float")
        row_sup = row["ind_sup"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2 = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.iloc[:nb_elt,:].round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    # Add variables informations
    print(f"\nCategories (the {nb_element} first)\n")
    mod_infos = mod["infos"]
    for i in np.arange(0,ncp,1):
        mod_coord = mod["coord"].iloc[:,i]
        mod_cos2 = mod["cos2"].iloc[:,i]
        mod_cos2.name = "cos2"
        mod_ctr = mod["contrib"].iloc[:,i]
        mod_ctr.name = "ctr"
        mod_vtest = mod["vtest"].iloc[:,i]
        mod_vtest.name = "vtest"
        mod_infos = pd.concat([mod_infos,mod_coord,mod_ctr,mod_cos2,mod_vtest],axis=1)
    mod_infos = mod_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(mod_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(mod_infos)
    
    # Add variables
    print("\nCategorical variables\n")
    var_infos = var["inertia"]
    for i in np.arange(0,ncp,1):
        var_eta2 = var["eta2"].iloc[:,i]
        var_eta2.name = "Dim."+str(i+1)
        var_cos2 = var["cos2"].iloc[:,i]
        var_cos2.name = "cos2"
        var_contrib = var["contrib"].iloc[:,i]
        var_contrib.name = "ctr"
        var_infos = pd.concat([var_infos,var_eta2,var_cos2,var_contrib],axis=1)
    
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)

    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup_labels_ is not None:
        print("\nSupplementary categories\n")
        mod_sup = mod["sup"]
        mod_sup_infos = np.sqrt(mod_sup["dist"])
        for i in np.arange(0,ncp,1):
            mod_sup_coord = mod_sup["coord"].iloc[:,i]
            mod_sup_cos2 = mod_sup["cos2"].iloc[:,i]
            mod_sup_cos2.name = "cos2"
            mod_sup_vtest = mod_sup["vtest"].iloc[:,i]
            mod_sup_vtest.name = "v.test"
            mod_sup_infos = pd.concat([mod_sup_infos,mod_sup_coord,mod_sup_cos2,mod_sup_vtest],axis=1)
        mod_sup_infos = mod_sup_infos.round(decimals=digits)

        if to_markdown:
            print(mod_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(mod_sup_infos)
        
        print("\nSupplementary categorical variables\n")
        var_sup_infos = pd.DataFrame().astype("float")
        var_sup = var['quali_sup']
        for i in np.arange(0,ncp,1):
            var_sup_eta2 = var_sup["eta2"].iloc[:,i]
            var_sup_eta2.name = "Dim."+str(i+1)
            var_sup_cos2 = var_sup["cos2"].iloc[:,i]
            var_sup_cos2.name = "cos2"
            var_sup_infos = pd.concat([var_sup_infos,var_sup_eta2,var_sup_cos2],axis=1)
        
        var_sup_infos = var_sup_infos.round(decimals=digits)
        if to_markdown:
            print(var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(var_sup_infos)

    # Add supplementary continuous variables informations
    if self.quanti_sup_labels_ is not None:
        print(f"\nSupplementary continuous variable\n")
        col_sup_infos = pd.DataFrame().astype("float")
        col_sup= var["quanti_sup"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos = pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        
        col_sup_infos = col_sup_infos.round(decimals=digits)
        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
###### PCA

def summaryPCA(self,
               digits=3,
               nb_element=10,
               ncp=3,
               to_markdown=False,
               tablefmt = "pipe",
               **kwargs):
    """Printing summaries of principal component analysis model

    Parameters
    ----------
    self        :   an obect of class PCA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_pca(self,choice="row")
    col = get_pca(self,choice="var")


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Principal Components Analysis Results
    print("                     Principal Component Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["infos"]
    for i in np.arange(0,ncp,1):
        row_coord = row["coord"].iloc[:,i]
        row_cos2 = row["cos2"].iloc[:,i]
        row_cos2.name = "cos2"
        row_ctr = row["contrib"].iloc[:,i]
        row_ctr.name = "ctr"
        row_infos = pd.concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)

    # Add supplementary individuals
    if self.row_sup_labels_ is not None:
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        row_sup = row["ind_sup"]
        row_sup_infos = row_sup["dist"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2 = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    # Add variables informations
    print(f"\nContinues variables\n")
    col_infos = pd.DataFrame(index=self.col_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_cos2 = col["cos2"].iloc[:,i]
        col_cos2.name = "cos2"
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup_labels_ is not None:
        print(f"\nSupplementary continuous variable\n")
        col_sup_infos = pd.DataFrame(index=self.quanti_sup_labels_).astype("float")
        col_sup = col["quanti_sup"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos =pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.round(decimals=digits)

        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup_labels_ is not None:
        print("\nSupplementary categories\n")
        mod_sup = col["quali_sup"]
        mod_sup_infos = mod_sup["dist"]
        for i in np.arange(0,ncp,1):
            mod_sup_coord = mod_sup["coord"].iloc[:,i]
            mod_sup_cos2 = mod_sup["cos2"].iloc[:,i]
            mod_sup_cos2.name = "cos2"
            mod_sup_vtest = mod_sup["vtest"].iloc[:,i]
            mod_sup_vtest.name = "v.test"
            mod_sup_infos = pd.concat([mod_sup_infos,mod_sup_coord,mod_sup_cos2,mod_sup_vtest],axis=1)
        mod_sup_infos = mod_sup_infos.round(decimals=digits)

        if to_markdown:
            print(mod_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(mod_sup_infos)
        
        # Add supplementary qualitatives - correlation ration
        print("\nSupplementatry categorical variable (eta2)\n")
        corr_ratio = mod_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(corr_ratio.to_markdown(tablefmt=tablefmt))
        else:
            print(corr_ratio)

########### Partial PCA

def summaryPPCA(self,
                digits=3,
                nb_element=10,
                ncp=3,
                to_markdown=False,
                tablefmt = "pipe",
                **kwargs):
    """Printing summaries of partial principal component analysis model

    Parameters
    ----------
    self        :   an obect of class PPCA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_ppca(self,choice="row")
    col = get_ppca(self,choice="var")


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Partial Principal Components Analysis Results
    print("                     Partial Principal Component Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["infos"]
    for i in np.arange(0,ncp,1):
        row_coord = row["coord"].iloc[:,i]
        row_cos2 = row["cos2"].iloc[:,i]
        row_cos2.name = "cos2"
        row_ctr = row["contrib"].iloc[:,i]
        row_ctr.name = "ctr"
        row_infos = pd.concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)
    
    # Add variables informations
    print(f"\nContinues variables\n")
    col_infos = pd.DataFrame(index=self.col_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_cos2 = col["cos2"].iloc[:,i]
        col_cos2.name = "cos2"
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)


###############################################################################################
#       Canonical Discriminant Analysis (CANDISC)
###############################################################################################


def get_candisc_row(self):
    """
    self. : an instance of class CANDISC

    Returns
    -------
    Canonical Discriminant Analysis Analysis - Results for rows
    =========================================================
        Name        Description
    1   "coord"     "coordinates for the rows"
    """
    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class CANDISC.")

    df = dict({
        "coord" : pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_)
    })
    return df


def get_candisc_var(self,choice="correlation"):
    """
    self. : an instance of class CANDISC

    Returns
    -------
    Canonical Discriminant Analysis Analysis - Results for rows
    =========================================================
        Name        Description
    """

    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class CANDISC.")
    
    if choice == "correlation":
        df = dict({
            "Total" : self.tcorr_,
            "Between" : self.bcorr_,
            "Within" : self.wcorr_
        })
    elif choice == "covariance":
        df = dict({
            "Total" : self.tcov_,
            "Between" : self.bcov_,
            "Within" : self.wcov_
        })

    return df

def get_candisc(self,choice = "row"):
    """
    self. : an instance of class CANDISC

    Returns
    -------
    Canonical Discriminant Analysis Analysis - Results for rows
    =========================================================
        Name        Description
    1   "coord"     "coordinates for the rows"
    """

    if choice == "row":
        return get_candisc_row(self)
    elif choice == "var":
        return get_candisc_var(self)
    else:
        raise ValueError("Error : Allowed values are either 'row' or 'var'.")
   
def get_candisc_coef(self,choice="absolute"):
    """
    
    """
    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class CANDISC.")
    
    if choice == "absolute":
        coef = pd.concat([pd.DataFrame(self.coef_,index=self.features_labels_,columns=self.dim_index_),
                         pd.DataFrame(self.intercept_.T,columns=["Intercept"],index=self.dim_index_).T],axis=0)
    elif choice == "score":
        coef = pd.concat([pd.DataFrame(self.score_coef_,index=self.features_labels_,columns=self.classes_),
                          pd.DataFrame(self.score_intercept_,index=["Intercept"],columns=self.classes_)],axis=0)
    return coef


def summaryCANDISC(self,digits=3,
                   nb_element=10,
                   ncp=3,
                   to_markdown=False,
                   tablefmt = "pipe",
                   **kwargs):
    """Printing summaries of Canonical Discriminant Analysis model

    Parameters
    ----------
    self        :   an obect of class CANDISC.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """
    
    row = get_candisc(self,choice="row")
    var = get_candisc(self,choice="var")
    coef = get_candisc_coef(self,choice="absolute").round(decimals=digits)
    score_coef = get_candisc_coef(self,choice="score").round(decimals=digits)
    gmean = self.gmean_.round(decimals=digits)


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Partial Principal Components Analysis Results
    print("                     Canonical Discriminant Analysis - Results                     \n")

    print("\nSummary Information")
    summary = self.summary_information_.T
    if to_markdown:
        print(summary.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(summary)
    
    print("\nClass Level information")
    class_level_infos = self.class_level_information_
    if to_markdown:
        print(class_level_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(class_level_infos)

    # Add eigenvalues informations
    print("\nImportance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    print("\nGroup means:")
    if to_markdown:
        print(gmean.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(gmean)
    
    print("\nCoefficients of canonical discriminants:")
    if to_markdown:
        print(coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(coef)
    
    print("\nClassification functions coefficients:")
    if to_markdown:
        print(score_coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(score_coef)

    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["coord"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)
    
    # Add variables informations
    print(f"\nContinues variables\n")
    var_infos = pd.DataFrame(index=self.features_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        tcorr = var["Total"].iloc[:,i]
        tcorr.name ="total."+str(i+1)
        bcorr = var["Between"].iloc[:,i]
        bcorr.name ="between."+str(i+1)
        wcorr = var["Within"].iloc[:,i]
        wcorr.name ="within."+str(i+1)
        var_infos = pd.concat([var_infos,tcorr,bcorr,wcorr],axis=1)
    var_infos = var_infos.round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)


#############################################################################################
#           Multiple Factor Analysis (MFA)
#############################################################################################

def get_mfa_ind(self):
    """
    
    
    """
    df = dict({
        "coord" : pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_),
        "contrib" : pd.DataFrame(self.row_contrib_,index=self.row_labels_,columns=self.dim_index_),
        "cos2" : pd.DataFrame(self.row_cos2_,index=self.row_labels_,columns=self.dim_index_),
        "coord_partial" : None,
        "within_inertia" : None,
        "within_partial_inertia" : None
    })
    
    return df

def get_mfa_var(self):
    raise ValueError("Error : This method is not yet implemented.")


def get_mfa(self,choice="row"):
    """
    
    """
    if choice == "row":
        return get_mfa_ind(self)
    elif choice == "col":
        return get_mfa_var(self)
    else:
        raise ValueError("Error : Allowed values are 'row' and 'var'.")
    

def summaryMFA(self,
               digits=3,
               nb_element=10,
               ncp=3,
               to_markdown=False,
               tablefmt = "pipe",
               **kwargs):
    """Printing summaries of Multiple Factor Analysis model

    Parameters
    ----------
    self        :   an obect of class MFA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_mfa(self,choice="row")

    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Principal Components Analysis Results
    print("                     Multiple Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["coord"].iloc[:,:ncp]
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)


###################################################""
#       Description of axis
#############################################################
def dimdesc(self,axis=None,proba=0.05):
    """
    Description of axis

    Parameters
    ----------
    self : an instace of class PCA, CA, MCA, FAMD
    axis : int or list. default axis= 0
    proba : critical probabilitie

    Return
    ------
    corrDim : dict
    
    """
    if self.model_ == "pca":
        # Active data
        data = self.active_data_
        row_coord = get_pca_ind(self)["coord"]
        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()

        # Add supplementary continuous variables
        if self.quanti_sup_labels_ is not None:
            quanti_sup = self.data_[self.quanti_sup_labels_]
            # Drop supplementary row
            if self.row_sup_labels_ is not None:
                quanti_sup = quanti_sup.drop(index=self.row_sup_labels_)
                data = pd.concat([data,quanti_sup],axis=1)
    
        corrdim = {}
        for idx in row_coord.columns:
            # For continuous variables
            corDim = pd.DataFrame(columns=["correlation","pvalue"]).astype("float")
            for col in data.columns:
                if (data[col].dtypes in ["float64","int64","float32","int32"]):
                    res = st.pearsonr(data[col],row_coord[idx])
                    row_RD = pd.DataFrame({"correlation" : res.statistic,"pvalue":res.pvalue},index = [col])
                    corDim = pd.concat([corDim,row_RD],axis=0)
            # Filter by pvalue
            corDim = (corDim.query(f'pvalue < {proba}')
                            .sort_values(by="correlation",ascending=False))

            # For categorical variables
            if self.quali_sup_labels_ is not None:
                quali_sup = self.data_[self.quali_sup_labels_]
                if self.row_sup_labels_ is not None:
                    quali_sup = quali_sup.drop(index=self.row_sup_labels_)
                
                corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','correlation ratio','F-stats','pvalue'])
                for col in quali_sup.columns:
                    row_RD = pd.DataFrame(eta2(quali_sup[col],row_coord[idx],digits=8),index=[col])
                    corqDim = pd.concat([corqDim,row_RD],axis=0)
                # Filter by pvalue
                corqDim = (corqDim.query(f'pvalue < {proba}')
                                  .sort_values(by="correlation ratio",ascending=False)
                                  .rename(columns={"R2": "correlation ratio"}))
            
            if self.quali_sup_labels_ is None:
                res = corDim
            else:
                if corqDim.shape[0] != 0 :
                    res = {"quanti":corDim,"quali":corqDim}
                else:
                    res = corDim
            
            corrdim[idx] = res
    elif self.model_ == "ca":
        # Extract row coordinates
        row_coord = get_ca_row(self)["coord"]
        # Exctrac columns coordinates
        col_coord = get_ca_col(self)["coord"]

        # Add Supplementary row
        if self.row_sup_labels_ is not None:
            row_coord_sup = get_ca_row(self)["row_sup"]["coord"]
            row_coord = pd.concat([row_coord,row_coord_sup],axis=0)
        
        # Add supplmentary columns
        if self.col_sup_labels_ is not None:
            col_coord_sup = get_ca_col(self)["col_sup"]["coord"]
            col_coord = pd.concat([col_coord,col_coord_sup],axis=0)

        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            col_coord = col_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()
                col_coord = col_coord.to_frame()
        
        corrdim = {}
        for idx in row_coord.columns:
            corrdim[idx] = {"row" : (row_coord[idx].to_frame()
                                                   .sort_values(by=idx,ascending=True)
                                                   .rename(columns={idx:"coord"})),
                            "col" : (col_coord[idx].to_frame()
                                                   .sort_values(by=idx,ascending=True)
                                                   .rename(columns={idx:"coord"}))}
    elif self.model_ == "mca":
        # Select data
        data = self.original_data_
        # Extract row coordinates
        row_coord = get_mca_ind(self)["coord"]
        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()

        if self.quali_sup_labels_ is not None:
            quali_sup = self.data_[self.quali_sup_labels_]
            data = pd.concat([data,quali_sup],axis=1)
            if self.row_sup_labels_ is not None:
                data = data.drop(index=self.row_sup_labels_)

        corrdim = {}
        for idx in row_coord.columns:
            # Pearson correlation test
            if self.quanti_sup_labels_ is not None:
                quanti_sup = self.data_[self.quanti_sup_labels_]
                if self.row_sup_labels_ is not None:
                    quanti_sup = quanti_sup.drop(index=self.row_sup_labels_)
                
                corDim = pd.DataFrame(columns=["statistic","pvalue"]).astype("float")
                for col in quanti_sup.columns:
                    if (quanti_sup[col].dtypes in ["float64","int64","float32","int32"]):
                        res = st.pearsonr(quanti_sup[col],row_coord[idx])
                        row_RD = pd.DataFrame({"statistic" : res.statistic,"pvalue":res.pvalue},index = [col])
                        corDim = pd.concat([corDim,row_RD])
                # Filter by pvalue
                corDim = (corDim.query(f'pvalue < {proba}')
                                .sort_values(by="statistic",ascending=False))

            # Correlation ratio (eta2)
            corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','correlation ratio','F-stats','pvalue'])
            for col in data.columns:
                row_RD = pd.DataFrame(eta2(data[col],row_coord[idx],digits=8),index=[col])
                corqDim = pd.concat([corqDim,row_RD])
            # Filter by pvalue
            corqDim = (corqDim.query(f'pvalue < {proba}')
                              .sort_values(by="correlation ratio",ascending=False)
                              .rename(columns={"correlation ratio" : "R2"}))
        
            if self.quanti_sup_labels_ is None:
                res = corqDim
            else:
                if corDim.shape[0] != 0 :
                    res = {"quali":corqDim,"quanti":corDim}
                else:
                    res = corqDim
            corrdim[idx] = res
    elif self.model_ == "famd":
        # Extract row coord
        row_coord = get_famd_ind(self)["coord"]
        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()
        
        # Select continuous active data
        quanti_data = self.active_data_[self.quanti_labels_]
        # Add supplementary continuous variables
        if self.quanti_sup_labels_ is not None:
            quanti_sup = self.data_[self.quanti_sup_labels_]
            if self.row_sup_labels_ is not None:
                quanti_sup = quanti_sup.drop(index=self.row_sup_labels_)
                quanti_data = pd.concat([quanti_data,quanti_sup],axis=1)

        # Select categorical active variables
        quali_data = self.active_data_[self.quali_labels_]
        # Add supplementary categorical variables
        if self.quali_sup_labels is not None:
            quali_sup = self.data_[self.quali_sup_labels_]
            if self.row_sup_labels_ is not None:
                quali_sup = quali_sup.drop(index=self.row_sup_labels_)
                quali_data = pd.concat([quali_data,quali_sup],axis=1)
        
        # Correlation between coninuous variable and axis
        corrdim = {}
        for idx in row_coord.columns:
            # For continuous variables
            corDim = pd.DataFrame(columns=["correlation","pvalue"]).astype("float")
            for col in quanti_data.columns:
                if (quanti_data[col].dtypes in ["float64","int64","float32","int32"]):
                    res = st.pearsonr(quanti_data[col],row_coord[idx])
                    row_RD = pd.DataFrame({"correlation" : res.statistic,"pvalue":res.pvalue},index = [col])
                    corDim = pd.concat([corDim,row_RD],axis=0)
            # Filter by pvalue
            corDim = (corDim.query(f'pvalue < {proba}')
                            .sort_values(by="correlation",ascending=False))

            # For categorical variable    
            corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','correlation ratio','F-stats','pvalue'])
            for col in quali_data.columns:
                row_RD = pd.DataFrame(eta2(quali_data[col],row_coord[idx],digits=8),index=[col])
                corqDim = pd.concat([corqDim,row_RD],axis=0)
            # Filter by pvalue
            corqDim = (corqDim.query(f'pvalue < {proba}')
                              .sort_values(by="correlation ratio",ascending=False)
                              .rename(columns={"correlation ratio" : "R2"}))

            if corDim.shape[0] == 0 and corqDim.shape[0] != 0:
                res = corqDim
            elif corDim.shape[0] != 0 and corqDim.shape[0] == 0:
                res = corDim
            else:
                res = {"quanti":corDim,"quali":corqDim}
              
            corrdim[idx] = res

    return corrdim


###############################################" Fonction de reconstruction #######################################################

def reconst(self,n_components=None):
    """
    Reconstitution of data

    This function reconstructs a data set from the result of a PCA 

    Parameters:
    -----------
    self : an instance of class PCA
    n_components : int, the number of dimensions to use to reconstitute data.

    Return
    ------
    X : Reconstitution data.
    """

    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an instance of class PCA.")
    if n_components is not None:
        if n_components > self.n_components_:
            raise ValueError("Error : Enter good number of n_components" )
    else:
        raise ValueError("Error : 'n_components' must be pass.")
    
    # Valeurs centrées
    Z = np.dot(self.row_coord_[:,:n_components],self.eigen_vectors_[:,:n_components].T)

    # Déstandardisation et décentrage
    X = self.active_data_.copy()
    for k in np.arange(len(self.col_labels_)):
        X.iloc[:,k] = Z[:,k]*self.std_[0][k] + self.means_[0][k]
    
    return X










###########################################"" Discriminant Correspondence Analysis (CDA) ###########################################

# Row informations
def get_disca_ind(self):
    pass

# Categories informations
def get_disca_mod(self):
    pass

# Group informations
def get_disca_group(self):
    pass

# Disca extract informations
def get_disca(self,choice="ind"):
    """
    
    """
    if choice == "ind":
        return get_disca_ind(self)
    elif choice == "mod":
        return get_disca_mod(self)
    elif choice == "group":
        return get_disca_group(self)
    else:
        raise ValueError("Error : give a valid choice.")

# Summary DISCA
def summaryDISCA(self):
    pass