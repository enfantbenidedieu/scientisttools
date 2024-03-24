# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy
from scientisttools.utils import eta2
import scipy.stats as st


################## Eigenvalues
def get_eig(self) -> pd.DataFrame:

    """
    Extract the eigenvalues/variances of dimensions
    -----------------------------------------------

    Description
    -----------
    Eigenvalues correspond to the amount of the variation explained by each principal component.

    Parameters:
    -----------
    self : an object of class PCA, PartialPCA, CA, MCA, FAMD, MFA, MFAQUAL, CMDS, MFA, HMFA

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent
    """
    if self.model_ in ["pca","partialpca","ca","mca","famd","efa","mfa","mfaqual","mfamix","cmds","candisc","hmfa"]:
        return self.eig_
    else:
        raise ValueError("Error : 'self' must be an object of class PCA, PartialPCA, CA, MCA, FAMD, EFA, MFA, MFAQUAL, MFAMIX, CMDS, HMFA")

def get_eigenvalue(self) -> pd.DataFrame:

    """
    Extract the eigenvalues/variances of dimensions
    -----------------------------------------------

    Description
    -----------
    Eigenvalues correspond to the amount of the variation explained by each principal component.

    Parameters:
    -----------
    self : an object of class PCA, PartialPCA, CA, MCA, FAMD, MFA, MFAQUAL, CMDS, MFA, HMFA

    Returns
    -------
    eigenvalue, difference, variance percent and cumulative variance of percent
    """
    return get_eig(self)

######################################## Principal Components Analysis (PCA) extractions ######################################################

############ Principal Components Analysis

def get_pca_ind(self,choice="ind") -> dict:

    """
    Extract the results for individuals - PCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions) for the active individuals from Principal Component Analysis (PCA) outputs.

    Parameters:
    -----------
    self : an object of class PCA

    choice : 

    Returns
    -------
    Principal Component Analysis - Results for individuals
    ===============================================================
        Names       Description
    1   "coord"     "coordinates for the individuals"
    2   "cos2"      "cos2 for the individuals"
    3   "contrib"   "contributions of the individuals"
    4   "dist"      "square distance between individuals and origin"
    5   "infos"     "additionnal informations for the individuals :"
                        - square distance between individuals and origin
                        - weights for the individuals
                        - inertia for the individuals
    
    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")
    
    if choice not in ["ind", "ind_sup"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'ind_sup'")
    
    if choice == "ind":
        return  self.ind_
    elif choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("Error : No supplementary individuals")
        return self.ind_sup_

### Extract variables
def get_pca_var(self,choice="var") -> dict:

    """
    Extract the results for variables - PCA
    ---------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions) for the active variables from Principal Component Analysis (PCA) outputs.

    Parameters
    ----------
    self : an object of class PCA

    choice : 

    Returns
    -------
    Principal Component Analysis - Results for variables
    ==============================================================
        Names       Description
    1   "coord"     "coordinates for the continuous variables"
    2   "cor"       "correlations between variables and dimensions"
    3   "cos2"      "cos2 for the continuous variables"
    4   "contrib"   "contributions of the continuous variables"
    5   "weighted"  "weighted Peron correlation between continuous variables"
    6   "infos"     "additionnal informations for the variables :"
                        - square distance between variables and origin
                        - weights for the variables
                        - inertia for the variables
    
    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA")
    
    if choice not in ["var", "quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of 'var', 'quanti_sup', 'quali_sup'")
    
    if choice == "var":
        return self.var_
    elif choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("Error : No supplementary quantitatives variables")
        return self.quanti_sup_  
    elif choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("Error : No supplementary qualitatives variables")
        return self.quali_sup_

def get_pca(self,choice="ind")-> dict:

    """
    Extract the results for individuals/variables - PCA
    ---------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions) for the active individuals/variables from Principal Component Analysis (PCA) outputs.

    *   get_pca() : Extract the results for variables and individuals
    *   get_pca_ind() : Extract the results for individuals only
    *   get_pca_var() : Extract the results for variables only

    Parameters
    ----------
    self : an object of class PCA

    choice : the element to subset from the output. Allowed values are :
                - "ind" for active individuals
                - "ind_sup" for supplementary individuals
                - "var" for active variables
                - "quanti_sup" for supplementary quantitatives variables
                - "quali_sup" for supplementary qualitatives variables

    Returns
    -------
    a dictionary of dataframes containing all the results for the active individuals/variables including:
    - coord : coordinates for the individuals/variables
    - cos2: cos2 for the individuals/variables
    - contrib : contributions of the individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")

    if choice not in ["ind","ind_sup","var","quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'ind_sup', 'var', 'quanti_var', 'quali_var'")

    if choice in ["ind","ind_sup"]:
        return get_pca_ind(self,choice=choice)
    else:
        return get_pca_var(self,choice=choice)

##### Summary 
def summaryPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Principal Components Analysis objects
    -----------------------------------------------------------

    Description
    -----------
    Printing summaries of principal component analysis (PCA) objects

    Parameters
    ----------
    self        :   an obect of class PCA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA")

    # Define number of components
    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])

    # Principal Components Analysis Results
    print("                     Principal Component Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative % of var."]
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

    # Add supplementary individuals
    if self.ind_sup is not None:
        print(f"\nSupplementary individuals\n")
        # Save all informations
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in np.arange(0,ncp,1):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat([ind_sup_infos,ind_sup_coord,ind_sup_cos2],axis=1)
        ind_sup_infos = ind_sup_infos.round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

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
    var_infos = var_infos.round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup is not None:
        print(f"\nSupplementary continuous variables\n")
        quanti_sup_infos = pd.DataFrame().astype("float")
        quanti_sup = self.quanti_sup_
        for i in np.arange(0,ncp,1):
            quanti_sup_coord = quanti_sup["coord"].iloc[:,i]
            quanti_sup_cos2 = quanti_sup["cos2"].iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos =pd.concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup is not None:
        print("\nSupplementary categories\n")
        quali_sup = self.quali_sup_
        quali_sup_infos = quali_sup["dist"]
        for i in np.arange(0,ncp,1):
            quali_sup_coord = quali_sup["coord"].iloc[:,i]
            quali_sup_cos2 = quali_sup["cos2"].iloc[:,i]
            quali_sup_cos2.name = "cos2"
            quali_sup_vtest = quali_sup["vtest"].iloc[:,i]
            quali_sup_vtest.name = "v.test"
            quali_sup_infos = pd.concat([quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest],axis=1)
        quali_sup_infos = quali_sup_infos.round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        print("\nSupplementary categorical variable (eta2)\n")
        quali_sup_eta2 = quali_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)

#############################
def get_ca_row(self,choice="row")-> dict:
    """
    Extract the resultst for rows - CA
    ----------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions and inertia) for the active row variables from Correspondence Analysis (CA) outputs.

    Parameters
    ----------
    self. : an object of class CA

    choice : 

    Returns
    -------
    a dictionary of dataframes containing the results for the active rows including :
    coord   : coordinates for the rows of shape (n_rows, n_components)
    cos2    : cos2 for the rows of shape (n_rows, n_components)
    contrib : contributions for the rows of shape (n_rows, n_components)
    infos   : additionnal informations for the rows:
                - square root distance between rows and inertia
                - marge for the rows
                - inertia for the rows
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")

    if choice not in ["row", "row_sup"]:
        raise ValueError("Error : 'choice' should be one of 'row', 'row_sup'")
    
    if choice == "row":
        return self.row_
    elif choice == "row_sup":
        if self.row_sup is None:
            raise ValueError("Error : No supplementary rows")
        return self.row_sup_
            
def get_ca_col(self,choice="col")-> dict:

    """
    Extract the results for columns - CA
    ------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions and inertia) for the active column variables from Correspondence Analysis (CA) outputs.

    Parameters
    ----------
    self : an object of class CA

    choice : 

    Returns
    -------
    a dictionary of dataframes containing the results for the active columns including :
    coord   : coordinates for the columns of shape (n_cols, n_components)
    cos2    : cos2 for the columns of shape (n_cols, n_components)
    contrib : contributions for the columns of shape (n_cols, n_components)
    infos   : additionnal informations for the columns:
                - square root distance between columns and inertia
                - marge for the columns
                - inertia for the columns
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if choice not in ["col","col_sup","quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of 'col', 'col_sup', 'quanti_sup', 'quali_sup'")
    
    if choice == "col":
        return self.col_
    elif choice == "col_sup":
        if self.col_sup is None:
            raise ValueError("Error : No supplementary columns")
        return self.col_sup_
    elif choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("Error : No supplementary quantitatives columns")
        return self.quanti_sup_
    elif choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("Error : No supplementary qualitatives columns")

def get_ca(self,choice = "row")-> dict:
    """
    Extract the results for rows/columns - CA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions and inertia) for the active row/column variables from Correspondence Analysis (CA) outputs.

    * get_ca() : Extract the results for rows and columns
    * get_ca_row() : Extract the results for rows only
    * get_ca_col() : Extract the results for columns only

    Parameters
    ----------
    self : an object of class CA

    choice : 

    Return
    ------
    a dictionary of dataframes containing the results for the active rows/columns including :
    coord   : coordinates for the rows/columns
    cos2    : cos2 for the rows/columns
    contrib	: contributions of the rows/columns
    infos   : additionnal informations for the row/columns:
                - square root distance between rows/columns and inertia
                - marge for the rows/columns
                - inertia for the rows/columns

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if choice not in ["row","row_sup","col","col_sup","quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of 'row', 'row_sup', 'col', 'col_sup', 'quanti_sup', 'quali_sup'")
    
    if choice in ["row","row_sup"]:
        return get_ca_row(self,choice=choice)
    else:
        return get_ca_col(self,choice=choice)
    

def summaryCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt="pipe",**kwargs):
    """
    Printing summaries of Correspondence Analysis model
    ---------------------------------------------------

    Parameters
    ----------
    self        :   an obect of class CA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA")

    # Set number of components
    ncp = min(ncp,self.call_["n_components"])
    # Set number of elements
    nb_element = min(nb_element,self.call_["X"].shape[0])

    # Principal Components Analysis Results
    print("                     Correspondence Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nRows\n")
    row = get_ca(self,choice="row")
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
    if self.row_sup is not None:
        print(f"\nSupplementary rows\n")
        # Save all informations
        row_sup = self.row_sup_
        row_sup_infos = row_sup["dist"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2  = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    # Add variables informations
    print(f"\nColumns\n")
    col = get_ca(self,choice="col")
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
    if self.col_sup is not None:
        print(f"\nSupplementary columns\n")
        col_sup = self.col_sup_
        col_sup_infos = col_sup["dist"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos = pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
    # Add supplementary quantitatives informations
    if self.quanti_sup is not None:
        print(f"\nSupplementary quantitatives columns\n")
        quanti_sup = self.quanti_sup_
        quanti_sup_infos = pd.DataFrame().astype("float")
        for i in np.arange(0,ncp,1):
            quanti_sup_coord = quanti_sup["coord"].iloc[:,i]
            quanti_sup_cos2 = quanti_sup["cos2"].iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos = pd.concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    # Add supplementary qualitatives informations
    if self.quali_sup is not None:
        print(f"\nSupplementary categories\n")
        quali_sup = self.quali_sup_
        quali_sup_infos = quali_sup["dist"]
        for i in np.arange(0,ncp,1):
            quali_sup_coord = quali_sup["coord"].iloc[:,i]
            quali_sup_cos2 = quali_sup["cos2"].iloc[:,i]
            quali_sup_cos2.name = "cos2"
            quali_sup_vtest = quali_sup["vtest"].iloc[:,i]
            quali_sup_vtest.name = "vtest"
            quali_sup_infos = pd.concat([quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest],axis=1)
        quali_sup_infos = quali_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        print("\nSupplementary qualitatives variables (eta2)\n")
        quali_sup_eta2 = quali_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)


########## Multiple Correspondence Analysis

def get_mca_ind(self,choice="ind") -> dict:
    """
    Extract the results for individuals - MCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals 
    from Multiple Correspondence Analysis (MCA) outputs.

    Parameters
    ----------
    self : an object of class MCA

    choice  : the element to subset from the output. Possible values are :
                - "ind" for individuals, 
                - "ind_sup" for supplementary individuals.

    Return
    ------
    a dictionary of dataframes containing the results for the active individuals categories including :

    coord   : coordinates for the individuals

    cos2    : cos2 for the individuals

    contrib : contributions of the individuals

    infos   : additionnal informations for the individuals :
                - square root distance between individuals and inertia
                - weights for the individuals
                - inertia for the individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    if choice == "ind":
        return self.ind_
    elif choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("Error : No supplementary individuals.")
        return self.ind_sup_
            
def get_mca_var(self,choice="var") -> dict:
    """
    Extract the results for the variables - MCA
    -------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active variable 
    categories from Multiple Correspondence Analysis (MCA) outputs.

    Parameters
    ----------
    self : an object of class MCA

    choice  : the element to subset from the output. Possible values are :
                - "var" for variables, 
                - "quanti_sup" for quantitative supplementary variables,
                - "quali_sup" for qualitatives supplementary variables

    Returns
    -------
    a dictionary of dataframes containing the results for the active variable categories including :

    coord           : coordinates for the variables categories

    corrected_coord : corrected coordinates for the variables categories

    cos2            : cos2 for the variables categories

    contrib         : contributions of the variables categories

    infos           : additionnal informations for the variables categories :
                        - square root distance between variables categories and inertia
                        - weights for the variables categories
                        - inertia for the variables categories

    vtest           : v-test for the variables categories

    eta2            : squared correlation ratio for the variables

    inertia         : inertia of the variables

    var_contrib     : contributions of the variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if choice not in ["var","quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of 'var', 'quanti_sup', 'quali_sup'")

    if choice == "var":
        return self.var_
    elif choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("Error : No quantitatives supplementary variables.")
        return self.quanti_sup_
    elif choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("Error : No qualitatives supplementary variables.")
        return self.quali_sup_
            
def get_mca(self,choice="ind") -> dict:
    """
    Extract the results for individuals/variables - MCA
    ---------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals/variable 
    categories from Multiple Correspondence Analysis (MCA) outputs.

    * get_mca()     : Extract the results for vriables and individuals
    * get_mca_ind() : Extract the results for individuals only
    * get_mca_var() : Extract the results for variables only

    Parameters
    ----------
    self    : an object of class MCA

    choice  : the element to subset from the output. Possible values are :
                - "var" for variables, 
                - "ind" for individuals, 
                - "ind_sup" for supplementary individuals,
                - "quanti_sup" for quantitative supplementary variables,
                - "quali_sup" for qualitatives supplementary variables
    
    Return
    ------
    a dictionary of dataframes containing the results for the active individuals/variable categories including :

    coord : coordinates for the individuals/variable categories

    cos2 : cos2 for the individuals/variable categories

    contrib	: contributions of the individuals/variable categories

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA.")
    
    if choice not in ["ind","ind_sup","var","quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of : 'ind', 'ind_sup', 'var', 'quanti_sup', 'quali_sup'")

    if choice in ["ind","ind_sup"]:
        return get_mca_ind(self,choice=choice)
    else:
        return get_mca_var(self,choice=choice)
    

def summaryMCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Correspondence Analysis model
    ------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MCA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mca":
        raise ValueError("Error : 'self' must be an object of class MCA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0],self.var_["coord"].shape[0])

    # Multiple correspondance Analysis - Results
    print("                     Multiple Correspondance Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.T.round(decimals=digits)
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

    # Add supplementary individuals
    if self.ind_sup is not None:
        nb_elt = min(nb_element,self.ind_sup_["coord"].shape[0])
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in np.arange(0,ncp,1):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat([ind_sup_infos,ind_sup_coord,ind_sup_cos2],axis=1)
        ind_sup_infos = ind_sup_infos.iloc[:nb_elt,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    # Add variables informations
    print(f"\nCategories (the {nb_element} first)\n")
    var = self.var_
    var_infos = var["infos"]
    for i in np.arange(0,ncp,1):
        var_coord = var["coord"].iloc[:,i]
        var_cos2 = var["cos2"].iloc[:,i]
        var_cos2.name = "cos2"
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr"
        var_vtest = var["vtest"].iloc[:,i]
        var_vtest.name = "vtest"
        var_infos = pd.concat([var_infos,var_coord,var_ctr,var_cos2,var_vtest],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add variables
    print("\nCategorical variables (eta2)\n")
    quali_var_infos = var["inertia"]
    for i in np.arange(0,ncp,1):
        quali_var_eta2 = var["eta2"].iloc[:,i]
        quali_var_eta2.name = "Dim."+str(i+1)
        quali_var_contrib = var["var_contrib"].iloc[:,i]
        quali_var_contrib.name = "ctr"
        quali_var_infos = pd.concat([quali_var_infos,quali_var_eta2,quali_var_contrib],axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)

    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup is not None:
        print("\nSupplementary categories\n")
        var_sup = self.quali_sup_
        var_sup_infos = var_sup["dist"]
        for i in np.arange(0,ncp,1):
            var_sup_coord = var_sup["coord"].iloc[:,i]
            var_sup_cos2 = var_sup["cos2"].iloc[:,i]
            var_sup_cos2.name = "cos2"
            var_sup_vtest = var_sup["vtest"].iloc[:,i]
            var_sup_vtest.name = "v.test"
            var_sup_infos = pd.concat([var_sup_infos,var_sup_coord,var_sup_cos2,var_sup_vtest],axis=1)
        var_sup_infos = var_sup_infos.round(decimals=digits)
        if to_markdown:
            print(var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(var_sup_infos)
        
        print("\nSupplementary categorical variables (eta2)\n")
        quali_var_sup_infos = var_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_var_sup_infos)

    # Add supplementary continuous variables informations
    if self.quanti_sup is not None:
        print(f"\nSupplementary continuous variable\n")
        quanti_sup = self.quanti_sup_
        quanti_sup_infos = pd.DataFrame().astype("float")
        for i in np.arange(0,ncp,1):
            quanti_sup_coord = quanti_sup["coord"].iloc[:,i]
            quanti_sup_cos2 = quanti_sup["cos2"].iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos = pd.concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos) 

############ Factor analysis of mixed data

def get_famd_ind(self,choice = "ind") -> dict:
    """
    Extract the results for individuals
    -----------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the individuals 
    from Factor Analysis of Mixed Date (FAMD) outputs.

    Parameters
    ----------
    self : an object of class FAMD

    choice : the element to subset from the output. Possible values are 
                - "ind" for active individuals, 
                - "ind_sup" for supplementary individuals

    Returns
    -------
    a dictionary of dataframes containing the results for the individuals, including :
    coord	: coordinates of indiiduals.
    cos2	: cos2 values representing the quality of representation on the factor map.
    contrib	: contributions of individuals to the principal components.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if choice not in ["ind","ind_sup"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'ind_sup'")
    
    if choice == "ind":
        return self.ind_
    elif choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("Error : No supplementary individuals")
        return self.ind_sup_

def get_famd_var(self,choice="var") -> dict:
    """
    Extract the results for quantitative and qualitative variables
    --------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for quantitative and 
    qualitative variables from Factor Analysis of Mixed Date (FAMD) outputs.

    Parameters
    ----------
    self : an object of class FAMD

    choice : the element to subset from the output. Possible values are 
                - "quanti_var" for active quantitatives variables
                - "quali_var" for active qualitatives variables (categories)
                - "var" for active variables
                - "quanti_sup" for supplementary quantitatives variables
                - "quali_sup" for supplementary qualitatives variables (categories)
                - "var_sup" for supplementary variables

    Returns
    -------
    a list of matrices containing the results for the active individuals and variables, including :
    coord	: coordinates of variables.
    cos2	: cos2 values representing the quality of representation on the factor map.
    contrib	: contributions of variables to the principal components.

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD")
    
    if choice not in ["quanti_var","quali_var","var","quanti_sup","quali_sup","var_sup"]:
        raise ValueError("Error : 'choice' should be one of 'quanti_var', 'quali_var', 'var', 'quanti_sup', 'quali_sup', 'var_sup'")
    
    if choice == "quanti_var":
        return self.quanti_var_
    elif choice == "quali_var":
        return self.quali_var_
    elif choice == "var":
        return self.var_
    elif choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("Error : No supplementary quantitatives columns")
        return self.quanti_sup_
    elif choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("Error : No supplementary qualitatives columns")
    elif choice == "var_sup":
        if self.quanti_sup is None or self.quali_sup is None:
            raise ValueError("Error : No supplementary columns")
        return self.var_sup_

def get_famd(self,choice = "ind")-> dict:
    """
    Extract the results for individuals and variables - FAMD
    --------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the individuals and variables 
    from Factor Analysis of Mixed Date (FAMD) outputs.

    Parameters
    ----------
    self : an object of class FAMD

    choice : the element to subset from the output. 

    Return
    ------
    a dict of dataframes containing the results for the active individuals and variables, including :
    coord	: coordinates of indiiduals/variables.
    cos2	: cos2 values representing the quality of representation on the factor map.
    contrib	: contributions of individuals / variables to the principal components.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD.")
    
    if choice not in ["ind","ind_sup","quanti_var","quali_var","var","quanti_sup","quali_sup","var_sup"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'ind_sup', 'quanti_var', 'quali_var', 'var', 'quanti_sup', 'quali_sup', 'var_sup'")
    

    if choice in ["ind", "ind_sup"]:
        return get_famd_ind(self,choice=choice)
    elif choice not in ["ind","ind_sup"]:
        return get_famd_var(self,choice=choice)

###### FAMD
def summaryFAMD(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Factor Analysis of Mixed Data model
    ---------------------------------------------------------

    Parameters
    ----------
    self        :   an obect of class FAMD.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    # check if famd model
    if self.model_ != "famd":
        raise ValueError("Error : 'self' must be an object of class FAMD")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])

    # Principal Components Analysis Results
    print("                     Factor Analysis of Mixed Data - Results                     \n")

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

    # Add supplementary individuals
    if self.ind_sup is not None:
        print(f"\nSupplementary individuals\n")
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in np.arange(0,ncp,1):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat([ind_sup_infos,ind_sup_coord,ind_sup_cos2],axis=1)
        ind_sup_infos = ind_sup_infos.round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    # Add variables informations
    quanti_var = self.quanti_var_
    if quanti_var["coord"].shape[0]>nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
         print("\nContinuous variables\n")
    quanti_var_infos = pd.DataFrame().astype("float")
    for i in np.arange(0,ncp,1):
        quanti_var_coord = quanti_var["coord"].iloc[:,i]
        quanti_var_cos2 = quanti_var["cos2"].iloc[:,i]
        quanti_var_cos2.name = "cos2"
        quanti_var_ctr = quanti_var["contrib"].iloc[:,i]
        quanti_var_ctr.name = "ctr"
        quanti_var_infos = pd.concat([quanti_var_infos,quanti_var_coord,quanti_var_ctr,quanti_var_cos2],axis=1)
    quanti_var_infos = quanti_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quanti_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quanti_var_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup is not None:
        print(f"\nSupplementary continuous variable\n")
        quanti_sup = self.quanti_sup_
        quanti_sup_infos = pd.DataFrame().astype("float")
        for i in np.arange(0,ncp,1):
            quanti_sup_coord = quanti_sup["coord"].iloc[:,i]
            quanti_sup_cos2 = quanti_sup["cos2"].iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos =pd.concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    # Add variables informations
    quali_var = self.quali_var_
    if quali_var["coord"].shape[0] > nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
        print("\nCategories\n")
    quali_var_infos = quali_var["infos"]
    for i in np.arange(0,ncp,1):
        quali_var_coord = quali_var["coord"].iloc[:,i]
        quali_var_cos2 = quali_var["cos2"].iloc[:,i]
        quali_var_cos2.name = "cos2"
        quali_var_ctr = quali_var["contrib"].iloc[:,i]
        quali_var_ctr.name = "ctr"
        quali_var_vtest = quali_var["vtest"].iloc[:,i]
        quali_var_vtest.name = "vtest"
        quali_var_infos = pd.concat([quali_var_infos,quali_var_coord,quali_var_ctr,quali_var_cos2,quali_var_vtest],axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)
    
    # Add variables
    print("\nCategorical variables (eta2)\n")
    quali_var_eta2 = self.var_["coord"].loc[self.call_["quali"].columns.tolist(),:].iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_eta2)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup is not None:
        print("\nSupplementary categories\n")
        quali_sup = self.quali_sup_
        quali_sup_infos = quali_sup["dist"]
        for i in np.arange(0,ncp,1):
            quali_sup_coord = quali_sup["coord"].iloc[:,i]
            quali_sup_cos2 = quali_sup["cos2"].iloc[:,i]
            quali_sup_cos2.name = "cos2"
            quali_sup_vtest = quali_sup["vtest"].iloc[:,i]
            quali_sup_vtest.name = "v.test"
            quali_sup_infos = pd.concat([quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest],axis=1)
        quali_sup_infos = quali_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        print("\nSupplementary categorical variables (eta2)\n")
        quali_sup_eta2 = self.quali_sup_["eta2"].iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)

########## Partial Principal Components Analysis
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
        raise ValueError("Error : 'self' must be an object of class PartialPCA.")
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
        raise ValueError("Error : 'self' must be an object of class PartialPCA")
    
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
        raise ValueError("Error : 'self' must be an object of class PartialPCA.")
    if choice not in ["ind","var"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'var'")
    
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
    self        :   an obect of class PartialPCA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "partialpca":
        raise ValueError("Error : 'self' must be an object of class PartialPCA")

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

# -*- coding: utf-8 -*-
        
################### Exploratory factor analysis

def get_efa_ind(self,choice = "ind") -> dict:

    """
    Extract the results for individuals - EFA
    -----------------------------------------

    Parameters
    ----------
    self : an object of class EFA

    Returns
    -------
    

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")

    if choice not in ["ind","ind_sup"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'ind_sup'")
    
    if choice == "ind":
        return self.ind_
    elif choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("Error : No supplementary individuals")
        return self.ind_sup_

def get_efa_var(self) -> dict:

    """
    Extract the results for variables - EFA
    ---------------------------------------

    Parameters
    ----------
    self : an instance of class EFA

    Returns
    -------
    

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")
    return self.var_

def get_efa(self,choice = "ind")-> dict:

    """
    Extract the results for individuals/variables - EFA
    ---------------------------------------------------

    Parameters
    ---------
    self : an instance of class EFA

    choice : 

    Returns
    -------
    

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA.")

    if choice not in ["ind","ind_sup","var"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'ind_sup', 'var'")
    if choice in ["ind", "ind_sup"]:
        return get_efa_ind(self,choice=choice)
    elif choice == "var":
        return get_efa_var(self)

def summaryEFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Exploratory Factor Analysis model
    -------------------------------------------------------

    Parameters
    ----------
    self        :   an obect of class EFA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    # check if EFA model
    if self.model_ != "efa":
        raise ValueError("Error : 'self' must be an object of class EFA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0],self.var_["coord"].shape[0])

    # Exploratory Factor Analysis Results
    print("                     Exploratory Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first) \n")
    ind_coord = self.ind_["coord"].iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(ind_coord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_coord)

    # Add supplementary individuals
    if self.ind_sup is not None:
        nb_elt = min(nb_element,self.ind_sup_["coord"].shape[0])
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        ind_sup_infos = self.ind_sup_["coord"].iloc[:nb_elt,:ncp].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    # Add variables informations
    print(f"\nContinues Variables\n")
    var = self.var_
    var_infos = pd.DataFrame().astype("float")
    for i in np.arange(0,ncp,1):
        var_coord = var["coord"].iloc[:,i]
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr"
        var_infos = pd.concat([var_infos,var_coord,var_ctr],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
#############################################################################################
#           Multiple Factor Analysis (MFA)
#############################################################################################

def get_mfa_ind(self):
    """
    Extract the results for individuals - MFA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals
    from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    self : an object of class MFA

    Return
    ------
    a dictionnary of dataframes containing the results for the active individuals including :
    coord	: coordinates for the individuals
    
    cos2	: cos2 for the individuals
    
    contrib	: contributions of the individuals
    
    inertia	: inertia of the individuals


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an object of class MFA.")
    
    return self.ind_

def get_mfa_var(self,choice = "group"):
    """
    Extract the results for variables (quantitatives and groups) MFA
    ------------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active quantitative variable/groups 
    from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    self  : an object of class MFA
    
    choice : the element to subset from the output. Possible values are "quanti_var","group","quali_var"

    Value
    -----
    a dictionnary of dataframes containing the results for the active quantitative variable/groups including :

    coord	: coordinates for the quantitative variable/groups
    
    cos2	: cos2 for the quantitative variable/groups
    
    contrib	: contributions of the quantitative variable/groups

    Usage
    -----

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an object of class MFA.")
    
    if choice not in ["group","quanti_var","quali_var"]:
        raise ValueError("Error : 'choice' should be one of 'group', 'quanti_var', 'quali_var'")
    
    if choice == "group":
        return self.group_
    if choice == "quanti_var":
        return self.quanti_var_
    if choice == "quali_var":
        if self.num_group_sup is None:
            raise ValueError("Erro : No categorical variable")
        return self.quali_var_sup_

def get_mfa_partial_axes(self):
    """
    Extract the results for partial axes - MFA
    ------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active 
    partial axes from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    self : an object of class MFA

    Return
    ------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    
    """
    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an object of class MFA.")
    
    return self.partial_axes_

def get_mfa(self,choice="ind"):
    """
    Extract the results for individuals/variables/group/partial axes - MFA
    ----------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals/quantitative 
    variables/groups/partial axes from Multiple Factor Analysis (MFA) outputs.
    
    * get_mfa(): Extract the results for variables and individuals
    
    * get_mfa_ind(): Extract the results for individuals only
    
    * get_mfa_var(): Extract the results for variables (quantitatives, qualitatives and groups)
    
    * get_mfa_partial_axes(): Extract the results for partial axes only

    Parameters
    ----------
    self : an object of class MFA

    choice : he element to subset from the output. Possible values are "ind", "quanti_var", "group", 'quali_var' or "partial_axes".

    Return
    ------
    a dictionary of dataframes containing the results for the active individuals/quantitative variable groups/partial axes including :

    coord	: coordinates for the individuals/quantitative variable/groups/partial axes
    
    cos2	: cos2 for the individuals/quantitative variable/groups/partial axes

    contrib	: contributions of the individuals/quantitative variable/groups/partial axes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an object of class MFA.")

    if choice not in ["ind","quanti_var","group","quali_var","partial_axes"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'quanti_var', 'group', 'quali_var', 'partial_axes'")
    
    if choice == "ind":
        return get_mfa_ind(self)
    elif choice == "partial_axes":
        return get_mfa_partial_axes(self)
    else:
        return get_mfa_var(self,choice=choice)
    
def summaryMFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis model
    ----------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.coms
    """

    if self.model_ != "mfa":
        raise ValueError("Error : 'self' must be an object of class MFA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                     Multiple Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    for i in range(ncp):
        group_coord = group["coord"].iloc[:,i]
        group_contrib = group["contrib"].iloc[:,i]
        group_contrib.name = "ctr"
        group_cos2 = group["cos2"].iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if self.ind_sup is not None:
        if self.quanti_var_["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup = self.ind_sup_
        ind_sup_infos = ind["dist"]
        for i in range(ncp):
            ind_sup_coord = ind["coord"].iloc[:,i]
            ind_sup_cos2 = ind["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.quanti_var_["coord"].shape[0] > nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
        print("\nContinuous variables\n")
    quanti_var = self.quanti_var_
    quanti_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quanti_var_coord = quanti_var["coord"].iloc[:,i]
        quanti_var_contrib = quanti_var["contrib"].iloc[:,i]
        quanti_var_contrib.name = "ctr"
        quanti_var_cos2 = quanti_var["cos2"].iloc[:,i]
        quanti_var_cos2.name = "cos2"
        quanti_var_infos = pd.concat((quanti_var_infos,quanti_var_coord,quanti_var_contrib,quanti_var_cos2),axis=1)
    quanti_var_infos = quanti_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quanti_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quanti_var_infos)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        if self.quanti_var_sup_ is not None:
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)
        
        if self.quali_var_sup_ is not None:
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories\n")
            quali_var_sup = self.quali_var_sup_
            quali_var_sup_infos = quali_var_sup["dist"]
            for i in range(ncp):
                quali_var_sup_coord = quali_var_sup["coord"].iloc[:,i]
                quali_var_sup_cos2 = quali_var_sup["cos2"].iloc[:,i]
                quali_var_sup_cos2.name = "cos2"
                quali_var_sup_vtest = quali_var_sup["vtest"].iloc[:,i]
                quali_var_sup_vtest.name = "vtest"
                quali_var_sup_infos = pd.concat((quali_var_sup_infos,quali_var_sup_coord,quali_var_sup_cos2,),axis=1)
            quali_var_sup_infos = quali_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_infos)

            # Add correlation ratio
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories (eta2)\n")
            quali_var_sup_eta2 = quali_var_sup["eta2"].iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_eta2)
    
    
#############################################################################################
#           Multiple Factor Analysis for qualitatives variables (MFAQUAL)
#############################################################################################

def get_mfaqual_ind(self):
    """
    Extract the results for individuals - MFAQUAL
    ---------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals
    from Multiple Factor Analysis for qualitatives variables (MFAQUAL) outputs.

    Parameters
    ----------
    self : an object of class MFAQUAL

    Return
    ------
    a dictionnary of dataframes containing the results for the active individuals including :
    coord	: coordinates for the individuals
    
    cos2	: cos2 for the individuals
    
    contrib	: contributions of the individuals
    
    inertia	: inertia of the individuals


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfaqual":
        raise ValueError("Error : 'self' must be an object of class MFAQUAL")
    
    return self.ind_

def get_mfaqual_var(self,choice = "group"):
    """
    Extract the results for variables (quantitatives, qualitatives and groups) MFAQUAL
    ----------------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active qualitative variable/groups 
    from Multiple Factor Analysis for qualitatives variables (MFAQUAL) outputs.

    Parameters
    ----------
    self  : an object of class MFAQUAL
    
    choice : the element to subset from the output. Possible values are "quali_var", "quanti_var","group"

    Value
    -----
    a dictionnary of dataframes containing the results for the active qualitative variable/roups including :

    coord	: coordinates for the qualitatives variable/groups
    
    cos2	: cos2 for the qualitative variable/groups
    
    contrib	: contributions of the qualitative variable/groups

    Usage
    -----

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfaqual":
        raise ValueError("Error : 'self' must be an object of class MFAQUAL.")
    
    if choice not in ["quali_var", "quanti_var","group"]:
        raise ValueError("Error : 'choice' should be one of 'quali_var', 'quanti_var','group'")
    
    if choice == "group":
        return self.group_
    if choice == "quali_var":
        return self.quali_var_
    if choice == "quanti_var":
        if self.num_group_sup is None:
            raise TypeError("Error : No supplementary quantitatives variables")
        return self.quanti_var_sup_

def get_mfaqual_partial_axes(self):
    """
    Extract the results for partial axes - MFAQUAL
    ----------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active 
    partial axes from Multiple Factor Analysis for qualitatives variables (MFAQUAL) outputs.

    Parameters
    ----------
    self : an object of class MFAQUAL

    Return
    ------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    
    """
    if self.model_ != "mfaqual":
        raise ValueError("Error : 'self' must be an object of class MFAQUAL")
    return self.partial_axes_

def get_mfaqual(self,choice="ind"):
    """
    Extract the results for individuals/variables/group/partial axes - MFAQUAL
    --------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals/quantitative 
    variables/groups/partial axes from Multiple Factor Analysis for qualitatives variables (MFAQUAL) outputs.
    
    * get_mfaqual(): Extract the results for variables and individuals
    
    * get_mfaqual_ind(): Extract the results for individuals only
    
    * get_mfaqual_var(): Extract the results for variables (qualitatives and groups)
    
    * get_mfaqual_partial_axes(): Extract the results for partial axes only

    Parameters
    ----------
    self : an object of class MFAQUAL

    choice : he element to subset from the output. Possible values are "ind", "quali_var", "group" or "partial_axes".

    Return
    ------
    a dictionary of dataframes containing the results for the active individuals/qualitative variable groups/partial axes including :

    coord	: coordinates for the individuals/qualitative variable/groups/partial axes
    
    cos2	: cos2 for the individuals/qualitative variable/groups/partial axes

    contrib	: contributions of the individuals/qualitative variable/groups/partial axes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfaqual":
        raise ValueError("Error : 'self' must be an object of class MFAQUAL")

    if choice not in ["ind","quali_var","quanti_var","group","partial_axes"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'quali_var', 'quanti_var', 'group', 'partial_axes'")
    
    if choice == "ind":
        return get_mfaqual_ind(self)
    elif choice == "partial_axes":
        return get_mfaqual_partial_axes(self)
    else:
        return get_mfaqual_var(self,choice=choice)
    
def summaryMFAQUAL(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis for Qualitatives Variables model
    -------------------------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFAQUAL
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfaqual":
        raise ValueError("Error : 'self' must be an object of class MFAQUAL")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                  Multiple Factor Analysis for Qualitatives Variables - Results                   \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    for i in range(ncp):
        group_coord = group["coord"].iloc[:,i]
        group_contrib = group["contrib"].iloc[:,i]
        group_contrib.name = "ctr"
        group_cos2 = group["cos2"].iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if self.ind_sup is not None:
        ind_sup = self.ind_sup_
        ind_sup_infos = ind["dist"]
        for i in range(ncp):
            ind_sup_coord = ind["coord"].iloc[:,i]
            ind_sup_cos2 = ind["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.quali_var_["coord"].shape[0] > nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
        print("\nCategories\n")
    quali_var = self.quali_var_
    quali_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quali_var_coord = quali_var["coord"].iloc[:,i]
        quali_var_contrib = quali_var["contrib"].iloc[:,i]
        quali_var_contrib.name = "ctr"
        quali_var_cos2 = quali_var["cos2"].iloc[:,i]
        quali_var_cos2.name = "cos2"
        quali_var_infos = pd.concat((quali_var_infos,quali_var_coord,quali_var_contrib,quali_var_cos2),axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)
    
    # Add correlation ratio
    if self.quali_var_["eta2"].shape[0] > nb_element:
        print(f"\nCategories eta2 (the {nb_element} first)\n")
    else:
        print("\nCategories (eta2)\n")
    quali_var_eta2 = quali_var["eta2"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_eta2)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        # Add supplementary qualitatives variables
        if self.quali_var_sup_ is not None:
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories\n")
            quali_var_sup = self.quali_var_sup_
            quali_var_sup_infos = quali_var_sup["dist"]
            for i in range(ncp):
                quali_var_sup_coord = quali_var_sup["coord"].iloc[:,i]
                quali_var_sup_cos2 = quali_var_sup["cos2"].iloc[:,i]
                quali_var_sup_cos2.name = "cos2"
                quali_var_sup_vtest = quali_var_sup["vtest"].iloc[:,i]
                quali_var_sup_vtest.name = "vtest"
                quali_var_sup_infos = pd.concat((quali_var_sup_infos,quali_var_sup_coord,quali_var_sup_cos2,),axis=1)
            quali_var_sup_infos = quali_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_infos)

            # Add correlation ratio
            if self.quali_var_sup_["eta2"].shape[0] > nb_element:
                print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories (eta2)\n")
            quali_var_sup_eta2 = quali_var_sup["eta2"].iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_eta2)

        # Add supplementary quantitatives variables
        if self.quanti_var_sup_ is not None:
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)

#############################################################################################
#           Multiple Factor Analysis of mixed data (MFAMIX)
#############################################################################################

def get_mfamix_ind(self):
    """
    Extract the results for individuals - MFAMIX
    --------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals
    from Multiple Factor Analysis of MIXed data (MFAMIX) outputs.

    Parameters
    ----------
    self : an object of class MFAMIX

    Return
    ------
    a dictionnary of dataframes containing the results for the active individuals including :
    coord	: coordinates for the individuals
    
    cos2	: cos2 for the individuals
    
    contrib	: contributions of the individuals
    
    inertia	: inertia of the individuals


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")
    
    return self.ind_

def get_mfamix_var(self,choice = "group"):
    """
    Extract the results for variables (quantitatives and groups) MFAMIX
    --------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active quantitative variable/groups 
    from Multiple Factor Analysis of MIXed data (MFAQUAL) outputs.

    Parameters
    ----------
    self  : an object of class MFAMIX
    
    choice : the element to subset from the output. Possible values are "quanti_var","quali_var","group"

    Value
    -----
    a dictionnary of dataframes containing the results for the active quantitative/qualitative variable/groups including :

    coord	: coordinates for the quantitatives/qualitatives variable/groups
    
    cos2	: cos2 for the quantitatives/qualitative variable/groups
    
    contrib	: contributions of the quantitatives/qualitative variable/groups

    Usage
    -----

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")
    
    if choice not in ["quanti_var","quali_var","group"]:
        raise ValueError("Error : 'choice' should be one of 'quanti_var','quali_var', 'group'")
    
    if choice == "group":
        return self.group_
    if choice == "quanti_var":
        return self.quanti_var_
    if choice == "quali_var":
        return self.quali_var_
    
def get_mfamix_partial_axes(self):
    """
    Extract the results for partial axes - MFAMIX
    ---------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active 
    partial axes from Multiple Factor Analysis of MIX data (MFAMIX) outputs.

    Parameters
    ----------
    self : an object of class MFAMIX

    Return
    ------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")
    
    return self.partial_axes_

def get_mfamix(self,choice="ind"):
    """
    Extract the results for individuals/variables/group/partial axes - MFAMIX
    -------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals/quantitative 
    variables/groups/partial axes from Multiple Factor Analysis of MIXed data (MFAMIX) outputs.
    
    * get_mfaqual(): Extract the results for variables and individuals
    
    * get_mfaqual_ind(): Extract the results for individuals only
    
    * get_mfaqual_var(): Extract the results for variables (quantitatives, qualitatives and groups)
    
    * get_mfaqual_partial_axes(): Extract the results for partial axes only

    Parameters
    ----------
    self : an object of class MFAQUAL

    choice : he element to subset from the output. Possible values are "ind", "quali_var", "group" or "partial_axes".

    Return
    ------
    a dictionary of dataframes containing the results for the active individuals/qualitative variable groups/partial axes including :

    coord	: coordinates for the individuals/quantitatives/qualitative variable/groups/partial axes
    
    cos2	: cos2 for the individuals/quantitatives/qualitative variable/groups/partial axes

    contrib	: contributions of the individuals/quantitatives/qualitative variable/groups/partial axes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")

    if choice not in ["ind","quanti_var","quali_var","group","partial_axes"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'quanti_var','quali_var', 'group', 'partial_axes'")
    
    if choice == "ind":
        return get_mfamix_ind(self)
    elif choice == "partial_axes":
        return get_mfamix_partial_axes(self)
    else:
        return get_mfamix_var(self,choice=choice)
    
def summaryMFAMIX(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis of MIXed data model
    ------------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFAMIX
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                  Multiple Factor Analysis of MIXed data - Results                   \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    for i in range(ncp):
        group_coord = group["coord"].iloc[:,i]
        group_contrib = group["contrib"].iloc[:,i]
        group_contrib.name = "ctr"
        group_cos2 = group["cos2"].iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if self.ind_sup is not None:
        ind_sup = self.ind_sup_
        ind_sup_infos = ind["dist"]
        for i in range(ncp):
            ind_sup_coord = ind["coord"].iloc[:,i]
            ind_sup_cos2 = ind["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.quanti_var_["coord"].shape[0] > nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
        print("\nContinuous variables\n")
    quanti_var = self.quanti_var_
    quanti_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quanti_var_coord = quanti_var["coord"].iloc[:,i]
        quanti_var_contrib = quanti_var["contrib"].iloc[:,i]
        quanti_var_contrib.name = "ctr"
        quanti_var_cos2 = quanti_var["cos2"].iloc[:,i]
        quanti_var_cos2.name = "cos2"
        quanti_var_infos = pd.concat((quanti_var_infos,quanti_var_coord,quanti_var_contrib,quanti_var_cos2),axis=1)
    quanti_var_infos = quanti_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quanti_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quanti_var_infos)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        if self.quanti_var_sup_ is not None:
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)
    
    # Add continuous variables
    if self.quali_var_["coord"].shape[0] > nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
        print("\nCategories\n")
    quali_var = self.quali_var_
    quali_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quali_var_coord = quali_var["coord"].iloc[:,i]
        quali_var_contrib = quali_var["contrib"].iloc[:,i]
        quali_var_contrib.name = "ctr"
        quali_var_cos2 = quali_var["cos2"].iloc[:,i]
        quali_var_cos2.name = "cos2"
        quali_var_infos = pd.concat((quali_var_infos,quali_var_coord,quali_var_contrib,quali_var_cos2),axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)
    
    # Add correlation ratio
    if self.quali_var_["eta2"].shape[0] > nb_element:
        print(f"\nCategories eta2 (the {nb_element} first)\n")
    else:
        print("\nCategories (eta2)\n")
    quali_var_eta2 = quali_var["eta2"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_eta2)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        # Add supplementary qualitatives variables
        if self.quali_var_sup_ is not None:
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories\n")
            quali_var_sup = self.quali_var_sup_
            quali_var_sup_infos = quali_var_sup["dist"]
            for i in range(ncp):
                quali_var_sup_coord = quali_var_sup["coord"].iloc[:,i]
                quali_var_sup_cos2 = quali_var_sup["cos2"].iloc[:,i]
                quali_var_sup_cos2.name = "cos2"
                quali_var_sup_vtest = quali_var_sup["vtest"].iloc[:,i]
                quali_var_sup_vtest.name = "vtest"
                quali_var_sup_infos = pd.concat((quali_var_sup_infos,quali_var_sup_coord,quali_var_sup_cos2,),axis=1)
            quali_var_sup_infos = quali_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_infos)

            # Add correlation ratio
            if self.quali_var_sup_["eta2"].shape[0] > nb_element:
                print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories (eta2)\n")
            quali_var_sup_eta2 = quali_var_sup["eta2"].iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_eta2)
        
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






############# Hierarchical 

def get_hclust(X, method='single', metric='euclidean', optimal_ordering=False):
    Z = hierarchy.linkage(X,method=method, metric=metric)
    if optimal_ordering:
        order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z,X))
    else:
        order = hierarchy.leaves_list(Z)
    return dict({"order":order,"height":Z[:,2],"method":method,
                "merge":Z[:,:2],"n_obs":Z[:,3],"data":X})


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





###### PCA
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
        data = self.call_["X"]
        row_coord = get_pca_ind(self)["coord"]
        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()

        # Add supplementary continuous variables
        if self.quanti_sup is not None:
            X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
            data = pd.concat([data,X_quanti_sup],axis=1)
    
    
        corrdim = {}
        for idx in row_coord.columns:
            # For continuous variables
            corDim = pd.DataFrame(columns=["correlation","pvalue"]).astype("float")
            for col in data.columns:
                if np.issubdtype(data[col].dtype, np.number): #(data[col].dtypes in ["float64","int64","float32","int32"]):
                    res = st.pearsonr(data[col],row_coord[idx])
                    row_RD = pd.DataFrame({"correlation" : res.statistic,"pvalue":res.pvalue},index = [col])
                    corDim = pd.concat([corDim,row_RD],axis=0)
            # Filter by pvalue
            corDim = corDim.query('pvalue < @proba').sort_values(by="correlation",ascending=False)

            # For categorical variables
            if self.quali_sup is not None:
                quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
                if self.ind_sup is not None:
                    quali_sup = quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
                
                corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','correlation ratio','F-stats','pvalue'])
                for col in quali_sup.columns.tolist():
                    row_RD = pd.DataFrame(eta2(quali_sup[col],row_coord[idx],digits=8),index=[col])
                    corqDim = pd.concat([corqDim,row_RD],axis=0)
                # Filter by pvalue
                corqDim = (corqDim.query('pvalue < @proba').sort_values(by="correlation ratio",ascending=False).rename(columns={"R2": "correlation ratio"}))
            
            if self.quali_sup is None:
                res = corDim
            else:
                if corqDim.shape[0] != 0 :
                    res = {"quanti":corDim,"quali":corqDim}
                else:
                    res = corDim
            
            corrdim[idx] = res
    elif self.model_ == "ca":
        # Extract row coordinates
        row_coord = self.row_["coord"]
        # Exctrac columns coordinates
        col_coord = self.col_["coord"]

        # Add Supplementary row
        if self.row_sup is not None:
            row_sup_coord = self.row_sup_["coord"]
            row_coord = pd.concat([row_coord,row_sup_coord],axis=0)
        
        # Add supplmentary columns
        if self.col_sup is not None:
            col_coord_sup = self.col_sup_["coord"]
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
            corrdim[idx] = {"row" : (row_coord[idx].to_frame().sort_values(by=idx,ascending=True).rename(columns={idx:"coord"})),
                            "col" : (col_coord[idx].to_frame().sort_values(by=idx,ascending=True).rename(columns={idx:"coord"}))}
    elif self.model_ == "mca":
        # Select data
        data = self.call_["X"]
        # Extract individuals coordinates
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()

        if self.quali_sup is not None:
            X_quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
                data = pd.concat([data,X_quali_sup],axis=1)

        corrdim = {}
        for idx in ind_coord.columns:
            # Pearson correlation test
            if self.quanti_sup is not None:
                X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
                if self.ind_sup is not None:
                    X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
                
                corDim = pd.DataFrame(columns=["statistic","pvalue"]).astype("float")
                for col in X_quanti_sup.columns.tolist():
                    if (X_quanti_sup[col].dtypes in ["float64","int64","float32","int32"]):
                        res = st.pearsonr(X_quanti_sup[col],ind_coord[idx])
                        row_RD = pd.DataFrame({"statistic" : res.statistic,"pvalue":res.pvalue},index = [col])
                        corDim = pd.concat([corDim,row_RD])
                # Filter by pvalue
                corDim = (corDim.query('pvalue < @proba').sort_values(by="statistic",ascending=False))

            # Correlation ratio (eta2)
            corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','correlation ratio','F-stats','pvalue']).astype("float")
            for col in data.columns.tolist():
                row_RD = pd.DataFrame(eta2(data[col],ind_coord[idx],digits=8),index=[col])
                corqDim = pd.concat([corqDim,row_RD])
            # Filter by pvalue
            corqDim = (corqDim.query('pvalue < @proba').sort_values(by="correlation ratio",ascending=False).rename(columns={"correlation ratio" : "R2"}))
        
            if self.quanti_sup is None:
                res = corqDim
            else:
                if corDim.shape[0] != 0 :
                    res = {"quali":corqDim,"quanti":corDim}
                else:
                    res = corqDim
            corrdim[idx] = res
    elif self.model_ == "famd":
        # Extract row coord
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()
        
        # Select continuous active data
        quanti_data = self.call_["X"][self.quanti_var_["coord"].index.tolist()]
        # Add supplementary continuous variables
        if self.quanti_sup is not None:
            X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
            quanti_data = pd.concat([quanti_data,X_quanti_sup],axis=1)

        # Select categorical active variables
        quali_data = self.call_["X"].drop(columns=quanti_data.columns.tolist())
        # Add supplementary categorical variables
        if self.quali_sup is not None:
            X_quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
            quali_data = pd.concat([quali_data,X_quali_sup],axis=1)
        
        # Correlation between coninuous variable and axis
        corrdim = {}
        for idx in ind_coord.columns:
            # For continuous variables
            corDim = pd.DataFrame(columns=["correlation","pvalue"]).astype("float")
            for col in quanti_data.columns.tolist():
                if (quanti_data[col].dtypes in ["float64","int64","float32","int32"]):
                    res = st.pearsonr(quanti_data[col],ind_coord[idx])
                    row_RD = pd.DataFrame({"correlation" : res.statistic,"pvalue":res.pvalue},index = [col])
                    corDim = pd.concat([corDim,row_RD],axis=0)
            # Filter by pvalue
            corDim = (corDim.query('pvalue < @proba').sort_values(by="correlation",ascending=False))

            # For categorical variable    
            corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','correlation ratio','F-stats','pvalue'])
            for col in quali_data.columns.tolist():
                row_RD = pd.DataFrame(eta2(quali_data[col],ind_coord[idx],digits=8),index=[col])
                corqDim = pd.concat([corqDim,row_RD],axis=0)
            # Filter by pvalue
            corqDim = (corqDim.query('pvalue < @proba').sort_values(by="correlation ratio",ascending=False).rename(columns={"correlation ratio" : "R2"}))

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
    ----------------------

    This function reconstructs a data set from the result of a PCA 

    Parameters:
    -----------
    self : an object of class PCA

    n_components : int, the number of dimensions to use to reconstitute data.

    Return
    ------
    X : Reconstitution data.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "pca":
        raise ValueError("Error : 'self' must be an object of class PCA.")
    if n_components is not None:
        if n_components > self.call_["n_components"]:
            raise ValueError("Error : Enter good number of n_components" )
    else:
        raise ValueError("Error : 'n_components' must be pass.")
    
    # Valeurs centrées
    Z = np.dot(self.ind_["coord"].iloc[:,:n_components],self.svd_["V"][:,:n_components].T)
    
    #np.dot(self.row_coord_[:,:n_components],self.eigen_vectors_[:,:n_components].T)

    # Déstandardisation et décentrage
    X = self.call_["X"].copy()
    for k in np.arange(self.var_["coord"].shape[0]):
        X.iloc[:,k] = Z[:,k]*self.call_["std"].values[k] + self.call_["means"].values[k]
    
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