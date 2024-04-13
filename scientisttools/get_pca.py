# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_pca_ind(self,choice="ind") -> dict:
    """
    Extract the results for individuals - PCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions) for the active/supplementary 
    individuals from Principal Component Analysis (PCA) outputs.

    Parameters:
    -----------
    self : an object of class PCA

    choice : the element to subset from the output. Allowed values are "ind" (for active individuals) or "ind_sup" (for supplementary individuals).

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
        raise ValueError("'self' must be an object of class PCA.")
    
    if choice not in ["ind", "ind_sup"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup'")
    
    if choice == "ind":
        return  self.ind_
    if choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("No supplementary individuals")
        return self.ind_sup_

def get_pca_var(self,choice="var") -> dict:

    """
    Extract the results for variables - PCA
    ---------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions) for the active/supplementary 
    variables from Principal Component Analysis (PCA) outputs.

    Parameters
    ----------
    self : an object of class PCA

    choice : the element to subset from the output. Allowed values are "var" (for active variables), "quanti_sup" (for supplementary quantitatives variables) or
                "quali_sup" (for supplementary categorical variables)

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
        raise ValueError("'self' must be an object of class PCA")
    
    if choice not in ["var", "quanti_sup","quali_sup"]:
        raise ValueError("'choice' should be one of 'var', 'quanti_sup', 'quali_sup'")
    
    if choice == "var":
        return self.var_
    if choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("No supplementary quantitatives variables")
        return self.quanti_sup_  
    if choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("No supplementary categorical variables")
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
        raise ValueError("'self' must be an object of class PCA.")

    if choice not in ["ind","ind_sup","var","quanti_sup","quali_sup"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup', 'var', 'quanti_var', 'quali_var'")

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
    self        :   an obect of class PCA

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
        raise ValueError("'self' must be an object of class PCA")

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
    if self.var_["coord"].shape[0]>nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
         print("\nContinuous variables\n")
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