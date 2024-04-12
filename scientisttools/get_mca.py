# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

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
    self        :   an object of class MCA

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
        raise ValueError("'self' must be an object of class MCA")

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
