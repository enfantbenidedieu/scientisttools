# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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
        raise ValueError("'self' must be an object of class FAMD")
    
    if choice not in ["ind","ind_sup"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup'")
    
    if choice == "ind":
        return self.ind_
    elif choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("No supplementary individuals")
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
        raise ValueError("'self' must be an object of class FAMD")
    
    if choice not in ["quanti_var","quali_var","var","quanti_sup","quali_sup","var_sup"]:
        raise ValueError("'choice' should be one of 'quanti_var', 'quali_var', 'var', 'quanti_sup', 'quali_sup', 'var_sup'")
    
    if choice == "quanti_var":
        return self.quanti_var_
    elif choice == "quali_var":
        return self.quali_var_
    elif choice == "var":
        return self.var_
    elif choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("No supplementary quantitatives columns")
        return self.quanti_sup_
    elif choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("No supplementary qualitatives columns")
    elif choice == "var_sup":
        if self.quanti_sup is None or self.quali_sup is None:
            raise ValueError("No supplementary columns")
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
        raise ValueError("'self' must be an object of class FAMD")
    
    if choice not in ["ind","ind_sup","quanti_var","quali_var","var","quanti_sup","quali_sup","var_sup"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup', 'quanti_var', 'quali_var', 'var', 'quanti_sup', 'quali_sup', 'var_sup'")
    

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
    self        :   an obect of class FAMD

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    # check if famd model
    if self.model_ != "famd":
        raise ValueError("'self' must be an object of class FAMD")

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