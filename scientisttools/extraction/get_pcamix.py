# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_pcamix_ind(self) -> dict:
    """
    Extract the results for individuals - PCAMIX
    --------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus and relative contributions) for the individuals from Principal Component Analysis of Mixed Data (PCAMIX) outputs.

    Usage
    -----
    ```python
    >>> get_pcamix_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class PCAMIX

    Returns
    -------
    dictionary of dataframes containing the results for the individuals, including :

    `coord` : factor coordinates of individuals.

    `cos2`	: square cosinus representing the quality of representation on the factor map.
    
    `contrib` : relative contributions of individuals to the principal components.

    `infos` : additionals informations (weight, squared distance to origin and inertia) of the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load gironde dataset
    >>> from scientisttools import load_gironde
    >>> gironde = load_gironde()
    >>> from scientisttools import PCAMIX, get_pcamix_ind
    >>> res_pcamix = PCAMIX().fit(gironde)
    >>> # results for individuals
    >>> ind = get_pcamix_ind(res_pcamix)
    ```
    """
    # Check if self is an object of class PCAMIX
    if self.model_ != "pcamix":
        raise ValueError("'self' must be an object of class PCAMIX")
    return self.ind_
    
def get_pcamix_var(self, choice="var") -> dict:
    """
    Extract the results for variables - PCAMIX
    ------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosinus and relative contributions) for quantitative and qualitative variables from Principal Component Analysis of Mixed Data (PCAMIX) outputs.

    Usage
    -----
    ```
    >>> get_pcamix_var(self,choice=("var","quanti_var", "quali_var"))
    ```

    Parameters
    ----------
    `self` : an object of class PCAMIX

    `choice` : the element to subset from the output. Possible values are :
        * "var" for active variables
        * "quanti_var" for active quantitatives variables
        * "quali_var" for active qualitatives variables (categories)

    Returns
    -------
    dictionary of dataframes containing the results for the active variables, including :

    `coord`	: factor coordinates of variables.

    `cos2` : squared cosinus values representing the quality of representation on the factor map.

    `contrib` : relative contributions of variables to the principal components.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load gironde dataset
    >>> from scientisttools import load_gironde
    >>> gironde = load_gironde()
    >>> from scientisttools import PCAMIX, get_pcamix_var
    >>> res_pcamix = PCAMIX().fit(gironde)
    >>> # Results for quantitatives variables
    >>> quanti_var = get_pcamix_var(res_pcamix, choice = "quanti_var")
    >>> # Results for qualitatives variables
    >>> quali_var = get_pcamix_var(res_pcamix, choice = "quali_var")
    >>> # Results for variables
    >>> var = get_pcamix_var(res_pcamix, choice = "var")
    ```
    """
    # Check if self is an object of class PCAMIX
    if self.model_ != "pcamix":
        raise ValueError("'self' must be an object of class PCAMIX")
    
    if choice not in ["quanti_var","quali_var","var"]:
        raise ValueError("'choice' should be one of 'quanti_var', 'quali_var', 'var'")
    
    if choice == "quanti_var":
        if not hasattr(self,"quanti_var_"):
            raise ValueError("No quantitatives columns")
        return self.quanti_var_
    
    if choice == "quali_var":
        if not hasattr(self,"quali_var_"):
            raise ValueError("No qualitatives columns")
        return self.quali_var_
    
    if choice == "var":
        if not hasattr(self,"var_"):
            raise ValueError("No mixed data")
        return self.var_

def get_pcamix(self,choice = "ind")-> dict:
    """
    Extract the results for individuals and variables - PCAMIX
    ----------------------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosine and relative contributions) for the individuals and variables from Principal Component Analysis of Mixed Data (PCAMIX) outputs.

    Usage
    -----
    ```python
    >>> get_pcamix(self, choice = ("ind", "var", "quanti_var", "quali_var"))
    ```

    Parameters
    ----------
    `self` : an object of class PCAMIX

    `choice` : the element to subset from the output. Possibles values are :
        * "ind" for individuals
        * "var" for active variables
        * "quanti_var" for active quantitatives variables
        * "quali_var" for active qualitatives variables (categories)

    Return
    ------
    dictionary of dataframes containing the results for the active individuals and variables, including :
    
    `coord` : factor coordinates of individuals/variables.
    
    `cos2` : square cosinus values representing the quality of representation on the factor map.
    
    `contrib` : relative contributions of individuals/variables to the principal components.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load gironde dataset
    >>> from scientisttools import load_gironde
    >>> gironde = load_gironde()
    >>> from scientisttools import PCAMIX, get_pcamix
    >>> res_pcamix = PCAMIX().fit(gironde)
    >>> # Results for individuals
    >>> ind = get_pcamix(res_pcamix,choice = "ind")
    >>> # Results for quantitatives variables
    >>> quanti_var = get_pcamix(res_pcamix, choice = "quanti_var")
    >>> # Results for qualitatives variables
    >>> quali_var = get_pcamix(res_pcamix, choice = "quali_var")
    >>> # Results for variables
    >>> var = get_pcamix(res_pcamix, choice = "var")
    ```
    """
    # Check if self is an object of class PCAMIX
    if self.model_ != "pcamix":
        raise ValueError("'self' must be an object of class PCAMIX")
    
    if choice not in ["ind","quanti_var","quali_var","var"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup', 'quanti_var', 'quali_var', 'var'")
    
    if choice == "ind":
        return get_pcamix_ind(self)
    else:
        return get_pcamix_var(self,choice=choice)

def summaryPCAMIX(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Principal Component Analysis of Mixed Data model
    ----------------------------------------------------------------------

    Description
    -----------
    Printing summaries of principal component analysis of mixed data objects

    Usage
    -----
    ```python
    >>> summaryPCAMIX(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class PCAMIX

    `digits` : int, default=3. Number of decimal printed

    `nb_element` : int, default = 10. Number of element

    `ncp` : int, default = 3. Number of components

    `to_markdown` : Print DataFrame in Markdown-friendly format

    `tablefmt` : Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs` : These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load gironde dataset
    >>> from scientisttools import load_gironde
    >>> gironde = load_gironde()
    >>> from scientisttools import PCAMIX, summaryPCAMIX
    >>> res_pcamix = PCAMIX().fit(gironde)
    >>> summaryPCAMIX(res_pcamix)
    ```
    """
    # check if self is an object of class PCAMIX
    if self.model_ != "pcamix":
        raise ValueError("'self' must be an object of class PCAMIX")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])

    print("                Principal Component Analysis of Mixed Data - Results                 \n")
    
    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    ind = self.ind_
    if ind["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
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
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup_infos = ind_sup["dist"]
        for i in np.arange(0,ncp,1):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat([ind_sup_infos,ind_sup_coord,ind_sup_cos2],axis=1)
        ind_sup_infos = ind_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    # Add variables informations
    if hasattr(self,"quanti_var_"):
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
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup["coord"].shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
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
    
    if hasattr(self,"quali_var_"):
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
    
        # Add categoricals variables square correlation ratio
        if hasattr(self,"quanti_var_"):
            quali_var_eta2 = self.var_["coord"].drop(index=self.quanti_var_["coord"].index)
        else:
            quali_var_eta2 = self.quali_var_["eta2"]

        if quali_var_eta2.shape[0] > nb_element:
            print(f"\nCategoricals variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nCategoricals variables (eta2)\n")
        quali_var_eta2 = quali_var_eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_var_eta2)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup["coord"].shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
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
        if quali_sup["eta2"].shape[0] > nb_element:
            print(f"\nSupplementary categorical variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary categorical variables (eta2)\n")
        quali_sup_eta2 = self.quali_sup_["eta2"].iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)