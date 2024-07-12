# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_pca_ind(self) -> dict:
    """
    Extract the results for individuals - PCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) of the active individuals from Principal Component Analysis (PCA) outputs.

    Usage
    -----
    ```python
    >>> get_pca_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    Returns
    -------
    dictionary of dataframes containing all the results for the active individuals including:

    `coord` : factor coordinates (scores) of the individuals

    `cos2` : square cosinus of the individuals

    `contrib` : relative contributions of the individuals

    `infos` : additionals informations (weight, squared distance to origin and inertia) of the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load decathlon2 dataset
    >>> from scientisttools import load_decathlon2
    >>> X = decathlon2()
    >>> from scientisttools import PCA, get_pca_ind
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Extract the results for individuals
    >>> ind = get_pca_ind(res_pca) 
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    return self.ind_

def get_pca_var(self) -> dict:
    """
    Extract the results for variables - PCA
    ---------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, contributions) for the active variables from Principal Component Analysis (PCA) outputs

    Usage
    -----
    ```python
    >>> get_pca_var(self)
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    Returns
    -------
    dictionary of dataframes containing all the results for the active variables including:

    `coord` : factor coordinates (scores) of the variables

    `cor` : correlation between variables and axes of the variables

    `contrib` : relative contributions of the variables

    `cos2` : square cosinus of the variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load decathlon2 dataset
    >>> from scientisttools import load_decathlon2
    >>> X = decathlon2()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Extract the results for variables
    >>> var = get_pca_var(res_pca)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")
    return self.var_

def get_pca(self,choice="ind")-> dict:
    """
    Extract the results for individuals/variables - PCA
    ---------------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) for the active individuals/variables from Principal Component Analysis (PCA) outputs.

        * get_pca() : Extract the results for variables and individuals
        * get_pca_ind() : Extract the results for individuals only
        * get_pca_var() : Extract the results for variables only

    Usage
    -----
    ```python
    >>> get_pca(self,choice=("ind","var"))
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    `choice` : the element to subset from the output. Allowed values are :
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    dictionary of dataframes containing all the results for the active individuals/variables including:

    `coord` : factor coordinates (scores) of the individuals/variables
    
    `cos2` : square cosinus of the individuals/variables
    
    `contrib` : relative contributions of the individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load decathlon2 dataset
    >>> from scientisttools import load_decathlon2
    >>> X = decathlon2()
    >>> from scientisttools import PCA, get_pca
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> # Extract the results for individuals
    >>> ind = get_pca(res_pca,choice = "ind")
    >>> # Extract the results for variables
    >>> var = get_pca(res_pca,choice = "var") 
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")

    if choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'ind','var'")

    if choice == "ind":
        return get_pca_ind(self)
    else:
        return get_pca_var(self)

def summaryPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Principal Component Analysis objects
    ----------------------------------------------------------

    Description
    -----------
    Printing summaries of principal component analysis (PCA) objects

    Usage
    -----
    ```python
    >>> summaryPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class PCA

    `digits` : int, default=3. Number of decimal printed

    `nb_element` : int, default = 10. Number of element

    `ncp` : int, default = 3. Number of components

    `to_markdown` : Print DataFrame in Markdown-friendly format.

    `tablefmt` : Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs` : These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load decathlon2 dataset
    >>> from scientisttools import load_decatlon2
    >>> X = load_decathlon2()
    >>> from scientisttools import PCA, summaryPCA
    >>> res_pca = PCA(standardize=True,n_components=None,ind_sup=list(range(23,X.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(X)
    >>> summaryPCA(res_pca)
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "pca":
        raise TypeError("'self' must be an object of class PCA")

    # Define number of components
    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])

    # Principal Components Analysis Results
    print("                     Principal Component Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Eigenvalues")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Bartlette's test of spericity
    print("\nBartlett's test of sphericity\n")
    bartlett = self.others_["bartlett"]
    if to_markdown:
        print(bartlett.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(bartlett)

    # Add individuals informations
    ind = self.ind_
    if ind["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_infos = ind["infos"]
    for i in np.arange(ncp):
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
        if ind_sup["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup_infos = ind_sup["dist"]
        for i in np.arange(ncp):
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
    var = self.var_
    if var["coord"].shape[0]>nb_element:
        print(f"\nVariables (the {nb_element} first)\n")
    else:
         print("\nVariables\n")
    var_infos = pd.DataFrame().astype("float")
    for i in np.arange(ncp):
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
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup["coord"].shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
        quanti_sup_infos = pd.DataFrame().astype("float")
        for i in np.arange(ncp):
            quanti_sup_coord = quanti_sup["coord"].iloc[:,i]
            quanti_sup_cos2 = quanti_sup["cos2"].iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos =pd.concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup["coord"].shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup["dist"]
        for i in np.arange(ncp):
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
        if quali_sup["eta2"].shape[0] > nb_element:
            print(f"\nSupplementary categorical variable (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary categorical variable (eta2)\n")
        quali_sup_eta2 = quali_sup["eta2"].iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)