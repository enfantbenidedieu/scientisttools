# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_efa_ind(self) -> dict:
    """
    Extract the results for individuals - EFA
    -----------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates) of the active individuals from Exploratory Factor Analysis (EFA) outputs.

    Usage
    -----
    ```python
    >>> get_efa_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class EFA

    Returns
    -------
    dictionary of dataframes containing all the results for the active individuals including:

    `coord` : factor coordinates (scores) of the individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")
    return self.ind_

def get_efa_var(self) -> dict:
    """
    Extract the results for variables - EFA
    ---------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates,) for the active variables from Exploratory Factor Analysis (EFA) outputs

    Usage
    -----
    ```python
    >>> get_efa_var(self)
    ```

    Parameters
    ----------
    `self` : an instance of class EFA

    Returns
    -------
    dictionary of dataframes containing all the results for the active variables including:

    `coord` : factor coordinates (scores) of the variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> 
    ```
    """
    # Check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")
    return self.var_

def get_efa(self,choice = "ind")-> dict:
    """
    Extract the results for individuals/variables - EFA
    ---------------------------------------------------

    Description
    -----------
    Extract all the results for the active individuals/variables from Exploratory Factor Analysis (EFA) outputs.

        * get_efa() : Extract the results for variables and individuals
        * get_efa_ind() : Extract the results for individuals only
        * get_efa_var() : Extract the results for variables only

    Usage
    -----
    ```python
    >>> get_efa(self,choice=("ind","var"))
    ```

    Parameters
    ---------
    `self` : an instance of class EFA

    `choice` : the element to subset from the output. Allowed values are :
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    dictionary of dataframes containing all the results for the active individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> 
    ```
    """
    # Check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")

    if choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'ind', 'var'")
    
    if choice == "ind":
        return get_efa_ind(self)
    elif choice == "var":
        return get_efa_var(self)

def summaryEFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Exploratory Factor Analysis model
    -------------------------------------------------------

    Description
    -----------
    Printing summaries of exploratory factor analysis (PCA) objects

    Usage
    -----
    ```python
    >>> summaryEFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class EFA

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
    >>> 
    ```
    """
    # check if self is an object of class EFA
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])

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
    ind = self.ind_
    if ind["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_coord = ind["coord"].iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(ind_coord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_coord)

    # Add supplementary individuals
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind_sup["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        # Save all informations
        ind_sup_coord = ind_sup["coord"].iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(ind_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_coord)

    # Add variables informations
    var = self.var_
    if var["coord"].shape[0]>nb_element:
        print(f"\nVariables (the {nb_element} first)\n")
    else:
         print("\nVariables\n")
    var_infos = pd.DataFrame().astype("float")
    for i in np.arange(ncp):
        var_coord = var["coord"].iloc[:,i]
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr"
        var_infos = pd.concat([var_infos,var_coord,var_ctr],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)