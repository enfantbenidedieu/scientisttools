# -*- coding: utf-8 -*-
from pandas import concat
from typing import NamedTuple

def get_fa_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - FA
    ----------------------------------------

    Description
    -----------
    Extract all the results of the active individuals from Factor Analysis (FA) outputs.

    Usage
    -----
    ```python
    >>> get_fa_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class FA

    Returns
    -------
    a namedtuple of pandas Dataframe containing all the results for the active individuals including:

    `coord`: coordinates of the individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, FA, get_fa_ind
    >>> beer = load_dataset("beer")
    >>> res_fa = FA(n_components=2,max_iter=1)
    >>> res_fa.fit(beer)
    >>> #extract the results for individuals
    >>> ind = get_fa_ind(res_fa)
    >>> ind.coord.head() # coordinates of the individuals
    ```
    """
    if self.model_ != "fa": #check if self is an object of class FA
        raise TypeError("'self' must be an object of class FA")
    return self.ind_

def get_fa_var(self) -> NamedTuple:
    """
    Extract the results for variables - FA
    --------------------------------------

    Description
    -----------
    Extract all the results for the active variables from Factor Analysis (FA) outputs

    Usage
    -----
    ```python
    >>> get_fa_var(self)
    ```

    Parameters
    ----------
    `self`: an instance of class FA

    Returns
    -------
    a namedtuple of pandas Dataframes containing all the results for the active variables including:

    `coord`: loadings or coordinates of the variables,

    `contrib`: contribution of the variables

    `f_score`: factor score coefficients of the variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, FA, get_fa_var
    >>> beer = load_dataset("beer")
    >>> res_fa = FA(max_iter=1)
    >>> res_fa.fit(beer)
    >>> #extract the results for variables
    >>> var = get_fa_var(res_fa)
    >>> var.coord #coordinates of variables (loadings)
    >>> var.contrib #contributions of variables
    >>> var.f_score #projection function
    ```
    """
    if self.model_ != "fa": #check if self is an object of class FA
        raise TypeError("'self' must be an object of class FA")
    return self.var_

def get_fa(self, element= "ind")-> NamedTuple:
    """
    Extract the results for individuals/variables - FA
    --------------------------------------------------------------

    Description
    -----------
    Extract all the results for the active individuals/variables from Factor Analysis (FactorAnalysis) outputs.

        * `get_fa()`: Extract the results for variables and individuals
        * `get_fa_ind()`: Extract the results for individuals only
        * `get_fa_var()`: Extract the results for variables only

    Usage
    -----
    ```python
    >>> get_fa(self, element = ("ind","var"))
    >>> get_fa(self, element = "ind")
    >>> get_fa(self, element = "var")
    ```

    Parameters
    ---------
    `self`: an instance of class FA

    `element`: the element to subset from the output. Allowed values are :
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    a namedtuple of pandas Dataframes containing all the results for the active individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, FA, get_fa
    >>> beer = load_dataset("beer")
    >>> res_fa = FA(n_components=2,max_iter=1)
    >>> res_fa.fit(beer)
    >>> #extract the results for individuals
    >>> ind = get_fa(res_fa,"ind")
    >>> ind.coord # coordinates of individuals
    >>> #extract the results for variables
    >>> var = get_fa(res_fa, "var")
    >>> var.coord #coordinates of variables (loadings)
    >>> var.contrib #contributions of variables
    >>> var.f_score #projection function  
    ```
    """
    if element == "ind":
        return get_fa_ind(self)
    elif element == "var":
        return get_fa_var(self)
    else:
        ValueError("'choice' should be one of 'ind', 'var'")

def summaryFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Factor Analysis model
    -------------------------------------------

    Description
    -----------
    Printing summaries of factor analysis (FA) objects

    Usage
    -----
    ```python
    >>> summaryFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FA

    `digits`: int, default=3. Number of decimal printed

    `nb_element`: int, default = 10. Number of element

    `ncp`: int, default = 3. Number of components

    `to_markdown`: print DataFrame in Markdown-friendly format.

    `tablefmt`: Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs`: These parameters will be passed to tabulate.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, FA, summaryFA
    >>> beer = load_dataset("beer")
    >>> #non iterative principal factor analysis
    >>> res_fa = FA(n_components=2,max_iter=1).fit(beer)
    >>> summaryFA(res_fa)
    ```
    """
    if self.model_ != "fa": #check if self is an object of class FA
        raise TypeError("'self' must be an object of class FA")

    #set number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    #Factor Analysis Results
    print("                     Factor Analysis - Results                     \n")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add eigenvalues informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    eig = self.eig_.iloc[:self.call_.n_components,:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #variance accounted
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nVariance accounted\n")
    vaccounted = self.vaccounted_.round(decimals=digits)
    if to_markdown:
        print(vaccounted.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(vaccounted)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ind = self.ind_
    if ind.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_coord = ind.coord.iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(ind_coord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_coord)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary individuals
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind_sup.coord.shape[0] > nb_element:
            print("\nSupplementary individuals (the {} first):".format(nb_element))
        else:
            print("\nSupplementary individuals:")
        # Save all informations
        ind_sup_coord = ind_sup.coord.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(ind_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_coord)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    var = self.var_
    if var.coord.shape[0]>nb_element:
        print("\nVariables (the {} first):".format(nb_element))
    else:
         print("\nVariables:")
    var_infos = self.others_.communality
    for i in range(ncp):
        var_coord, var_ctr = var.coord.iloc[:,i], var.contrib.iloc[:,i]
        var_ctr.name = "ctr"
        var_infos = concat([var_infos,var_coord,var_ctr],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)