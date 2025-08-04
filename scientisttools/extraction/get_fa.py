# -*- coding: utf-8 -*-
from pandas import concat
from typing import NamedTuple

def get_fa_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - FactorAnalysis
    ----------------------------------------------------

    Description
    -----------
    Extract all the results of the active individuals from Factor Analysis (FactorAnalysis) outputs.

    Usage
    -----
    ```python
    >>> get_fa_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class FactorAnalysis

    Returns
    -------
    a namedtuple of pandas Dataframe containing all the results for the active individuals including:

    `coord`: factor coordinates
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, get_fa_ind
    >>> res_fa = FactorAnalysis(rotation=None,max_iter=1).fit(beer)
    >>> #extract the results for individuals
    >>> ind = get_fa_ind(res_fa)
    >>> ind.coord.head() # coordinates of individuals
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")
    return self.ind_

def get_fa_var(self) -> NamedTuple:
    """
    Extract the results for variables - FactorAnalysis
    --------------------------------------------------

    Description
    -----------
    Extract all the results for the active variables from Factor Analysis (FactorAnalysis) outputs

    Usage
    -----
    ```python
    >>> get_fa_var(self)
    ```

    Parameters
    ----------
    `self`: an instance of class FactorAnalysis

    Returns
    -------
    a namedtuple of pandas Dataframes containing all the results for the active variables including:

    `coord`: factor coordinates

    `contrib`: relative contribution

    `f_score`: normalized factor coefficients
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, get_fa_var
    >>> res_fa = FactorAnalysis(rotation=None,max_iter=1).fit(beer)
    >>> #extract the results for variables
    >>> var = get_fa_var(res_fa)
    >>> var.coord.head() # coordinates of variables (loadings)
    >>> var.contrib.head() # contributions of variables
    >>> var.f_score.head() # projection function
    ```
    """
    # Check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")
    return self.var_

def get_fa(self, element= "ind")-> NamedTuple:
    """
    Extract the results for individuals/variables - FactorAnalysis
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
    `self`: an instance of class FactorAnalysis

    `element`: the element to subset from the output. Allowed values are :
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    namedtuple of pandas Dataframes containing all the results for the active individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import beer, FactorAnalysis, get_fa
    >>> res_fa = FactorAnalysis(rotation=None,max_iter=1).fit(beer)
    >>> #extract the results for individuals
    >>> ind = get_fa(res_fa,element="ind")
    >>> ind.coord.head() # coordinates of individuals
    >>> #extract the results for variables
    >>> var = get_fa(res_fa,element="var")
    >>> var.coord.head() # coordinates of variables (loadings)
    >>> var.contrib.head() # contributions of variables
    >>> var.f_score.head() # projection function  
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
    Printing summaries of factor analysis (FactorAnalysis) objects

    Usage
    -----
    ```python
    >>> summaryFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FactorAnalysis

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
    >>> from scientisttools import beer, FactorAnalysis, summaryFA
    >>> #non iterative principal factor analysis
    >>> res_fa = FactorAnalysis(rotation=None,max_iter=1).fit(beer)
    >>> summaryFA(res_fa)
    ```
    """
    # check if self is an object of class FactorAnalysis
    if self.model_ != "fa":
        raise TypeError("'self' must be an object of class FactorAnalysis")

    ncp = min(ncp,self.call_.n_components)
    nb_element = min(nb_element,self.call_.X.shape[0])

    #Principa l Factor Analysis Results
    print("                     Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    eig = self.eig_.iloc[:self.call_.n_components,:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)

    #variance accounted
    print("\nVariance accounted\n")
    vaccounted = self.vaccounted_.T.round(decimals=digits)
    if to_markdown:
        print(vaccounted.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(vaccounted)
    
    # Add individuals informations
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

    # Add supplementary individuals
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        # Save all informations
        ind_sup_coord = ind_sup.coord.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(ind_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_coord)

    # Add variables informations
    var = self.var_
    if var.coord.shape[0]>nb_element:
        print(f"\nVariables (the {nb_element} first)\n")
    else:
         print("\nVariables\n")
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