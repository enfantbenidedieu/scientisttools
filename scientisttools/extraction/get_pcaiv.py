# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_pcaiv_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - PCAiv
    -------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals from Principal Component Analysis with Instrumental Variables (PCAiv) outputs.

    Usage
    -----
    ```python
    >>> get_pcaiv_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class PCAiv

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals, including:

    `coord`: coordinates of the individuals,

    `contrib`: relative contributions of the individuals,

    `cos2`: squared cosinus of the individuals,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import rhone
    >>> from scientisttools import PCAiv, get_pcaiv_ind
    >>> res_pcaiv = PCAiv(iv=(15,16,17))
    >>> res_pcaiv.fit(rhone)
    >>> #extract the results for individuals
    >>> ind = get_pcaiv_ind(res_pcaiv)
    >>> ind.coord #coordinates of individuals
    >>> ind.contrib #contributions of individuals
    >>> ind.cos2 #cos2 of individuals
    >>> ind.infos #additionals informations of individuals
    ```
    """
    if self.model_ != "pcaiv": #check if self is an object of class PCAiv
        raise TypeError("'self' must be an object of class PCAiv")
    return self.ind_

def get_pcaiv_var(self) -> NamedTuple:
    """
    Extract the results for variables - PCAiv
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Principal Component Analysis with Instrumental Variables (PCAiv) outputs

    Usage
    -----
    ```python
    >>> get_pcaiv_var(self)
    ```

    Parameters
    ----------
    `self`: an object of class PCAiv

    Returns
    -------
    namedtuple of pandas DataFrames containing all the results for the active variables including:

    `coord`: coordinates of the variables,

    `contrib`: relative contributions of the variables,

    `cos2`: squared cosinus of the variables,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import rhone
    >>> from scientisttools import PCAiv, get_pcaiv_var
    >>> res_pcaiv = PCAiv(iv=(15,16,17))
    >>> res_pcaiv.fit(rhone)
    >>> #extract the results for variables
    >>> var = get_pcaiv_var(res_pcaiv)
    >>> var.coord #coordinates of variables
    >>> var.contrib #contributions of variables
    >>> var.cos2 #cos2 of variables
    >>> var.infos #additionals informations of variables
    ```
    """
    if self.model_ != "pcaiv": #check if self is an object of class PCAiv
        raise TypeError("'self' must be an object of class PCAiv")
    return self.var_

def get_pcaiv(self,element="ind")-> NamedTuple:
    """
    Extract the results for individuals/variables - PCAIV
    -----------------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables from Principal Component Analysis with Instrumental Variables (PCAIV) outputs.

        * `get_pcaiv()`: Extract the results for variables and individuals
        * `get_pcaiv_ind()`: Extract the results for individuals only
        * `get_pcaiv_var()`: Extract the results for variables only

    Usage
    -----
    ```python
    >>> get_pcaiv(self,element="ind")
    >>> get_pcaiv(self,element="var")
    ```

    Parameters
    ----------
    `self`: an object of class PCAiv

    `element`: a string indicating the element to subset from the output. Allowed values are:
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals/variables, including:

    `coord`: coordinates of the individuals/variables,
    
    `contrib`: relative contributions of the individuals/variables,

    `cos2`: squared cosinus of the individuals/variables,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals/variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import rhone
    >>> from scientisttools import PCAIV, get_pcaiv
    >>> res_pcaiv = PCAIV(iv=(15,16,17))
    >>> res_pcaiv.fit(rhone)
    >>> #extract the results for individuals
    >>> ind = get_pcaiv(res_pcaiv, "ind")
    >>> ind.coord #coordinates of individuals
    >>> ind.contrib #contributions of individuals
    >>> ind.cos2 #cos2 of individuals
    >>> ind.infos #additionals informations of individuals
    >>> #extract the results for variables
    >>> var = get_pcaiv(res_pcaiv, "var") 
    >>> var.coord #coordinates of variables
    >>> var.contrib #contributions of variables
    >>> var.cos2 #cos2 of variables
    >>> var.infos #additionals informations of variables
    ```
    """
    if element == "ind":
        return get_pcaiv_ind(self)
    elif element == "var":
        return get_pcaiv_var(self)
    else:
        raise ValueError("'element' should be one of 'ind', 'var'")

def summaryPCAiv(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Principal Component Analysis with Instrumental Variables objects
    --------------------------------------------------------------------------------------

    Description
    -----------
    Printing summaries of principal component analysis with instrumental variables (PCAiv) objects

    Usage
    -----
    ```python
    >>> summaryPCAiv(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCAiv

    `digits`: int, default=3. Number of decimal printed

    `nb_element`: int, default = 10. Number of element

    `ncp`: int, default = 3. Number of components

    `to_markdown`: Print DataFrame in Markdown-friendly format.

    `tablefmt`: Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs`: These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import rhone
    >>> from scientisttools import PCAIV, summaryPCAIV
    >>> res_pcaiv = PCAIV(iv=(15,16,17))
    >>> res_pcaiv.fit(rhone)
    >>> summaryPCAIV(res_pcaiv)
    ```
    """
    if self.model_ != "pcaiv": #check if self is an object of class PCAiv
        raise TypeError("'self' must be an object of class PCAiv")

    #define number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    # Principal Components Analysis Results
    print("         Principal Component Analysis with Instrumental Variables - Results               \n")

    # Add eigenvalues informations
    print("Eigenvalues")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ind = self.ind_
    if ind.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_infos = self.ind_.infos
    for i in range(ncp):
        ind_coord, ind_sqcos, ind_ctr = ind.coord.iloc[:,i], ind.cos2.iloc[:,i], ind.contrib.iloc[:,i]
        ind_sqcos.name, ind_ctr.name = "cos2", "ctr"
        ind_infos = concat((ind_infos,ind_coord,ind_ctr,ind_sqcos),axis=1)
    ind_infos = ind_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    var = self.var_
    if var.coord.shape[0]>nb_element:
        print(f"\nVariables (the {nb_element} first)\n")
    else:
         print("\nVariables\n")
    var_infos = DataFrame().astype("float")
    for i in range(ncp):
        var_coord, var_sqcos, var_ctr = var.coord.iloc[:,i], var.cos2.iloc[:,i], var.contrib.iloc[:,i]
        var_sqcos.name, var_ctr.name = "cos2", "ctr"
        var_infos = concat((var_infos,var_coord,var_ctr,var_sqcos),axis=1)
    var_infos = var_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary quantitativve variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self, "quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
        quanti_sup_infos = DataFrame().astype("float")
        for i in range(ncp):
            quanti_sup_coord, quanti_sup_sqcos = quanti_sup.coord.iloc[:,i], quanti_sup.cos2.iloc[:,i]
            quanti_sup_sqcos.name = "cos2"
            quanti_sup_infos =concat((quanti_sup_infos,quanti_sup_coord,quanti_sup_sqcos),axis=1)
        quanti_sup_infos = quanti_sup_infos.head(nb_element).round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)