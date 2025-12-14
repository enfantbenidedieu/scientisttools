# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_dmfa_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - DMFA
    ------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals from Dual Multiple Factor Analysis (DMFA) outputs.

    Usage
    -----
    ```python
    >>> get_dmfa_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class DMFA

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals, including:

    `coord`: coordinates for the individuals,
    
    `cos2`: squared cosinus for the individuals,
    
    `contrib`: relative contributions for the individuals,
    
    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the individuals.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from seaborn import load_dataset
    >>> from scientisttools.datasets import DMFA, get_dmfa_ind
    >>> iris = load_dataset('iris')
    >>> res_dmfa = DMFA(group=4)
    >>> res_dmfa.fit(iris)
    >>> #extract all results for the individuals
    >>> ind = get_dmfa_ind(res_dmfa)
    >>> ind.coord #coordinates for the individuals
    >>> ind.cos2 #cos2 for the individuals
    >>> ind.contrib #contributions for the individuals
    >>> ind.infos #additionals informations for the individuals
    ```
    """
    if self.model_ != "dmfa": #check if self is not an instance DMFA
        raise TypeError("'self' must be an object of class DMFA")
    return self.ind_

def get_dmfa_var(self, element = "group"):
    """
    Extract the results for variables - DMFA
    ----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables/groups from Dual Multiple Factor Analysis (DMFA) outputs.

    Usage
    -----
    ```python
    >>> get_dmfa_var(self)
    ```

    Parameters
    ----------
    `self`: an object of class DMFA
    
    `element`: a string indicating the element to subset from the output. Possibles values are:
        * "var" for active variables
        * "group" for group

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active variables/groups, including:

    `coord`: coordinates for the variables/groups
    
    `cos2`: square cosinus for the variables/groups
    
    `contrib`: relative contributions of the variables/groups

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import DMFA, get_dmfa_var
    >>> from seaborn import load_dataset
    >>> iris = load_dataset('iris')
    >>> res_dmfa = DMFA(group=4)
    >>> res_dmfa.fit(iris)
    >>> #extract all the results for the variables
    >>> var = get_dmfa_var(res_dmfa, "var")
    ```
    """
    if self.model_ != "dmfa": #check if self is an object of class DMFA
        raise TypeError("'self' must be an object of class DMFA")
    
    if element == "var": #for variables
        return self.var_
    elif element == "group": #for group informations
        return self.group_
    else:
        raise ValueError("'element' should be one of 'group', 'var'")
        
def get_dmfa(self,element="ind"):
    """
    Extract the results for individuals/variables/groups - DMFA
    -----------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables/groups from Dual Multiple Factor Analysis (DMFA) outputs.
    
        * `get_dmfa()`: Extract the results for variables/groups and individuals
        * `get_dmfa_ind()`: Extract the results for individuals only
        * `get_dmfa_var()`: Extract the results for variables/groups
    
    Usage
    -----
    ```python
    >>> get_dmfa(self, element = "ind")
    >>> get_dmfa(self, element = "var")
    >>> get_dmfa(self, element = "group")
    ```

    Parameters
    ----------
    `self`: an object of class DMFA

    `element`: a string indicating the element to subset from the output. Possibles values are:
        * "ind" for individuals 
        * "var" for variables
        * "group" for groups

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals/variables/groups, including:

    `coord`: coordinates for the individuals/variables/groups
    
    `cos2`: squared cosinus for the individuals/variables/groups

    `contrib`: relative contributions of the individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    """
    if self.model_ != "dmfa": #check if self is an object of class DMFA
        raise TypeError("'self' must be an object of class DMFA")

    if element == "ind":
        return get_dmfa_ind(self)
    elif element in ["var","group"]:
        return get_dmfa_var(self,element=element)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'group'")

def summaryDMFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Dual Multiple Factor Analysis objects
    -----------------------------------------------------------

    Description
    -----------
    Printing summaries of dual multiple factor analysis (DMFA) objects

    Usage
    -----
    ```python
    >>> summaryDMFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class DMFA

    `digits`: an integer indicating the number of decimal printed (default=3)

    `nb_element`: an integer indicating the umber of element(default = 10)

    `ncp`: an integer indicating the number of components (default = 3)

    `to_markdown`: a boolean indicating Print DataFrame in Markdown-friendly format.

    `tablefmt`: a string indicating the table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs`: additionals parameters to pass to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    
    ```
    """
    if self.model_ != "dmfa": #check if self is an object of class DMFA
        raise TypeError("'self' must be an object of class DMFA")

    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.ind_.coord.shape[0])

    #dual multiple factor analysis results
    print("                     Dual Multiple Factor Analysis - Results                     \n")

    #add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_.n_components,:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add groups informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nGroups\n")
    group = self.group_
    group_infos = DataFrame().astype(float)
    for i in range(ncp):
        group_coord, group_sqcos  = group.coord.iloc[:,i], group.cos2.iloc[:,i]
        group_sqcos.name = "cos2"
        group_infos = concat((group_infos,group_coord,group_sqcos),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ind = self.ind_
    if ind.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_infos = ind.infos
    for i in range(ncp):
        ind_coord,ind_ctr, ind_cos2 = ind.coord.iloc[:,i], ind.contrib.iloc[:,i], ind.cos2.iloc[:,i]
        ind_ctr.name, ind_cos2.name = "ctr", "cos2"
        ind_infos = concat((ind_infos,ind_coord,ind_ctr,ind_cos2),axis=1)
    ind_infos = ind_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup_infos = ind_sup.dist2
        for i in range(ncp):
            ind_sup_coord, ind_sup_sqcos = ind_sup.coord.iloc[:,i], ind_sup.sqcos.iloc[:,i]
            ind_sup_sqcos.name = "cos2"
            ind_sup_infos = concat((ind_sup_infos,ind_sup_coord,ind_sup_sqcos),axis=1)
        ind_sup_infos = ind_sup_infos.head(nb_element).round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    var = self.var_
    if var.coord.shape[0] > nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
        print("\nContinuous variables\n")
    var_infos = DataFrame().astype("float")
    for i in range(ncp):
        var_coord, var_ctr, var_sqcos = var.coord.iloc[:,i], var.contrib.iloc[:,i], var.cos2.iloc[:,i]
        var_ctr.name, var_sqcos.name = "ctr", "cos2"
        var_infos = concat((var_infos,var_coord,var_ctr,var_sqcos),axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary quantitativve variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quanti_sup_"):
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
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary categories
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self, "quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = DataFrame().astype("float")
        for i in range(ncp):
            quali_sup_coord, quali_sup_sqcos, quali_sup_vtest = quali_sup.coord.iloc[:,i], quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_sqcos.name, quali_sup_vtest.name = "cos2", "vtest"
            quali_sup_infos = concat((quali_sup_infos,quali_sup_coord,quali_sup_sqcos,quali_sup_vtest),axis=1)
        quali_sup_infos = quali_sup_infos.head(nb_element).round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)

        #add correlation ratio
        if self.quali_sup_.eta2.shape[0] > nb_element:
            print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories (eta2)\n")
        quali_sup_eta2 = quali_sup.eta2.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_eta2)