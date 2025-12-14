# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_bcpca_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - bcPCA
    -------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus and squared distance to origin) of the active individuals from Between-class Principal Component Analysis (bcPCA) outputs.

    Usage
    -----
    ```python
    >>> get_bcpca_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

    Returns
    -------
    a namedtuple of pandas DataFrames containing/Series all the results for the active individuals, including:

    `coord`: coordinates of the individuals,

    `cos2`: squared cosinus of the individuals,

    `dist2`: squared distance to origin of the individuals.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, get_bcpca_ind
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> #extract the results for individuals
    >>> ind = get_bcpca_ind(res_bcpca)
    >>> ind.coord #coordinates of individuals
    >>> ind.cos2 #cos2 of individuals
    >>> ind.dist2 #dist2 of individuals
    ```
    """
    if self.model_ != "bcpca": #check if self is an object of class bcPCA
        Exception("'self' must be an object of class bcPCA.")
    return self.ind_

def get_bcpca_var(self) -> NamedTuple:
    """
    Extract the results for variables - bcPCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Between-class Principal Component Analysis (bcPCA) outputs

    Usage
    -----
    ```python
    >>> get_bcpca_var(self)
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active variables, including:

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
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, get_bcpca_var
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> #extract the results for variables
    >>> var = get_bcpca_var(res_bcpca)
    >>> var.coord #coordinates of variables
    >>> var.contrib #contributions of variables
    >>> var.cos2 #cos2 of variables
    >>> var.infos #additionals informations of variables
    ```
    """
    if self.model_ != "bcpca": #check if self is an object of class bcPCA
        Exception("'self' must be an object of class bcPCA.")
    return self.var_

def get_bcpca_group(self) -> NamedTuple:
    """
    Extract the results for groups - bcPCA
    --------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active groups from Between-class Principal Component Analysis (bcPCA) outputs

    Usage
    -----
    ```python
    >>> get_bcpca_group(self)
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the groups, including:

    `coord`: coordinates of the groups,

    `contrib`: relative contributions of the groups,

    `cos2`: squared cosinus of the groups,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the groups.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_meaudret
    >>> meaudret = load_meaudret("actif")
    >>> from scientisttools import bcPCA, get_bcpca_group
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> #extract the results for groups
    >>> group = get_bcpca_group(res_bcpca)
    >>> group.coord #coordinates of groups
    >>> group.contrib #contributions of groups
    >>> group.cos2 #cos2 of groups
    >>> group.infos #additionals informations of groups
    ```
    """
    if self.model_ != "bcpca": #check if self is an object of class bcPCA
        Exception("'self' must be an object of class bcPCA.")
    return self.group_

def get_bcpca(self,element = "ind") -> NamedTuple:
    """
    Extract the results for individuals/variables/groups - bcPCA
    ------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables/groups from Between-class Principal Component Analysis (BetweenPCA) outputs.

        * `get_bcpca()`: Extract the results for variables, individuals and groups
        * `get_bcpca_ind()`: Extract the results for individuals only
        * `get_bcpca_var()`: Extract the results for variables only
        * `get_bcpca_group()`: Extract the results for groups only

    Usage
    -----
    ```python
    >>> get_bcpca(self,element="ind")
    >>> get_bcpca(self,element="var")
    >>> get_bcpca(self,element="group")
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

    `element`: a string indicating the element to subset from the output. Allowed values are:
        * "ind" for individuals
        * "var" for variables
        * "group" for groups
                
    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals/variables/groups including:

    `coord`: coordinates of the individuals/variables/groups,
    
    `contrib`: relative contributions of the individuals/variables,

    `cos2`: squared cosinus of the individuals/variables/groups,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals/variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, get_bcpca
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = BetweenPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> #extract the results for individuals
    >>> ind = get_bcpca(res_bcpca, "ind")
    >>> ind.coord #coordinates of individuals
    >>> ind.cos2 #cos2 of individuals
    >>> ind.dist2 #dist2 of individuals
    >>> #extract the results for variables
    >>> var = get_bcpca(res_bcpca, "var") 
    >>> var.coord #coordinates of variables
    >>> var.contrib #contributions of variables
    >>> var.cos2 #cos2 of variables
    >>> var.infos #additionals informations of variables
    >>> #extract the results for groups
    >>> group = get_bcpca(res_bcpca, "group") 
    >>> group.coord #coordinates of groups
    >>> group.contrib #contributions of groups
    >>> group.cos2 #cos2 of groups
    >>> group.infos #additionals informations of groups
    ```
    """
    if element == "ind":
        return get_bcpca_ind(self)
    elif element == "var":
        return get_bcpca_var(self)
    elif element == "group":
        return get_bcpca_group(self)
    else:
        Exception("'element' should be one of 'ind', 'var' or 'group'.")

def summarybcPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Between-class Principal Component Analysis objects
    ------------------------------------------------------------------------

    Description
    -----------
    Printing summaries of between-class principal component analysis (bcPCA) objects

    Usage
    -----
    ```python
    >>> summarybcPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class bcPCA

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
    >>> from scientisttools.datasets import load_meaudret
    >>> from scientisttools import bcPCA, summarybcPCA
    >>> meaudret = load_meaudret("actif")
    >>> res_bcpca = bcPCA(group=9)
    >>> res_bcpca.fit(meaudret)
    >>> summarybcPCA(res_bcpca)
    ```
    """
    if self.model_ != "bcpca": #check if self is an object of class bcPCA
        raise TypeError("'self' must be an object of class bcPCA")

    #define number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    #Between-class Principal Components Analysis Results
    print("          Between-class Principal Component Analysis - Results                     \n")

    #add eigenvalues informations
    print("Eigenvalues")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add groups informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    grp = self.group_
    if grp.coord.shape[0] > nb_element:
        print(f"\nGroups (the {nb_element} first)\n")
    else:
        print("\nGroups\n")
    grp_infos = grp.infos
    for i in range(ncp):
        grp_coord, grp_cos2, grp_ctr = grp.coord.iloc[:,i], grp.cos2.iloc[:,i], grp.contrib.iloc[:,i]
        grp_cos2.name, grp_ctr.name = "cos2", "ctr"
        grp_infos = concat((grp_infos,grp_coord,grp_ctr,grp_cos2),axis=1)
    grp_infos = grp_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(grp_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(grp_infos)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ind = self.ind_
    if ind.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_infos = ind.dist2
    for i in range(ncp):
        ind_coord, ind_cos2 = ind.coord.iloc[:,i], ind.cos2.iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = concat((ind_infos,ind_coord,ind_cos2),axis=1)
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
        var_coord, var_cos2, var_ctr = var.coord.iloc[:,i], var.cos2.iloc[:,i], var.contrib.iloc[:,i]
        var_cos2.name, var_ctr.name = "cos2", "ctr"
        var_infos = concat((var_infos,var_coord,var_ctr,var_cos2),axis=1)
    var_infos = var_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)