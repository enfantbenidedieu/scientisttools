# -*- coding: utf-8 -*-
from pandas import concat
from typing import NamedTuple

def get_pcarot_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - PCArot
    --------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) of the active individuals from Varimax rotation with Principal Component Analysis (PCArot) outputs.

    Usage
    -----
    ```python
    >>> get_pcarot_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class PCArot

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals after rotation, including:

    `coord`: coordinates of the individuals after rotation,

    `contrib`: relative contributions of the individuals after rotation,

    `cos2`: squared cosinus of the individuals after rotation,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, PCA, PCArot get_pcarot_ind
    >>> autos2006 = load_dataset("autos2006")
    >>> res_pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_pca.fit(autos2006)
    >>> res_pcarot = PCArot(n_components=2)
    >>> res_pcarot.fit(res_pca)
    >>> #extract the results for individuals
    >>> ind = get_pcaro_ind(res_pca)
    >>> ind.coord #coordinates of individuals after rotation
    >>> ind.contrib #contributions of individuals after rotation
    >>> ind.cos2 #cos2 of individuals after rotation
    >>> ind.infos #additionals informations of individuals
    ```
    """
    if self.model_ != "pcarot": #check if self is an object of class PCArot
        raise TypeError("'self' must be an object of class PCArot")
    return self.ind_

def get_pcarot_var(self) -> NamedTuple:
    """
    Extract the results for variables - PCArot
    ------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Varimax rotation with Principal Component Analysis (PCArot) outputs

    Usage
    -----
    ```python
    >>> get_pcarot_var(self)
    ```

    Parameters
    ----------
    `self`: an object of class PCArot

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active variables, including:

    `coord`: coordinates of the variables after rotation,

    `contrib`: relative contributions of the variables after rotation,

    `cos2`: squared cosinus of the variables after rotation,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, PCA, PCArot get_pcarot_var
    >>> autos2006 = load_dataset("autos2006")
    >>> res_pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_pca.fit(autos2006)
    >>> res_pcarot = PCArot(n_components=2)
    >>> res_pcarot.fit(res_pca)
    >>> #extract the results for variables
    >>> var = get_pcarot_var(res_pca)
    >>> var.coord #coordinates of variables after rotation
    >>> var.contrib #contributions of variables after rotation
    >>> var.cos2 #cos2 of variables after rotation
    >>> var.infos #additionals informations of variables
    ```
    """
    if self.model_ != "pcarot": #check if self is an object of class PCArot
        raise TypeError("'self' must be an object of class PCA")
    return self.var_

def get_pcarot(self,element="ind")-> NamedTuple:
    """
    Extract the results for individuals/variables - PCArot
    ------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables from Varimax rotation with Principal Component Analysis (PCArot) outputs.

        * `get_pcarot()`: Extract the results for variables and individuals after rotation
        * `get_pcarot_ind()`: Extract the results for individuals only after rotation
        * `get_pcarot_var()`: Extract the results for variables only after rotation

    Usage
    -----
    ```python
    >>> get_pcarot(self,element="ind")
    >>> get_pcarot(self,element="var")
    ```

    Parameters
    ----------
    `self`: an object of class PCArot

    `element`: a tsring indicating the element to subset from the output. Allowed values are:
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals/variables, including:

    `coord`: coordinates of the individuals/variables after rotation,
    
    `contrib`: relative contributions of the individuals/variables after rotation,

    `cos2`: squared cosinus of the individuals/variables after rotation,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals/variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, PCA, PCArot get_pcarot
    >>> autos2006 = load_dataset("autos2006")
    >>> res_pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_pca.fit(autos2006)
    >>> res_pcarot = PCArot(n_components=2)
    >>> res_pcarot.fit(res_pca)
    >>> #extract the results for individuals
    >>> ind = get_pcarot(res_pcarot, "ind")
    >>> ind.coord #coordinates of individuals after rotation
    >>> ind.contrib #contributions of individuals after rotation
    >>> ind.cos2 #cos2 of individuals after rotation
    >>> ind.infos #additionals informations of individuals
    >>> #extract the results for variables
    >>> var = get_pcarot(res_pcarot, "var") 
    >>> var.coord #coordinates of variables after rotation
    >>> var.contrib #contributions of variables after rotation
    >>> var.cos2 #cos2 of variables after rotation
    >>> var.infos #additionals informations of variables
    ```
    """
    if element == "ind":
        return get_pcarot_ind(self)
    elif element == "var":
        return get_pcarot_var(self)
    else:
        raise ValueError("'element' should be one of 'ind', 'var'")

def summaryPCArot(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Varimax rotation with Principal Component Analysis objects
    --------------------------------------------------------------------------------

    Description
    -----------
    Printing summaries of varimax rotation with principal component analysis (PCArot) objects

    Usage
    -----
    ```python
    >>> summaryPCArot(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PCArot

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
    >>> from scientisttools import load_dataset, PCA, PCArot, summaryPCArot
    >>> res_pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_pca.fit(autos2006)
    >>> res_pcarot = PCArot(n_components=2)
    >>> res_pcarot.fit(res_pca)
    >>> summaryPCArot(res_pcarot)
    ```
    """
    if self.model_ != "pcarot": #check if self is an object of class PCArot
        raise TypeError("'self' must be an object of class PCArot")

    #define number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.obj.call_.X.shape[0])

    #varimax rotation with principal components analysis Results
    print("                     Varimax rotation with Principal Component Analysis - Results                     \n")

    #add variance accounted informations
    print("Variance accounted:")
    vaccounted = self.vaccounted_.round(decimals=digits)
    if to_markdown:
        print(vaccounted.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(vaccounted)
    
    #rotation matrix
    print("\nRotation matrix:")
    rotmat = self.rotmat_
    if to_markdown:
        print(rotmat.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(rotmat)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ind = self.ind_
    if ind.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first):")
    else:
        print("\nIndividuals:")
    ind_infos = ind.infos
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
        print(f"\nVariables (the {nb_element} first):")
    else:
         print("\nVariables:")
    var_infos = self.others_.communality
    for i in range(ncp):
        var_coord, var_sqcos, var_ctr = var.coord.iloc[:,i], var.cos2.iloc[:,i], var.contrib.iloc[:,i]
        var_sqcos.name, var_ctr.name = "cos2", "ctr"
        var_infos = concat((var_infos,var_coord,var_ctr,var_sqcos),axis=1)
    var_infos = var_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)