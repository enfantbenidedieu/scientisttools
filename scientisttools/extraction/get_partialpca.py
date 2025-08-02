# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_partialpca_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - PartialPCA
    ------------------------------------------------

    Description
    ------------
    Extract all the results (factor coordinates, squared cosinus, relative contributions) of the active individuals from Partial Principal Component Analysis (PartialPCA) outputs.

    Usage
    -----
    ```python
    >>> get_partialpca_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class PartialPCA

    Returns
    -------
    a namedtuple of pandas Dataframes containing all the results for the active individuals including:

    `coord`: factor coordinates (scores)

    `cos2`: squared cosine

    `contrib`: relative contributions

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import autos2006, PartialPCA, summaryPartialPCA
    >>> #with integer
    >>> res_ppca = PartialPCA(partial=0)
    >>> res_ppca.fit(cars)
    >>> #extract the results for individuals
    >>> ind = get_partialpca_ind(res_ppca)
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    ```
    """
    # Check if self is an object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA.")
    return self.ind_

def get_partialpca_var(self) -> NamedTuple:
    """
    Extract the results for variables - PartialPCA
    ----------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, contributions) for the active variables from Partial Principal Component Analysis (PartialPCA) outputs

    Usage
    -----
    ```python
    >>> get_partialpca_var(self)
    ```

    Parameters
    ----------
    `self` : an instance of class PartialPCA

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active variables including:

    `coord`: factor coordinates

    `contrib`: relative contributions

    `cos2`: squared cosinus

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import autos2006, PartialPCA, get_partialpca_var
    >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> #extract results for variables
    >>> var = get_partialpca_var(res_pca)
    >>> var.coord.head() #coordinates of variables
    >>> var.cos2.head() #cos2 of variables
    >>> var.contrib.head() #contributions of variables
    >>> var.infos.head() #additionals informations of variables
    ```
    """
    # Check if self is an object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    return self.var_

def get_partialpca(self,element = "ind")-> NamedTuple:
    """
    Extract the results for individuals/variables - PartialPCA
    ----------------------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) for the active individuals/variables from Partial Principal Component Analysis (PartialPCA) outputs.

        * `get_partialpca()`: Extract the results for variables and individuals
        * `get_partialpca_ind()`: Extract the results for individuals only
        * `get_partialpca_var()`: Extract the results for variables only

    Usage
    -----
     ```python
    >>> get_partialpca(self,element=("ind","var"))
    ```

    Parameters
    ----------
    `self`: an object of class PartialPCA

    `element`: the element to subset from the output. Allowed values are :
        * "ind" for individuals 
        * "var" for variables
    
    Returns
    -------
    namedtuple of dataframes containing all the results for the active individuals/variables including:

    `coord`: factor coordinates (scores) of the individuals/variables
    
    `cos2`: squared cosinus of the individuals/variables
    
    `contrib`: relative contributions of the individuals/variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import autos2006, PartialPCA, get_partialpca
    >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> #extract results for individuals
    >>> ind = get_partialpca(res_partialpca,"ind")
    >>> #extract results for variables
    >>> var = get_partialpca(res_partialpca, "var")
    ```
    """
    # Check if self is an object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    
    if element == "ind":
        return get_partialpca_ind(self)
    elif element == "var":
        return get_partialpca_var(self)
    else:
        raise ValueError("'element' should be one of 'ind', 'var'")

def summaryPartialPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Partial Principal Component Analysis model
    ----------------------------------------------------------------

    Description
    -----------
    Printing summaries of partial principal component analysis (PartialPCA) objects

    Usage
    -----
    ```python
    >>> summaryPartialPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class PartialPCA

    `digits`: int, default=3. Number of decimal printed

    `nb_element`:   int, default = 10. Number of element

    ``ncp`:   int, default = 3. Number of components

    `to_markdown`: Print DataFrame in Markdown-friendly format

    `tablefmt`: Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    `**kwargs`: These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import autos2006, PartialPCA, summaryPartialPCA
    >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> summaryPartialPCA(res_ppca)
    ```
    """
    # Check if self is and object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")

    # Define number of components
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    # Partial Principal Components Analysis Results
    print("                     Partial Principal Component Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_.n_components,:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    ind = self.ind_
    if ind.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind_infos = ind.infos
    for i in range(ncp):
        ind_coord, ind_cos2, ind_ctr = ind.coord.iloc[:,i], ind.cos2.iloc[:,i], ind.contrib.iloc[:,i]
        ind_cos2.name, ind_ctr.name = "cos2", "ctr"
        ind_infos = concat([ind_infos,ind_coord,ind_ctr,ind_cos2],axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup_infos = ind_sup.dist
        for i in range(ncp):
            ind_sup_coord, ind_sup_cos2 = ind_sup.coord.iloc[:,i], ind_sup.cos2.iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = concat([ind_sup_infos,ind_sup_coord,ind_sup_cos2],axis=1)
        ind_sup_infos = ind_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add variables informations
    var = self.var_
    if var.coord.shape[0]>nb_element:
        print(f"\nVariables (the {nb_element} first)\n")
    else:
         print("\nVariables\n")
    var_infos = DataFrame().astype("float")
    for i in range(ncp):
        var_coord, var_cos2, var_ctr = var.coord.iloc[:,i], var.cos2.iloc[:,i], var.contrib.iloc[:,i]
        var_cos2.name, var_ctr.name = "cos2", "ctr"
        var_infos = concat([var_infos,var_coord,var_ctr,var_cos2],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add supplementary continuous variables informations
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
        quanti_sup_infos = DataFrame().astype("float")
        for i in range(ncp):
            quanti_sup_coord, quanti_sup_cos2 = quanti_sup.coord.iloc[:,i], quanti_sup.cos2.iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos =concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup.dist
        for i in range(ncp):
            quali_sup_coord, quali_sup_cos2,quali_sup_vtest = quali_sup.coord.iloc[:,i],quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_cos2.name, quali_sup_vtest.name = "cos2", "v.test"
            quali_sup_infos = concat([quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest],axis=1)
        quali_sup_infos = quali_sup_infos.round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        if quali_sup.eta2.shape[0] > nb_element:
            print(f"\nSupplementary categorical variable (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary categorical variable (eta2)\n")
        quali_sup_eta2 = quali_sup.eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2) 