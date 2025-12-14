# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_ppca_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - pPCA
    ------------------------------------------

    Description
    ------------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) of the active individuals from Partial Principal Component Analysis (pPCA) outputs.

    Usage
    -----
    ```python
    >>> get_ppca_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class pPCA

    Returns
    -------
    a namedtuple of pandas Dataframes containing all the results for the active individuals, including:

    `coord`: coordinates of the individuals,

    `cos2`: squared cosinus of the individuals

    `contrib`: relative contributions of the individuals

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2006
    >>> from scientisttools import pPCA, summarypPCA
    >>> res_ppca = pPCA(partial=0,ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_ppca.fit(autos2006)
    >>> #extract the results for individuals
    >>> ind = get_ppca_ind(res_ppca)
    >>> ind.coord # coordinates of individuals
    >>> ind.cos2 # cos2 of individuals
    >>> ind.contrib # contributions of individuals
    ```
    """
    if self.model_ != "ppca": #check if self is an object of class pPCA
        raise TypeError("'self' must be an object of class pPCA.")
    return self.ind_

def get_ppca_var(self) -> NamedTuple:
    """
    Extract the results for variables - pPCA
    ----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Partial Principal Component Analysis (pPCA) outputs

    Usage
    -----
    ```python
    >>> get_ppca_var(self)
    ```

    Parameters
    ----------
    `self`: an instance of class pPCA

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
    >>> from scientisttools.datasets import autos2006
    >>> from scientisttools import pPCA, get_ppca_var
    >>> res_ppca = pPCA(partial=0,ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_ppca.fit(autos2006)
    >>> #extract results for variables
    >>> var = get_pppca_var(res_ppca)
    >>> var.coord # coordinates of variables
    >>> var.cos2 # cos2 of variables
    >>> var.contrib # contributions of variables
    >>> var.infos # additionals informations of variables
    ```
    """
    if self.model_ != "ppca": #check if self is an object of class pPCA
        raise TypeError("'self' must be an object of class pPCA.")
    return self.var_

def get_ppca(self,element = "ind")-> NamedTuple:
    """
    Extract the results for individuals/variables - pPCA
    ----------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables from Partial Principal Component Analysis (pPCA) outputs.

        * `get_ppca()`: Extract the results for variables and individuals
        * `get_ppca_ind()`: Extract the results for individuals only
        * `get_ppca_var()`: Extract the results for variables only

    Usage
    -----
     ```python
    >>> get_ppca(self,element="ind")
    >>> get_ppca(self,element="var")
    ```

    Parameters
    ----------
    `self`: an object of class pPCA

    `element`: the element to subset from the output. Allowed values are :
        * "ind" for individuals 
        * "var" for variables
    
    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals/variables, including:

    `coord`: coordinates of the individuals/variables,
    
    `cos2`: squared cosinus of the individuals/variables,
    
    `contrib`: relative contributions of the individuals/variables,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals/variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2006
    >>> from scientisttools import pPCA, get_ppca
    >>> res_ppca = pPCA(partial=0,ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_ppca.fit(autos2006)
    >>> #extract results for individuals
    >>> ind = get_ppca(res_ppca,"ind")
    >>> ind.coord # coordinates of individuals
    >>> ind.cos2 # cos2 of individuals
    >>> ind.contrib # contributions of individuals
    >>> #extract results for variables
    >>> var = get_ppca(res_ppca, "var")
    >>> var.coord # coordinates of variables
    >>> var.cos2 # cos2 of variables
    >>> var.contrib # contributions of variables
    >>> var.infos # additionals informations of variables
    ```
    """
    if element == "ind":
        return get_ppca_ind(self)
    elif element == "var":
        return get_ppca_var(self)
    else:
        raise ValueError("'element' should be one of 'ind', 'var'")

def summarypPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Partial Principal Component Analysis model
    ----------------------------------------------------------------

    Description
    -----------
    Printing summaries of partial principal component analysis (pPCA) objects

    Usage
    -----
    ```python
    >>> summarypPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class pPCA

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
    >>> from scientisttools.datasets import autos2006
    >>> from scientisttools import pPCA, summarypPCA
    >>> res_ppca = pPCA(partial=0,ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_ppca.fit(autos2006)
    >>> summarypPCA(res_ppca)
    ```
    """
    if self.model_ != "ppca": #check if self is and object of class pPCA
        raise TypeError("'self' must be an object of class pPCA")

    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0]) #define number of components

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
        ind_coord, ind_sqcos, ind_ctr = ind.coord.iloc[:,i], ind.cos2.iloc[:,i], ind.contrib.iloc[:,i]
        ind_sqcos.name, ind_ctr.name = "cos2", "ctr"
        ind_infos = concat((ind_infos,ind_coord,ind_ctr,ind_sqcos),axis=1)
    ind_infos = ind_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    #add supplementary individuals
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup_infos = ind_sup.dist2
        for i in range(ncp):
            ind_sup_coord, ind_sup_sqcos = ind_sup.coord.iloc[:,i], ind_sup.cos2.iloc[:,i]
            ind_sup_sqcos.name = "cos2"
            ind_sup_infos = concat((ind_sup_infos,ind_sup_coord,ind_sup_sqcos),axis=1)
        ind_sup_infos = ind_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
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
    #add supplementary quantitative variables informations
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
    #add supplementary quantitative variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup.dist2
        for i in range(ncp):
            quali_sup_coord, quali_sup_sqcos,quali_sup_vtest = quali_sup.coord.iloc[:,i],quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_sqcos.name, quali_sup_vtest.name = "cos2", "v.test"
            quali_sup_infos = concat((quali_sup_infos,quali_sup_coord,quali_sup_sqcos,quali_sup_vtest),axis=1)
        quali_sup_infos = quali_sup_infos.head(nb_element).round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        #add supplementary qualitatives - correlation ratio
        if quali_sup.eta2.shape[0] > nb_element:
            print(f"\nSupplementary categorical variable (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary categorical variable (eta2)\n")
        quali_sup_eta2 = quali_sup.eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2) 