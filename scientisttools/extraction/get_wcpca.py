# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_wcpca_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - wcPCA
    -------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) of the active individuals from Within-class Principal Component Analysis (wcPCA) outputs.

    Usage
    -----
    ```python
    >>> get_wcpca_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class wcPCA

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals, including:

    `coord`: coordinates of the individuals,

    `contrib`: relative contributions of the individuals,

    `cos2`: squared cosinus of the individuals,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the informations.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import meaudret
    >>> from scientisttools import wcPCA, get_wcpca_ind
    >>> res_wcpca = wcPCA(group=9,sup_var=range(10,24))
    >>> res_wcpca.fit(meaudret)
    >>> #extract the results for individuals
    >>> ind = get_wcpca_ind(res_wcpca)
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.infos.head() #additionals informations of individuals
    ```
    """
    if self.model_ != "wcpca": #check if self is an object of class wcPCA
        Exception("'self' must be an object of class wcPCA.")
    return self.ind_

def get_wcpca_var(self) -> NamedTuple:
    """
    Extract the results for variables - wcPCA
    ---------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Within-class Principal Component Analysis (wcPCA) outputs

    Usage
    -----
    ```python
    >>> get_wcpca_var(self)
    ```

    Parameters
    ----------
    `self`: an object of class wcPCA

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
    >>> from scientisttools.datasets import meaudret
    >>> from scientisttools import wcPCA, get_wcpca_var
    >>> res_wcpca = wcPCA(group=9,sup_var=range(10,24))
    >>> res_wcpca.fit(meaudret)
    >>> #extract the results for variables
    >>> var = get_wcpca_var(res_wcpca)
    >>> var.coord.head() #coordinates of variables
    >>> var.contrib.head() #contributions of variables
    >>> var.cos2.head() #cos2 of variables
    >>> var.infos.head() #additionals informations of variables
    ```
    """
    if self.model_ != "wcpca": #check if self is an object of class wcPCA
        Exception("'self' must be an object of class wcPCA.")
    return self.var_

def get_wcpca(self,element = "ind") -> NamedTuple:
    """
    Extract the results for individuals/variables - wcPCA
    -----------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables from Within-class Principal Component Analysis (wcPCA) outputs.

        * `get_wcpca()`: Extract the results for variables and individuals
        * `get_wcpca_ind()`: Extract the results for individuals only
        * `get_wcpca_var()`: Extract the results for variables only

    Usage
    -----
    ```python
    >>> get_wcpca(self,element="ind")
    >>> get_wcpca(self,element="var")
    ```

    Parameters
    ----------
    `self`: an object of class wcPCA

    `element`: a string indicating the element to subset from the output. Allowed values are :
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
    >>> from scientisttools.datasets import meauret
    >>> from scientisttools import wcPCA, get_wcpca
    >>> res_wcpca = wcPCA(group=9,sup_var=range(10,24))
    >>> res_wcpca.fit(meaudret)
    >>> #extract the results for individuals
    >>> ind = get_wcpca(res_wcpca, "ind")
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.infos.head() #additionals informations of individuals
    >>> #extract the results for variables
    >>> var = get_wcpca(res_wcpca, "var") 
    >>> var.coord.head() #coordinates of variables
    >>> var.contrib.head() #contributions of variables
    >>> var.cos2.head() #cos2 of variables
    >>> var.infos.head() #additionals informations of variables
    ```
    """
    if element == "ind":
        return get_wcpca_ind(self)
    elif element == "var":
        return get_wcpca_var(self)
    else:
        Exception("'element' should be one of 'ind' or 'var'.")

def summarywcPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Within-class Principal Component Analysis objects
    -----------------------------------------------------------------------

    Description
    -----------
    Printing summaries of within-class principal component analysis (wcPCA) objects

    Usage
    -----
    ```python
    >>> summarywcPCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class wcPCA

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
    >>> from scientisttools.datasets import meaudret
    >>> from scientisttools import wcPCA, summarywcPCA
    >>> res_wcpca = wcPCA(classe=9,sup_var=range(10,24))
    >>> res_wcpca.fit(meaudret)
    >>> summaryWCPCA(res_wcpca)
    ```
    """
    #check if self is an object of class wcPCA
    if self.model_ != "wcpca":
        raise TypeError("'self' must be an object of class wcPCA")

    #define number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    #Within-class Principal Components Analysis Results
    print("          Within-class Principal Component Analysis - Results                     \n")

    #add eigenvalues informations
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
    ind_infos = ind.infos
    for i in range(ncp):
        ind_coord, ind_sqcos, ind_ctr = ind.coord.iloc[:,i], ind.cos2.iloc[:,i], ind.contrib.iloc[:,i]
        ind_sqcos.name, ind_ctr.name = "cos2", "ctr"
        ind_infos = concat((ind_infos,ind_coord,ind_sqcos,ind_ctr),axis=1)
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
            ind_sup_coord, ind_sup_sqcos = ind_sup.coord.iloc[:,i], ind_sup.cos2.iloc[:,i]
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
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup.dist2
        for i in range(ncp):
            quali_sup_coord, quali_sup_sqcos, quali_sup_vtest = quali_sup.coord.iloc[:,i], quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_sqcos.name, quali_sup_vtest.name = "cos2", "vtest"
            quali_sup_infos = concat((quali_sup_infos,quali_sup_coord,quali_sup_sqcos,quali_sup_vtest),axis=1)
        quali_sup_infos = quali_sup_infos.head(nb_element).round(decimals=digits)
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