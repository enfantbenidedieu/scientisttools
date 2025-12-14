# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_famd_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - FAMD
    ------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the individuals from Factor Analysis of Mixed Data (FAMD) outputs.

    Usage
    -----
    ```python
    >>> get_famd_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    Returns
    -------
    namedtuple of pandas DataFrames containing all the results for the individuals including:

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
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_famd_ind
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_famd.fit(autos2005)
    >>> #extract results for individuals
    >>> ind = get_famd_ind(res_famd)
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    ```
    """
    if self.model_ != "famd": #check if self is an object of class FAMD
        raise ValueError("'self' must be an object of class FAMD")
    return self.ind_
    
def get_famd_var(self, element="var") -> NamedTuple:
    """
    Extract the results for variables - FAMD
    ----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus and relative contributions) for quantitative and qualitative variables from Factor Analysis of Mixed Data (FAMD) outputs.

    Usage
    -----
    ```python
    >>> get_famd_var(self,choice="var")
    >>> get_famd_var(self,choice="quanti_var")
    >>> get_famd_var(self,choice="quali_var")
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `element`: the element to subset from the output. Possible values are :
        * "var" for active variables
        * "quanti_var" for active quantitatives variables
        * "quali_var" for active qualitatives variables (categories)

    Returns
    -------
    namedtuple of dataframes containing the results for the active variables, including :

    `coord`: coordinates for the variables,

    `cos2`: squared cosinus for the variables,

    `contrib`: relative contributions for the variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_famd_var
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_famd.fit(autos2005)
    >>> #extract results for quantitatives variables
    >>> quanti_var = get_famd_var(res_famd, element = "quanti_var")
    >>> quanti_var.coord.head() #coordinates of quantitative variables
    >>> quanti_var.cos2.head() #cos2 of quantitative variables
    >>> quanti_var.contrib.head() #contribution of quantitative variables
    >>> #extract results for qualitatives variables
    >>> quali_var = get_famd_var(res_famd, element = "quali_var")
    >>> quali_var.coord.head() #coordinates of categories/variables
    >>> quali_var.cos2.head() #cos2 of categories/variables
    >>> quali_var.contrib.head() #contribution of categories/variables
    >>> quali_var.vtest.head() #value-test of categories/variables
    >>> #extract results for variables
    >>> var = get_famd_var(res_famd, element = "var")
    >>> var.coord.head() #coordinates of variables
    >>> var.cos2.head() #cos2 of cvariables
    >>> var.contrib.head() #contribution of variables
    ```
    """
    if self.model_ != "famd": #check if self is an object of class FAMD
        raise ValueError("'self' must be an object of class FAMD")
    
    if element not in ["quanti_var","quali_var","var"]:
        raise ValueError("'element' should be one of 'quanti_var', 'quali_var', 'var'")
    
    if element == "quanti_var":
        if not hasattr(self,"quanti_var_"):
            raise ValueError("No quantitatives columns")
        return self.quanti_var_
    
    if element == "quali_var":
        if not hasattr(self,"quali_var_"):
            raise ValueError("No qualitatives columns")
        return self.quali_var_
    
    if element == "var":
        if not hasattr(self,"var_"):
            raise ValueError("No mixed data")
        return self.var_

def get_famd(self, element = "ind")-> NamedTuple:
    """
    Extract the results for individuals and variables - FAMD
    --------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and relative contributions) for the individuals and variables from Factor Analysis of Mixed Data (FAMD) outputs.

    Usage
    -----
    ```python
    >>> get_famd(self, element = "ind")
    >>> get_famd(self, element = "var")
    >>> get_famd(self, element = "quanti_var")
    >>> get_famd(self, element = "quali_var")
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `element`: the element to subset from the output. Possibles values are :
        * "ind" for individuals
        * "var" for active variables
        * "quanti_var" for active quantitatives variables
        * "quali_var" for active qualitatives variables (categories)

    Return
    ------
    namedtuple of dataframes containing the results for the active individuals and variables, including :
    
    `coord`: coordinates for the individuals/variables.
    
    `cos2`: squared cosinus for the individuals/variables.
    
    `contrib`: relative contributions for the individuals/variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_famd
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_famd.fit(autos2005)
    >>> #extract results for individuals
    >>> ind = get_famd(res_famd, "ind")
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> #extract results for quantitatives variables
    >>> quanti_var = get_famd(res_famd, element = "quanti_var")
    >>> quanti_var.coord.head() #coordinates of quantitative variables
    >>> quanti_var.cos2.head() #cos2 of quantitative variables
    >>> quanti_var.contrib.head() #contribution of quantitative variables
    >>> #extract results for qualitatives variables
    >>> quali_var = get_famd(res_famd, element = "quali_var")
    >>> quali_var.coord.head() #coordinates of categories/variables
    >>> quali_var.cos2.head() #cos2 of categories/variables
    >>> quali_var.contrib.head() #contribution of categories/variables
    >>> quali_var.vtest.head() #value-test of categories/variables
    >>> #extract results for variables
    >>> var = get_famd(res_famd, element = "var")
    >>> var.coord.head() #coordinates of variables
    >>> var.cos2.head() #cos2 of cvariables
    >>> var.contrib.head() #contribution of variables
    ```
    """
    if element == "ind":
        return get_famd_ind(self)
    elif element in ("var","quali_var","quanti_var"):
        return get_famd_var(self,element=element)
    else:
        raise ValueError("'element' should be one of 'ind', 'ind_sup', 'quanti_var', 'quali_var', 'var'")

def summaryFAMD(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Factor Analysis of Mixed Data model
    ---------------------------------------------------------

    Description
    -----------
    Printing summaries of factor analysis of mixed data (FAMD) objects

    Usage
    -----
    ```python
    >>> summaryFAMD(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `digits`: int, default=3. Number of decimal printed

    `nb_element`: int, default = 10. Number of element

    `ncp`: int, default = 3. Number of components

    `to_markdown`: Print DataFrame in Markdown-friendly format

    `tablefmt`: Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs`: These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, summaryFAMD
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_famd.fit(autos2005)
    >>> summaryFAMD(res_famd)
    ```
    """
    if self.model_ != "famd": #check if self is an object of class FAMD
        raise ValueError("'self' must be an object of class FAMD")

    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    # Title
    print("                     Factor Analysis of Mixed Data - Results                     \n")

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
        ind_infos = concat((ind_infos,ind_coord,ind_ctr,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)

    # Add supplementary individuals
    if hasattr(self,"ind_sup_"):
        ind_sup = self.ind_sup_
        if ind.coord.shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup_infos = ind_sup.dist2
        for i in range(ncp):
            ind_sup_coord, ind_sup_cos2 = ind_sup.coord.iloc[:,i], ind_sup.cos2.iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    # Add variables informations
    if hasattr(self,"quanti_var_"):
        quanti_var = self.quanti_var_
        if quanti_var.coord.shape[0]>nb_element:
            print(f"\nContinuous variables (the {nb_element} first)\n")
        else:
            print("\nContinuous variables\n")
        quanti_var_infos = DataFrame().astype("float")
        for i in range(ncp):
            quanti_var_coord, quanti_var_cos2, quanti_var_ctr = quanti_var.coord.iloc[:,i], quanti_var.cos2.iloc[:,i], quanti_var.contrib.iloc[:,i]
            quanti_var_cos2.name, quanti_var_ctr.name = "cos2", "ctr"
            quanti_var_infos = concat((quanti_var_infos,quanti_var_coord,quanti_var_ctr,quanti_var_cos2),axis=1)
        quanti_var_infos = quanti_var_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_var_infos)
    
    # Add supplementary continuous variables informations
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
        quanti_sup_infos = DataFrame().astype("float")
        for i in range(ncp):
            quanti_sup_coord, quanti_sup_cos2  = quanti_sup.coord.iloc[:,i], quanti_sup.cos2.iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos = concat((quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2),axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    if hasattr(self,"quali_var_"):
        quali_var = self.quali_var_
        if quali_var.coord.shape[0] > nb_element:
            print(f"\nCategories (the {nb_element} first)\n")
        else:
            print("\nCategories\n")
        quali_var_infos = quali_var.dist2
        for i in range(ncp):
            quali_var_coord,quali_var_cos2,quali_var_ctr,quali_var_vtest= quali_var.coord.iloc[:,i],quali_var.cos2.iloc[:,i],quali_var.contrib.iloc[:,i],quali_var.vtest.iloc[:,i]
            quali_var_cos2.name, quali_var_ctr.name, quali_var_vtest.name  = "cos2", "ctr", "vtest"
            quali_var_infos = concat((quali_var_infos,quali_var_coord,quali_var_ctr,quali_var_cos2,quali_var_vtest),axis=1)
        quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_var_infos)
    
        # Add categoricals variables square correlation ratio
        if hasattr(self,"quanti_var_"):
            quali_var_eta2 = self.var_.coord.drop(index=self.quanti_var_.coord.index)
        else:
            quali_var_eta2 = self.quali_var_.eta2

        if quali_var_eta2.shape[0] > nb_element:
            print(f"\nCategoricals variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nCategoricals variables (eta2)\n")
        quali_var_eta2 = quali_var_eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_var_eta2)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup.dist2
        for i in range(ncp):
            quali_sup_coord, quali_sup_cos2, quali_sup_vtest = quali_sup.coord.iloc[:,i], quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_cos2.name, quali_sup_vtest.name = "cos2", "v.test"
            quali_sup_infos = concat((quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest),axis=1)
        quali_sup_infos = quali_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        if quali_sup.eta2.shape[0] > nb_element:
            print(f"\nSupplementary categorical variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary categorical variables (eta2)\n")
        quali_sup_eta2 = self.quali_sup_.eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)