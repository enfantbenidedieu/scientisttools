# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_mca_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - MCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus and relative contributions) for the active individuals from Multiple Correspondence Analysis (MCA) outputs.

    Usage
    -----
    ```python
    >>> get_mca_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class MCA

    Returns
    -------
    namedtuple containing the results of the cative individuals including : 

    `coord` : factor coordinates (scores) of the individuals

    `cos2` : square cosinus of the individuals

    `contrib` : relative contributions of the individuals

    `infos` : additionnal informations (weight, square distance to origin, inertia) of the individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> from scientistools import get_mca_ind
    >>> # Extract results for the individuals
    >>> ind = get_mca_ind(res_mca) 
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    return self.ind_
            
def get_mca_var(self) -> NamedTuple:
    """
    Extract the results for the variables - MCA
    -------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) for the active variable categories from Multiple Correspondence Analysis (MCA) outputs.

    Usage
    -----
    ```python
    >>> get_mca_var(self)
    ```

    Parameters
    ----------
    `self` : an object of class MCA

    Returns
    -------
    namedtuple of dataframes containing the results for the active variable categories including :

    `coord` : factor coordinates (scores) for the variables categories

    `corrected_coord` : corrected factor coordinates for the variables categories

    `cos2` : square cosinus for the variables categories

    `contrib`  : relative contributions of the variables categories

    `vtest` : v-test for the variables categories

    `eta2` : squared correlation ratio for the variables

    `var_contrib` : contributions of the variables

    `infos` : additionnal informations (weight, square distance to oriigin, inertia) for the variables categories :

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> from scientistools import get_mca_var
    >>> # Extract results for the variables
    >>> var = get_mca_var(res_mca) 
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    return self.var_
    
def get_mca(self,choice="ind") -> NamedTuple:
    """
    Extract the results for individuals/variables - MCA
    ---------------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosine and contributions) for the active individuals/variable categories from Multiple Correspondence Analysis (MCA) outputs.

        * get_mca() : Extract the results for vriables and individuals
        * get_mca_ind() : Extract the results for individuals only
        * get_mca_var() : Extract the results for variables/categories only
    
    Usage
    -----
    ```python
    >>> get_mca(self,choice=c("ind","var"))
    ```

    Parameters
    ----------
    `self` : an object of class MCA

    `choice` : the element to subset from the output. Possible values are :
        * "ind" for individuals, 
        * "var" for variables/categories
    
    Returns
    -------
    dictionary of dataframes containing the results for the active individuals/variable categories including :

    `coord` : factor coordinates (scores) for the individuals/variable categories

    `cos2` : square cosinus of the individuals/variable categories

    `contrib` : relative contributions of the individuals/variable categories

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA, get_mca
    >>> res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1])
    >>> res_mca.fit(poison)
    >>> # Extract results for the individuals
    >>> ind = get_mca(res_mca, choice = "ind")
    >>> # Extract results for the categories
    >>> var = get_mca(res_mca, choice = "var")
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    
    if choice not in ["ind","var"]:
        raise ValueError("'choice' should be one of 'ind', 'var'")

    if choice == "ind":
        return get_mca_ind(self)
    else:
        return get_mca_var(self)
    
def summaryMCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Correspondence Analysis model
    ------------------------------------------------------------

    Description
    -----------
    Printing summaries of multiple correspondence analysis (MCA) objects

    Usage
    -----
    ```python
    >>> summaryMCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class MCA

    `digits` : int, default=3. Number of decimal printed

    `nb_element` : int, default = 10. Number of element

    `ncp` : int, default = 3. Number of components

    `to_markdown` : Print DataFrame in Markdown-friendly format.

    `tablefmt` : Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    
    `**kwargs` : These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA, summaryMCA
    >>> res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> summaryMCA(res_mca)
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise ValueError("'self' must be an object of class MCA")

    #et number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0],self.var_.coord.shape[0])

    # Multiple correspondance Analysis - Results
    print("                     Multiple Correspondance Analysis - Results                     \n")
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##Add eigenvalues informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    print("Importance of components")
    eig = self.eig_.iloc[:,:4].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add individuals informations
    #----------------------------------------------------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary individuals
    #----------------------------------------------------------------------------------------------------------------------------------------
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
            ind_sup_infos = concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add variables informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    var = self.var_
    if var.coord.shape[0] > nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
        print("\nCategories\n")
    var_infos = var.infos
    for i in range(ncp):
        var_coord, var_cos2, var_ctr, var_vtest = var.coord.iloc[:,i], var.cos2.iloc[:,i],  var.contrib.iloc[:,i], var.vtest.iloc[:,i]
        var_cos2.name, var_ctr.name, var_vtest.name = "cos2", "ctr", "vtest"
        var_infos = concat((var_infos,var_coord,var_ctr,var_cos2,var_vtest),axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add variables
    if var.var_inertia.shape[0] > nb_element:
        print(f"\nCategorical variables (eta2) (the {nb_element} first)\n")
    else:
        print("\nCategorical variables (eta2)\n")
    quali_var_infos = var.var_inertia
    for i in range(ncp):
        quali_var_eta2, quali_var_contrib = var.eta2.iloc[:,i], var.var_contrib.iloc[:,i]
        quali_var_eta2.name, quali_var_contrib.name = "Dim."+str(i+1), "ctr"
        quali_var_infos = concat((quali_var_infos,quali_var_eta2,quali_var_contrib),axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary categories
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quali_sup_"):
        var_sup = self.quali_sup_
        if var_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        var_sup_infos = var_sup.dist
        for i in range(ncp):
            var_sup_coord, var_sup_cos2, var_sup_vtest = var_sup.coord.iloc[:,i], var_sup.cos2.iloc[:,i], var_sup.vtest.iloc[:,i]
            var_sup_cos2.name, var_sup_vtest.name = "cos2", "v.test"
            var_sup_infos = concat((var_sup_infos,var_sup_coord,var_sup_cos2,var_sup_vtest),axis=1)
        var_sup_infos = var_sup_infos.round(decimals=digits)
        if to_markdown:
            print(var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(var_sup_infos)
        
        if var_sup.eta2.shape[0] > nb_element:
            print(f"\nSupplementary categorical variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary categorical variables (eta2)\n")
        quali_var_sup_infos = var_sup.eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_var_sup_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary continuous variables informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
        quanti_sup_infos = DataFrame().astype("float")
        for i in range(0,ncp,1):
            quanti_sup_coord, quanti_sup_cos2 = quanti_sup.coord.iloc[:,i], quanti_sup.cos2.iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos = concat((quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2),axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos) 