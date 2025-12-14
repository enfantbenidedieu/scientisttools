# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from typing import NamedTuple

def get_mca_ind(self) -> NamedTuple:
    """
    Extract the results for individuals - MCA
    -----------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosinus, relative contributions and additional informations) for the active individuals from Multiple Correspondence Analysis (MCA) outputs.

    Usage
    -----
    ```python
    >>> get_mca_ind(self)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals including:

    `coord`: factor coordinates of individuals,

    `cos2`: squared cosinus of individuals,

    `contrib`: relative contributions of individuals,

    `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of individuals.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca_ind
    >>> res_mca = MCA(sup_var = range(4))
    >>> res_mca.fit(poison)
    >>> #extract results for the individuals
    >>> ind = get_mca_ind(res_mca) 
    >>> ind.coord #coordinates of individuals
    >>> ind.cos2 #cos2 of individuals
    >>> ind.contrib #contributions of individuals
    >>> ind.infos #additionals informations of individuals
    ```
    """
    if self.model_ != "mca": #check if self is an object of class MCA
        raise TypeError("'self' must be an object of class MCA")
    return self.ind_
            
def get_mca_var(self, element = "var") -> NamedTuple:
    """
    Extract the results for the variables - MCA
    -------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variable categories from Multiple Correspondence Analysis (MCA) outputs.

    Usage
    -----
    ```python
    >>> get_mca_var(self, element = "var")
    >>> get_mca_var(self, element = "quali_var")
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `element`: a string indicating the element to subset from the output. Possible values are:
        * "var" for levels, 
        * "quali_var" for the qualitative variables.

    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active variables/levels including :

    `coord`: coordinates of the levels/qualitative variables,

    `coord_n`: normalized coordinates of the levels,

    `cos2`: squared cosinus of the levels,

    `contrib`: relative contributions of the levels/qualitative variables,

    `vtest`: value-test of the levels,

    `infos`: additionnal informations (weight, square distance to origin, inertia and percentage of inertia) of levels/qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca_var
    >>> res_mca = MCA(sup_var=range(4))
    >>> res_mca.fit(poison)
    >>> #extract results for the variables
    >>> var = get_mca_var(res_mca, "var") 
    >>> var.coord #coordinates of the levels
    >>> var.coord_n #normalized coordinates of the levels
    >>> var.cos2 #cos2 of the levels
    >>> var.contrib #contributions of the levels
    >>> var.vtest #vtest of the levels
    >>> var.infos #additionals informations of the levels
    >>> #extract results for the qualitative variables
    >>> quali_var = get_mca_var(res_mca, "quali_var") 
    >>> quali_var.coord #coordinates of the qualitative variables
    >>> quali_var.contrib #contributions of the qualitative variables
    >>> quali_var.infos #additionals informations of the qualitative variables
    ```
    """
    if self.model_ != "mca": #check if self is an object of class MCA
        raise TypeError("'self' must be an object of class MCA")
    
    if element == "var":
        return self.var_
    elif element == "quali_var":
        return self.quali_var_
    else:
        raise ValueError("'element' should be one of 'var', 'quali_var'.")
    
def get_mca(self, element="ind") -> NamedTuple:
    """
    Extract the results for individuals/variables - MCA
    ---------------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosine, contributions and additionals informations) for the active individuals/variable categories from Multiple Correspondence Analysis (MCA) outputs.

        * `get_mca()`: Extract the results for vriables and individuals
        * `get_mca_ind()`: Extract the results for individuals only
        * `get_mca_var()`: Extract the results for qualitative variables/levels only
    
    Usage
    -----
    ```python
    >>> get_mca(self,element="ind")
    >>> get_mca(self,element="var")
    >>> get_mca(self,element="quali_var")
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `element`: a string indicating the element to subset from the output. Possible values are:
        * "ind" for individuals, 
        * "var" for levels
        * "quali_var" for qualitative variables
    
    Returns
    -------
    a namedtuple of pandas DataFrames containing all the results for the active individuals/variable categories including :

    `coord`: coordinates for the individuals/levels/qualitative variables

    `cos2`: squared cosinus for the individuals/levels

    `contrib`: relative contributions for the individuals/levels/qualitative variables

    `infos`: additionals informations for the individuals/levels/qualitative variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca
    >>> res_mca = MCA(sup_var=range(4))
    >>> res_mca.fit(poison)
    >>> #extract results for the individuals
    >>> ind = get_mca(res_mca, element = "ind")
    >>> ind.coord #coordinates of individuals
    >>> ind.cos2 #cos2 of individuals
    >>> ind.contrib #contributions of individuals
    >>> ind.infos #additionals informations of individuals
    >>> #extract results for the levels
    >>> var = get_mca(res_mca, element = "var")
    >>> var.coord #coordinates of the levels
    >>> var.coord_n #normalized coordinates of the levels
    >>> var.cos2 #cos2 of the levels
    >>> var.contrib #contributions of the levels
    >>> var.vtest #vtest of the levels
    >>> var.infos #additionals informations of the levels
    >>> #extract results for the qualitative variables
    >>> quali_var = get_mca(res_mca, element = "quali_var")
    >>> quali_var.coord #coordinates of the qualitative variables
    >>> quali_var.contrib #contributions of the qualitative variables
    >>> quali_var.infos #additionals informations of the qualitative variables
    ```
    """
    if element == "ind":
        return get_mca_ind(self)
    elif element in ["var","quali_var"]:
        return get_mca_var(self, element)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'quali_var'")
    
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
    `self`: an object of class MCA

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
    >>> from scientisttools.datatsets import poison
    >>> from scientisttools import MCA, summaryMCA
    >>> res_mca = MCA(sup_var = (0,1,2,3))
    >>> res_mca.fit(poison)
    >>> summaryMCA(res_mca)
    ```
    """
    #check if self is an object of class MCA
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
    ind_infos = ind_infos.head(nb_element).round(decimals=digits)
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
        ind_sup_infos = ind_sup.dist2
        for i in range(ncp):
            ind_sup_coord, ind_sup_cos2 = ind_sup.coord.iloc[:,i], ind_sup.cos2.iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_sup_infos.head(nb_element).round(decimals=digits)
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
        var_coord, var_sqcos, var_ctr, var_vtest = var.coord.iloc[:,i], var.cos2.iloc[:,i],  var.contrib.iloc[:,i], var.vtest.iloc[:,i]
        var_sqcos.name, var_ctr.name, var_vtest.name = "cos2", "ctr", "vtest"
        var_infos = concat((var_infos,var_coord,var_ctr,var_sqcos,var_vtest),axis=1)
    var_infos = var_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    #add qualiattive variables
    quali_var = self.quali_var_
    if quali_var.coord.shape[0] > nb_element:
        print(f"\nQualitative variables (the {nb_element} first)\n")
    else:
        print("\nQualitative variables \n")
    quali_var_infos = quali_var.infos
    for i in range(ncp):
        quali_var_coord, quali_var_ctr = quali_var.coord.iloc[:,i], quali_var.contrib.iloc[:,i]
        quali_var_coord.name, quali_var_ctr.name = "Dim."+str(i+1), "ctr"
        quali_var_infos = concat((quali_var_infos,quali_var_coord, quali_var_ctr),axis=1)
    quali_var_infos = quali_var_infos.head(nb_element).round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary categories
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup.dist2
        for i in range(ncp):
            quali_sup_coord, quali_sup_sqcos, quali_sup_vtest = quali_sup.coord.iloc[:,i], quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_sqcos.name, quali_sup_vtest.name = "cos2", "v.test"
            quali_sup_infos = concat((quali_sup_infos,quali_sup_coord,quali_sup_sqcos,quali_sup_vtest),axis=1)
        quali_sup_infos = quali_sup_infos.head(nb_element).round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        if quali_sup.eta2.shape[0] > nb_element:
            print(f"\nSupplementary qualitative variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary qualitative variables (eta2)\n")
        quali_var_sup_infos = quali_sup.eta2.head(nb_element).iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_var_sup_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    #add supplementary continuous variables informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary continuous variables (the {nb_element} first)\n")
        else:
            print("\nSupplementary continuous variables\n")
        quanti_sup_infos = DataFrame().astype("float")
        for i in range(0,ncp,1):
            quanti_sup_coord, quanti_sup_sqcos = quanti_sup.coord.iloc[:,i], quanti_sup.cos2.iloc[:,i]
            quanti_sup_sqcos.name = "cos2"
            quanti_sup_infos = concat((quanti_sup_infos,quanti_sup_coord,quanti_sup_sqcos),axis=1)
        quanti_sup_infos = quanti_sup_infos.head(nb_element).round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos) 