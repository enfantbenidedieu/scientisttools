# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from typing import NamedTuple

def get_ca_row(self)-> NamedTuple:
    """
    Extract the results for rows - CA
    ----------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) of the active row variables from Correspondence Analysis (CA) outputs.

    Usage
    -----
    ```python
    >>> get_ca_row(self)
    ```

    Parameters
    ----------
    `self`: an object of class CA

    Return
    -------
    a namedtuple of pandas DataFrame containing the results for the active rows including: 

    `coord`: factor coordinates

    `cos2`: squared cosinus

    `contrib`: relative contributions

    `infos`: additionnal informations (weights, margin, square distance to origin and inertia)
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2 = load_children2()
    >>> from scientisttools import CA, get_ca_row
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children2)
    >>> #extract the results of rows
    >>> row = get_ca_row(res_ca)
    >>> row.coord.head() #row coordinates
    >>> row.cos2.head() #rows cos2
    >>> row.conrib.head() #row contributions
    >>> row.infos.head() #row additionals informations
    ```
    """
    #Check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    return self.row_
            
def get_ca_col(self)-> NamedTuple:
    """
    Extract the results for columns - CA
    ------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) of the active columns variables from Correspondence Analysis (CA) outputs.

    Usage
    -----
    ```python
    >>> get_ca_col(self)
    ```
    
    Parameters
    ----------
    `self`: an object of class CA

    Returns
    -------
    namedtuple of dataframes containing the results for the columns including : 

    `coord`: factor coordinates

    `cos2`: square cosinus

    `contrib`: relative contributions
    
    `infos`: additionals informations (margin, square distance to origin and inertia)
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2  = load_children2()
    >>> from scientisttools import CA, get_ca_col
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8)
    >>> res_ca.fit(children2)
    >>> #extract the results of columns
    >>> col = get_ca_col(res_ca)
    >>> col.coord.head() #column coordinates
    >>> col.cos2.head() #column cos2
    >>> col.contrib.head() #column contributions
    >>> col.infos.head() #column additionals informations
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    return self.col_

def get_ca(self, element = "row")-> NamedTuple:
    """
    Extract the results for rows/columns - CA
    -----------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, square cosinus, relative contributions) for the active row/column variables from Correspondence Analysis (CA) outputs.

        * `get_ca()`: Extract the results for rows and columns
        * `get_ca_row()`: Extract the results for rows only
        * `get_ca_col()`: Extract the results for columns only

    Usage
    -----
    ```python
    >>> get_ca(self, element = ("row", "col"))
    >>> get_ca_row(self)
    >>> get_ca_col(self) 
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `element`: the element to subset from the output. Possible values are : 
        * "row" for active rows
        * "col" for active columns

    Return
    ------
    namedtuple of dataframes containing the results for the active rows/columns including :

    `coord`: factor coordinates for the rows/columns

    `cos2`: square cosinus for the rows/columns

    `contrib`: relative contributions of the rows/columns

    `infos`: additionals informations of the rows/columns

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2 = load_children2()
    >>> from scientisttools import CA, get_ca
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children2)
    >>> #extract the results of rows
    >>> row = get_ca(res_ca, element = "row")
    >>> row.coord.head() #row coordinates
    >>> row.cos2.head() #rows cos2
    >>> row.conrib.head() #row contributions
    >>> row.infos.head() #row additionals informations
    >>> #extract the results of columns
    >>> col = get_ca(res_ca, element = "col")
    >>> col.coord.head() #column coordinates
    >>> col.cos2.head() #column cos2
    >>> col.contrib.head() #column contributions
    >>> col.infos.head() #column additionals informations
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    if element == "row":
        return get_ca_row(self)
    elif element == "col":
        return get_ca_col(self)
    else:
        raise ValueError("'element' should be one of 'row', 'col'")
    
def summaryCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt="pipe",**kwargs):
    """
    Printing summaries of Correspondence Analysis model
    ---------------------------------------------------

    Description
    -----------
    Printing summaries of correspondence analysis (CA) objects

    Usage
    -----
    ```python
    >>> summaryCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt="pipe",**kwargs)
    ```
    
    Parameters
    ----------
    `self`: an object of class CA

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
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2 = load_children2()
    >>> from scientisttools import CA, summaryCA
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children2)
    >>> summaryCA(res_ca)
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")

    #set number of components and number of elements
    ncp, nb_element = min(ncp,self.call_.n_components), min(nb_element,self.call_.X.shape[0])

    #----------------------------------------------------------------------------------------------------------------------------------------
    #Correspondence Analysis Results
    #----------------------------------------------------------------------------------------------------------------------------------------
    print("                     Correspondence Analysis - Results                     \n")

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add eigenvalues informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    print("Eigenvalues")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add chi-squared test
    #----------------------------------------------------------------------------------------------------------------------------------------
    print("\nChi-squared test\n")
    chi2_test = self.goodness_.chi2
    if to_markdown:
        print(chi2_test.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(chi2_test)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add log-likelihood test
    #----------------------------------------------------------------------------------------------------------------------------------------
    print("\nlog-likelihood test\n")
    g_test = self.goodness_.gtest
    if to_markdown:
        print(g_test.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(g_test)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##others measure of association
    #----------------------------------------------------------------------------------------------------------------------------------------
    print("\nOthers measure of association\n")
    association = self.goodness_.association.round(decimals=digits)
    if to_markdown:
        print(association.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(association)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add rows informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    row = self.row_
    if row.coord.shape[0] > nb_element:
        print(f"\nRows (the {nb_element} first)\n")
    else:
        print("\nRows\n")
    row_infos = row.infos
    for i in range(ncp):
        row_coord, row_cos2, row_ctr = row.coord.iloc[:,i], row.cos2.iloc[:,i], row.contrib.iloc[:,i]
        row_cos2.name, row_ctr.name = "cos2", "ctr"
        row_infos = concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary rows informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"row_sup_"):
        row_sup = self.row_sup_
        if row_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary rows (the {nb_element} first)\n")
        else:
            print("\nSupplementary rows\n")
        row_sup_infos = row_sup.dist
        for i in range(ncp):
            row_sup_coord, row_sup_cos2 = row_sup.coord.iloc[:,i], row_sup.cos2.iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add columns informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    col = self.col_
    if col.coord.shape[0] > nb_element:
        print(f"\nColumns (the {nb_element} first)\n")
    else:
        print("\nColumns\n")
    col_infos = col.infos
    for i in range(ncp):
        col_coord, col_cos2, col_ctr = col.coord.iloc[:,i], col.cos2.iloc[:,i], col.contrib.iloc[:,i]
        col_cos2.name, col_ctr.name = "cos2", "ctr"
        col_infos = concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary columns informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"col_sup_"):
        col_sup = self.col_sup_
        if col_sup.coord.shape[0]> nb_element:
            print(f"\nSupplementary columns (the {nb_element} first)\n")
        else:
            print("\nSupplementary columns\n")
        col_sup_infos = col_sup.dist
        for i in range(ncp):
            col_sup_coord, col_sup_cos2 = col_sup.coord.iloc[:,i], col_sup.cos2.iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos = concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary quantitatives informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quanti_sup_"):
        quanti_sup = self.quanti_sup_
        if quanti_sup.coord.shape[0]> nb_element:
            print(f"\nSupplementary quantitatives columns (the {nb_element})\n")
        else:
            print("\nSupplementary quantitatives columns\n")
        quanti_sup_infos = DataFrame().astype("float")
        for i in range(ncp):
            quanti_sup_coord, quanti_sup_cos2 = quanti_sup.coord.iloc[:,i], quanti_sup.cos2.iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos = concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##add supplementary qualitatives informations
    #----------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(self,"quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = quali_sup.dist
        for i in range(ncp):
            quali_sup_coord, quali_sup_cos2, quali_sup_vtest = quali_sup.coord.iloc[:,i], quali_sup.cos2.iloc[:,i], quali_sup.vtest.iloc[:,i]
            quali_sup_cos2.name, quali_sup_vtest.name = "cos2", "vtest"
            quali_sup_infos = concat([quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest],axis=1)
        quali_sup_infos = quali_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        if quali_sup.eta2.shape[0] > nb_element:
            print(f"\nSupplementary qualitatives variables (eta2) (the {nb_element} first)\n")
        else:
            print("\nSupplementary qualitatives variables (eta2)\n")
        quali_sup_eta2 = quali_sup.eta2.iloc[:nb_element,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)