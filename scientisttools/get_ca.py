# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def get_ca_row(self,choice="row")-> dict:
    """
    Extract the resultst for rows - CA
    ----------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions and inertia) for the active/supplementary 
    row variables from Correspondence Analysis (CA) outputs.

    Parameters
    ----------
    self. : an object of class CA

    choice : the element to subset from the output. Allowed values are "row" (for active rows) or "row_sup" (for supplementary rows).

    Returns
    -------
    a dictionary of dataframes containing the results for the active rows including :
    coord   : coordinates for the rows of shape (n_rows, n_components)
    cos2    : cos2 for the rows of shape (n_rows, n_components)
    contrib : contributions for the rows of shape (n_rows, n_components)
    infos   : additionnal informations for the rows:
                - square root distance between rows and inertia
                - marge for the rows
                - inertia for the rows
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "ca":
        raise ValueError("'self' must be an object of class CA.")

    if choice not in ["row", "row_sup"]:
        raise ValueError("'choice' should be one of 'row', 'row_sup'")
    
    if choice == "row":
        return self.row_
    elif choice == "row_sup":
        if self.row_sup is None:
            raise ValueError("Error : No supplementary rows")
        return self.row_sup_
            
def get_ca_col(self,choice="col")-> dict:

    """
    Extract the results for columns - CA
    ------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions and inertia) for the active column variables from Correspondence Analysis (CA) outputs.

    Parameters
    ----------
    self : an object of class CA

    choice : 

    Returns
    -------
    a dictionary of dataframes containing the results for the active columns including :
    coord   : coordinates for the columns of shape (n_cols, n_components)
    cos2    : cos2 for the columns of shape (n_cols, n_components)
    contrib : contributions for the columns of shape (n_cols, n_components)
    infos   : additionnal informations for the columns:
                - square root distance between columns and inertia
                - marge for the columns
                - inertia for the columns
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA.")
    
    if choice not in ["col","col_sup","quanti_sup","quali_sup"]:
        raise ValueError("Error : 'choice' should be one of 'col', 'col_sup', 'quanti_sup', 'quali_sup'")
    
    if choice == "col":
        return self.col_
    elif choice == "col_sup":
        if self.col_sup is None:
            raise ValueError("Error : No supplementary columns")
        return self.col_sup_
    elif choice == "quanti_sup":
        if self.quanti_sup is None:
            raise ValueError("Error : No supplementary quantitatives columns")
        return self.quanti_sup_
    elif choice == "quali_sup":
        if self.quali_sup is None:
            raise ValueError("Error : No supplementary qualitatives columns")

def get_ca(self,choice = "row")-> dict:
    """
    Extract the results for rows/columns - CA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine, contributions and inertia) for the active row/column variables from Correspondence Analysis (CA) outputs.

    * get_ca() : Extract the results for rows and columns
    * get_ca_row() : Extract the results for rows only
    * get_ca_col() : Extract the results for columns only

    Parameters
    ----------
    self : an object of class CA

    choice : the element to subset from the output. Possible values are "row" or "col", "biplot"

    Return
    ------
    a dictionary of dataframes containing the results for the active rows/columns including :
    coord   : coordinates for the rows/columns
    cos2    : cos2 for the rows/columns
    contrib	: contributions of the rows/columns
    infos   : additionnal informations for the row/columns:
                - square root distance between rows/columns and inertia
                - marge for the rows/columns
                - inertia for the rows/columns

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "ca":
        raise ValueError("'self' must be an object of class CA")
    
    if choice not in ["row","row_sup","col","col_sup","quanti_sup","quali_sup"]:
        raise ValueError("'choice' should be one of 'row', 'row_sup', 'col', 'col_sup', 'quanti_sup', 'quali_sup'")
    
    if choice in ["row","row_sup"]:
        return get_ca_row(self,choice=choice)
    else:
        return get_ca_col(self,choice=choice)
    

def summaryCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt="pipe",**kwargs):
    """
    Printing summaries of Correspondence Analysis model
    ---------------------------------------------------

    Parameters
    ----------
    self        :   an obect of class CA

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format.

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "ca":
        raise ValueError("Error : 'self' must be an object of class CA")

    # Set number of components
    ncp = min(ncp,self.call_["n_components"])
    # Set number of elements
    nb_element = min(nb_element,self.call_["X"].shape[0])

    # Principal Components Analysis Results
    print("                     Correspondence Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nRows\n")
    row = get_ca(self,choice="row")
    row_infos = row["infos"]
    for i in np.arange(0,ncp,1):
        row_coord = row["coord"].iloc[:,i]
        row_cos2 = row["cos2"].iloc[:,i]
        row_cos2.name = "cos2"
        row_ctr = row["contrib"].iloc[:,i]
        row_ctr.name = "ctr"
        row_infos = pd.concat([row_infos,row_coord,row_ctr,row_cos2],axis=1)
    row_infos = row_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)

    # Add supplementary individuals
    if self.row_sup is not None:
        print(f"\nSupplementary rows\n")
        # Save all informations
        row_sup = self.row_sup_
        row_sup_infos = row_sup["dist"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2  = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)

    # Add variables informations
    print(f"\nColumns\n")
    col = get_ca(self,choice="col")
    col_infos = col["infos"]
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_cos2 = col["cos2"].iloc[:,i]
        col_cos2.name = "cos2"
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary columns informations
    if self.col_sup is not None:
        print(f"\nSupplementary columns\n")
        col_sup = self.col_sup_
        col_sup_infos = col_sup["dist"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos = pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)
    
    # Add supplementary quantitatives informations
    if self.quanti_sup is not None:
        print(f"\nSupplementary quantitatives columns\n")
        quanti_sup = self.quanti_sup_
        quanti_sup_infos = pd.DataFrame().astype("float")
        for i in np.arange(0,ncp,1):
            quanti_sup_coord = quanti_sup["coord"].iloc[:,i]
            quanti_sup_cos2 = quanti_sup["cos2"].iloc[:,i]
            quanti_sup_cos2.name = "cos2"
            quanti_sup_infos = pd.concat([quanti_sup_infos,quanti_sup_coord,quanti_sup_cos2],axis=1)
        quanti_sup_infos = quanti_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quanti_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quanti_sup_infos)
    
    # Add supplementary qualitatives informations
    if self.quali_sup is not None:
        print(f"\nSupplementary categories\n")
        quali_sup = self.quali_sup_
        quali_sup_infos = quali_sup["dist"]
        for i in np.arange(0,ncp,1):
            quali_sup_coord = quali_sup["coord"].iloc[:,i]
            quali_sup_cos2 = quali_sup["cos2"].iloc[:,i]
            quali_sup_cos2.name = "cos2"
            quali_sup_vtest = quali_sup["vtest"].iloc[:,i]
            quali_sup_vtest.name = "vtest"
            quali_sup_infos = pd.concat([quali_sup_infos,quali_sup_coord,quali_sup_cos2,quali_sup_vtest],axis=1)
        quali_sup_infos = quali_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)
        
        # Add supplementary qualitatives - correlation ratio
        print("\nSupplementary qualitatives variables (eta2)\n")
        quali_sup_eta2 = quali_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt))
        else:
            print(quali_sup_eta2)