# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_efa_ind(self,choice = "ind") -> dict:

    """
    Extract the results for individuals - EFA
    -----------------------------------------

    Parameters
    ----------
    self : an object of class EFA

    Returns
    -------
    

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")

    if choice not in ["ind","ind_sup"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup'")
    
    if choice == "ind":
        return self.ind_
    elif choice == "ind_sup":
        if self.ind_sup is None:
            raise ValueError("No supplementary individuals")
        return self.ind_sup_

def get_efa_var(self) -> dict:

    """
    Extract the results for variables - EFA
    ---------------------------------------

    Parameters
    ----------
    self : an instance of class EFA

    Returns
    -------
    

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")
    return self.var_

def get_efa(self,choice = "ind")-> dict:

    """
    Extract the results for individuals/variables - EFA
    ---------------------------------------------------

    Parameters
    ---------
    self : an instance of class EFA

    choice : 

    Returns
    -------
    

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")

    if choice not in ["ind","ind_sup","var"]:
        raise ValueError("'choice' should be one of 'ind', 'ind_sup', 'var'")
    
    if choice in ["ind", "ind_sup"]:
        return get_efa_ind(self,choice=choice)
    elif choice == "var":
        return get_efa_var(self)

def summaryEFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Exploratory Factor Analysis model
    -------------------------------------------------------

    Parameters
    ----------
    self        :   an obect of class EFA

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
    # check if EFA model
    if self.model_ != "efa":
        raise TypeError("'self' must be an object of class EFA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0],self.var_["coord"].shape[0])

    # Exploratory Factor Analysis Results
    print("                     Exploratory Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first) \n")
    ind_coord = self.ind_["coord"].iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(ind_coord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_coord)

    # Add supplementary individuals
    if self.ind_sup is not None:
        nb_elt = min(nb_element,self.ind_sup_["coord"].shape[0])
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        ind_sup_infos = self.ind_sup_["coord"].iloc[:nb_elt,:ncp].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)

    # Add variables informations
    print(f"\nContinues Variables\n")
    var = self.var_
    var_infos = pd.DataFrame().astype("float")
    for i in np.arange(0,ncp,1):
        var_coord = var["coord"].iloc[:,i]
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr"
        var_infos = pd.concat([var_infos,var_coord,var_ctr],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)