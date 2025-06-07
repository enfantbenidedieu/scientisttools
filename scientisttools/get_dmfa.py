# -*- coding: utf-8 -*-
import pandas as pd

def get_dmfa_ind(self):
    """
    Extract the results for individuals - DMFA
    ------------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosinus and relative contributions) for the active individuals from Dual Multiple Factor Analysis (DMFA) outputs.

    Usage
    -----
    ```python
    >>> get_dmfa_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    Returns
    -------
    dictionnary of dataframes containing all the results for the active individuals including :

    `coord` : factor coordinates for the individuals
    
    `cos2` : square cosinus for the individuals
    
    `contrib` : relative contributions of the individuals
    
    `inertia` : inertia of the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, get_mfa_ind
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> # Results for individuals
    >>> ind = get_mfa_ind(res_mfa)
    >>> 
    ```
    """
    # Check if self is not an instance DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    return self.ind_

def get_dmfa_var(self,choice = "group"):
    """
    Extract the results for variables - DMFA
    ----------------------------------------

    Description
    -----------
    Extract all the results (factor coordinates, squared cosinus and relative contributions) for the active variables/groups/partiel variables from Dual Multiple Factor Analysis (DMFA) outputs.

    Usage
    -----
    ```python
    >>> get_dmfa_var(self)
    ```

    Parameters
    ----------
    `self` : an object of class DMFA
    
    `choice` : the element to subset from the output. Possible values are :
        * "quanti_var" for active quantitative variables
        * "group" for group
        * "var_partiel" for partial coordinates

    Returns
    -------
    dictionnary of dataframes containing all the results for the active variables/groups/partial coordinates including :

    `coord` : factor coordinates for the variables/groups/partial coordinates
    
    `cos2` : square cosinus for the variables/groups/partial coordinates
    
    `contrib` : relative contributions of the variables/groups/partial coordinates

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, get_mfa_var
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> # Results for quantitatives variables
    >>> quanti_var = get_mfa_var(res_mfa, choice = "quanti_var")
    ```
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")
    
    if choice not in ["group","quanti_var","var_partiel"]:
        raise ValueError("'choice' should be one of 'group', 'quanti_var', 'var_partiel'")
    
    # for group informations
    if choice == "group":
        return self.group_
    
    # For quantitatives variables
    if choice == "quanti_var":
        return self.var_
    
    # For frequencies
    if choice == "var_partiel":
        return self.var_partiel_
        
def get_dmfa(self,choice="ind"):
    """
    Extract the results for individuals/variables/groups/frequencies/partial axes - DMFA
    -----------------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active individuals/variables/groups/frequencies/partial axes from Multiple Factor Analysis (MFA, MFAQUAL, MFAMIX, MFACT) outputs.
    
    * get_dmfa(): Extract the results for variables and individuals
    
    * get_dmfa_ind(): Extract the results for individuals only
    
    * get_dmfa_var(): Extract the results for variables (quantitatives, qualitatives, groups and frequencies)
    
    Usage
    -----
    ```python
    >>> get_dmfa(self, choice = ("ind","group","quanti_var","var_partiel"))
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

    `choice` : the element to subset from the output. Possible values are :
        * "ind" for active individuals 
        * "quanti_var" for quantitative variables
        * "group" for groups
        * 'var_partiel' for partial coordinates

    Returns
    -------
    namedtuple of dataframes containing all the results for the active individuals/variables/groups/frequencies/partial axes including :

    `coord`	: factor coordinates for the individuals/variables/groups/frequencies/partial axes
    
    `cos2` : square cosinus for the individuals/variables/groups/frequencies/partial axes

    `contrib` : relative contributions of the individuals/variables/groups/frequencies/partial axes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    see get_dmfa_ind, get_dmfa_var
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")

    if choice not in ["ind","quanti_var","group","var_partial"]:
        raise ValueError("'choice' should be one of 'ind', 'quanti_var', 'group', 'var_partial'")
    
    if choice == "ind":
        return get_dmfa_ind(self)
    else:
        return get_dmfa_var(self,choice=choice)
    
def summaryDMFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Dual Multiple Factor Analysis objects
    ------------------------------------------------------

    Description
    -----------
    Printing summaries of dual multiple factor analysis (DMFA) objects

    Usage
    -----
    ```python
    >>> summaryDMFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class DMFA

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
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA, summaryMFA
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> summaryMFA(res_mfa)
    ```
    """
    # Check if self is an object of class DMFA
    if self.model_ != "dmfa":
        raise TypeError("'self' must be an object of class DMFA")

    ncp = min(ncp,self.call_.n_components)
    nb_element = min(nb_element,self.ind_.coord.shape[0])

    # Principal Components Analysis Results
    print("                     Dual Multiple Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_.n_components,:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = pd.DataFrame()
    for i in range(ncp):
        group_coord = group.coord.iloc[:,i]
        group_coord_n = group.coord_n.iloc[:,i]
        group_coord_n.name = "coord_n."+str(i+1)
        group_cos2 = group.cos2.iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_coord_n,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    # Add individuals informations
    if self.ind_.coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind.infos
    for i in range(ncp):
        ind_coord = ind.coord.iloc[:,i]
        ind_contrib = ind.contrib.iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind.cos2.iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # For quantitatives variables
    var = self.var_
    if var.coord.shape[0] > nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
        print("\nContinuous variables\n")
    var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        var_coord = var.coord.iloc[:,i]
        var_contrib = var.contrib.iloc[:,i]
        var_contrib.name = "ctr"
        var_cos2 = var.cos2.iloc[:,i]
        var_cos2.name = "cos2"
        var_infos = pd.concat((var_infos,var_coord,var_contrib,var_cos2),axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
        
    # For supplementary qualitatives variables
    if hasattr(self, "quali_sup_"):
        quali_sup = self.quali_sup_
        if quali_sup.coord.shape[0] > nb_element:
            print(f"\nSupplementary categories (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories\n")
        quali_sup_infos = pd.DataFrame().astype("float")
        for i in range(ncp):
            quali_sup_coord = quali_sup.coord.iloc[:,i]
            quali_sup_cos2 = quali_sup.cos2.iloc[:,i]
            quali_sup_cos2.name = "cos2"
            quali_sup_vtest = quali_sup.vtest.iloc[:,i]
            quali_sup_vtest.name = "vtest"
            quali_sup_infos = pd.concat((quali_sup_infos,quali_sup_coord,quali_sup_cos2,),axis=1)
        quali_sup_infos = quali_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_infos)

        # Add correlation ratio
        if self.quali_sup_.eta2.shape[0] > nb_element:
            print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
        else:
            print("\nSupplementary categories (eta2)\n")
        quali_sup_eta2 = quali_sup.eta2.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(quali_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(quali_sup_eta2)