# -*- coding: utf-8 -*-
import pandas as pd

def get_mfa_ind(self):
    """
    Extract the results for individuals - MFA
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals
    from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    self : an object of class MFA, MFAQUAL, MFAMIX

    Return
    ------
    a dictionnary of dataframes containing the results for the active individuals including :
    coord	: coordinates for the individuals
    
    cos2	: cos2 for the individuals
    
    contrib	: contributions of the individuals
    
    inertia	: inertia of the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX, MFACT")
    return self.ind_

def get_mfa_var(self,choice = "group"):
    """
    Extract the results for variables (quantitatives and groups) MFA
    ------------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active quantitative variable/groups 
    from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    self  : an object of class MFA
    
    choice : the element to subset from the output. Possible values are "quanti_var","group","quali_var"

    Value
    -----
    a dictionnary of dataframes containing the results for the active quantitative variable/groups including :

    coord	: coordinates for the quantitative variable/groups
    
    cos2	: cos2 for the quantitative variable/groups
    
    contrib	: contributions of the quantitative variable/groups

    Usage
    -----

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ not in ["mfa","mfaqual","mfamix"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX")
    
    if choice not in ["group","quanti_var","quali_var"]:
        raise ValueError("'choice' should be one of 'group', 'quanti_var', 'quali_var'")
    
    if choice == "group":
        return self.group_
    if choice == "quanti_var":
        if self.model_ in ["mfa","mfamix"]:
            return self.quanti_var_
        elif self.model_ == "mfaqual":
            if hasattr(self, "quanti_var_sup_"):
                return self.quanti_var_sup_
            else:
                raise ValueError("No quantitative variable")
    if choice == "quali_var":
        if self.model_ in ["mfaqual","mfamix"]:
            return self.quali_var_
        elif self.model_ == "mfa":
            if hasattr(self, "quali_var_sup_"):
                return self.quali_var_sup_
            else:
                raise ValueError("No categorical variable")
        
def get_mfa_freq(self):
    """
    Extract the results for frequence - MFACT
    -----------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active frequence
    from Multiple Factor Analysis for contingence tables (MFACT) outputs.

    Parameters
    ----------
    self : an object of class MFACT

    Return
    ------
    a dictionnary of dataframes containing the results for the active frequence including :
    coord	: coordinates for the frequences
    
    cos2	: cos2 for the frequences
    
    contrib	: contributions of the frequences

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfact":
        raise TypeError("'self' must be an object of class MFACT")
    return self.freq_
        
def get_mfa_partial_axes(self):
    """
    Extract the results for partial axes - MFA
    ------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active 
    partial axes from Multiple Factor Analysis (MFA, MFAQUAL, MFAMIX) outputs.

    Parameters
    ----------
    self : an object of class MFA, MFAQUAL, MFAMIX

    Return
    ------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    
    """
    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX, MFACT")
    
    return self.partial_axes_

def get_mfa(self,choice="ind"):
    """
    Extract the results for individuals/variables/group/partial axes - MFA
    ----------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals/quantitative 
    variables/groups/partial axes from Multiple Factor Analysis (MFA) outputs.
    
    * get_mfa(): Extract the results for variables and individuals
    
    * get_mfa_ind(): Extract the results for individuals only
    
    * get_mfa_var(): Extract the results for variables (quantitatives, qualitatives and groups)

    * get_mfa_freq(): Extract the results for frequencies
    
    * get_mfa_partial_axes(): Extract the results for partial axes only

    Parameters
    ----------
    self : an object of class MFA, MFAQUAL, MFAMIX, MFACT

    choice : he element to subset from the output. Possible values are "ind", "quanti_var", "group", 'quali_var', 'freq' or "partial_axes".

    Return
    ------
    a dictionary of dataframes containing the results for the active individuals/quantitative variable groups/partial axes including :

    coord	: coordinates for the individuals/quantitative variable/groups/partial axes
    
    cos2	: cos2 for the individuals/quantitative variable/groups/partial axes

    contrib	: contributions of the individuals/quantitative variable/groups/partial axes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ not in ["mfa","mfaqual","mfamix","mfact"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX, MFACT")

    if choice not in ["ind","quanti_var","group","quali_var","freq","partial_axes"]:
        raise ValueError("'choice' should be one of 'ind', 'quanti_var', 'group', 'quali_var', 'freq','partial_axes'")
    
    if choice == "ind":
        return get_mfa_ind(self)
    elif choice == "partial_axes":
        return get_mfa_partial_axes(self)
    elif choice == "freq":
        return get_mfa_freq(self)
    else:
        return get_mfa_var(self,choice=choice)
    
def summaryMFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis model
    ----------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFA

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format.

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.coms
    """

    if self.model_ != "mfa":
        raise TypeError("'self' must be an object of class MFA")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                     Multiple Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    for i in range(ncp):
        group_coord = group["coord"].iloc[:,i]
        group_contrib = group["contrib"].iloc[:,i]
        group_contrib.name = "ctr"
        group_cos2 = group["cos2"].iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if hasattr(self, "ind_sup_"):
        if self.ind_sup_["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in range(ncp):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.quanti_var_["coord"].shape[0] > nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
        print("\nContinuous variables\n")
    quanti_var = self.quanti_var_
    quanti_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quanti_var_coord = quanti_var["coord"].iloc[:,i]
        quanti_var_contrib = quanti_var["contrib"].iloc[:,i]
        quanti_var_contrib.name = "ctr"
        quanti_var_cos2 = quanti_var["cos2"].iloc[:,i]
        quanti_var_cos2.name = "cos2"
        quanti_var_infos = pd.concat((quanti_var_infos,quanti_var_coord,quanti_var_contrib,quanti_var_cos2),axis=1)
    quanti_var_infos = quanti_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quanti_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quanti_var_infos)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        if hasattr(self, "quanti_var_sup_"):
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)
        
        if hasattr(self, "quali_var_sup_"):
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories\n")
            quali_var_sup = self.quali_var_sup_
            quali_var_sup_infos = quali_var_sup["dist"]
            for i in range(ncp):
                quali_var_sup_coord = quali_var_sup["coord"].iloc[:,i]
                quali_var_sup_cos2 = quali_var_sup["cos2"].iloc[:,i]
                quali_var_sup_cos2.name = "cos2"
                quali_var_sup_vtest = quali_var_sup["vtest"].iloc[:,i]
                quali_var_sup_vtest.name = "vtest"
                quali_var_sup_infos = pd.concat((quali_var_sup_infos,quali_var_sup_coord,quali_var_sup_cos2,),axis=1)
            quali_var_sup_infos = quali_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_infos)

            # Add correlation ratio
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories (eta2)\n")
            quali_var_sup_eta2 = quali_var_sup["eta2"].iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_eta2)

def summaryMFAQUAL(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis for Qualitatives Variables model
    -------------------------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFAQUAL

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format.

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfaqual":
        raise TypeError("'self' must be an object of class MFAQUAL")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                  Multiple Factor Analysis for Qualitatives Variables - Results                   \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    for i in range(ncp):
        group_coord = group["coord"].iloc[:,i]
        group_contrib = group["contrib"].iloc[:,i]
        group_contrib.name = "ctr"
        group_cos2 = group["cos2"].iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if hasattr(self, "ind_sup_"):
        if self.ind_sup_["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in range(ncp):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.quali_var_["coord"].shape[0] > nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
        print("\nCategories\n")
    quali_var = self.quali_var_
    quali_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quali_var_coord = quali_var["coord"].iloc[:,i]
        quali_var_contrib = quali_var["contrib"].iloc[:,i]
        quali_var_contrib.name = "ctr"
        quali_var_cos2 = quali_var["cos2"].iloc[:,i]
        quali_var_cos2.name = "cos2"
        quali_var_infos = pd.concat((quali_var_infos,quali_var_coord,quali_var_contrib,quali_var_cos2),axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)
    
    # Add correlation ratio
    if self.quali_var_["eta2"].shape[0] > nb_element:
        print(f"\nCategories eta2 (the {nb_element} first)\n")
    else:
        print("\nCategories (eta2)\n")
    quali_var_eta2 = quali_var["eta2"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_eta2)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        # Add supplementary qualitatives variables
        if hasattr(self, "quali_var_sup_"):
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories\n")
            quali_var_sup = self.quali_var_sup_
            quali_var_sup_infos = quali_var_sup["dist"]
            for i in range(ncp):
                quali_var_sup_coord = quali_var_sup["coord"].iloc[:,i]
                quali_var_sup_cos2 = quali_var_sup["cos2"].iloc[:,i]
                quali_var_sup_cos2.name = "cos2"
                quali_var_sup_vtest = quali_var_sup["vtest"].iloc[:,i]
                quali_var_sup_vtest.name = "vtest"
                quali_var_sup_infos = pd.concat((quali_var_sup_infos,quali_var_sup_coord,quali_var_sup_cos2,),axis=1)
            quali_var_sup_infos = quali_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_infos)

            # Add correlation ratio
            if self.quali_var_sup_["eta2"].shape[0] > nb_element:
                print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories (eta2)\n")
            quali_var_sup_eta2 = quali_var_sup["eta2"].iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_eta2)

        # Add supplementary quantitatives variables
        if hasattr(self, "quanti_var_sup_"):
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)

def summaryMFAMIX(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis of Mixed Data model
    ------------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFAMIX

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfamix":
        raise TypeError("'self' must be an object of class MFAMIX")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                  Multiple Factor Analysis of Mixed Data - Results                   \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    for i in range(ncp):
        group_coord = group["coord"].iloc[:,i]
        group_contrib = group["contrib"].iloc[:,i]
        group_contrib.name = "ctr"
        group_cos2 = group["cos2"].iloc[:,i]
        group_cos2.name = "cos2"
        group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if hasattr(self, "ind_sup_"):
        if self.ind_sup_["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in range(ncp):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.quanti_var_["coord"].shape[0] > nb_element:
        print(f"\nContinuous variables (the {nb_element} first)\n")
    else:
        print("\nContinuous variables\n")
    quanti_var = self.quanti_var_
    quanti_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quanti_var_coord = quanti_var["coord"].iloc[:,i]
        quanti_var_contrib = quanti_var["contrib"].iloc[:,i]
        quanti_var_contrib.name = "ctr"
        quanti_var_cos2 = quanti_var["cos2"].iloc[:,i]
        quanti_var_cos2.name = "cos2"
        quanti_var_infos = pd.concat((quanti_var_infos,quanti_var_coord,quanti_var_contrib,quanti_var_cos2),axis=1)
    quanti_var_infos = quanti_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quanti_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quanti_var_infos)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        if self.quanti_var_sup_ is not None:
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)
    
    # Add continuous variables
    if self.quali_var_["coord"].shape[0] > nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
        print("\nCategories\n")
    quali_var = self.quali_var_
    quali_var_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        quali_var_coord = quali_var["coord"].iloc[:,i]
        quali_var_contrib = quali_var["contrib"].iloc[:,i]
        quali_var_contrib.name = "ctr"
        quali_var_cos2 = quali_var["cos2"].iloc[:,i]
        quali_var_cos2.name = "cos2"
        quali_var_infos = pd.concat((quali_var_infos,quali_var_coord,quali_var_contrib,quali_var_cos2),axis=1)
    quali_var_infos = quali_var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_infos)
    
    # Add correlation ratio
    if self.quali_var_["eta2"].shape[0] > nb_element:
        print(f"\nCategories eta2 (the {nb_element} first)\n")
    else:
        print("\nCategories (eta2)\n")
    quali_var_eta2 = quali_var["eta2"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(quali_var_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(quali_var_eta2)

    # Add Supplementary quantitatives variables
    if self.num_group_sup is not None:
        # Add supplementary qualitatives variables
        if hasattr(self, "quali_var_sup_"):
            if self.quali_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary categories (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories\n")
            quali_var_sup = self.quali_var_sup_
            quali_var_sup_infos = quali_var_sup["dist"]
            for i in range(ncp):
                quali_var_sup_coord = quali_var_sup["coord"].iloc[:,i]
                quali_var_sup_cos2 = quali_var_sup["cos2"].iloc[:,i]
                quali_var_sup_cos2.name = "cos2"
                quali_var_sup_vtest = quali_var_sup["vtest"].iloc[:,i]
                quali_var_sup_vtest.name = "vtest"
                quali_var_sup_infos = pd.concat((quali_var_sup_infos,quali_var_sup_coord,quali_var_sup_cos2,),axis=1)
            quali_var_sup_infos = quali_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_infos)

            # Add correlation ratio
            if self.quali_var_sup_["eta2"].shape[0] > nb_element:
                print(f"\nSupplementary categories eta2 (the {nb_element} first)\n")
            else:
                print("\nSupplementary categories (eta2)\n")
            quali_var_sup_eta2 = quali_var_sup["eta2"].iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quali_var_sup_eta2.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quali_var_sup_eta2)
        
        if hasattr(self, "quanti_var_sup_"):
            if self.quanti_var_sup_["coord"].shape[0] > nb_element:
                print(f"\nSupplementary Continuous variables (the {nb_element} first)\n")
            else:
                print("\nSupplementary Continuous variables\n")
            quanti_var_sup = self.quanti_var_sup_
            quanti_var_sup_infos = pd.DataFrame().astype("float")
            for i in range(ncp):
                quanti_var_sup_coord = quanti_var_sup["coord"].iloc[:,i]
                quanti_var_sup_cos2 = quanti_var_sup["cos2"].iloc[:,i]
                quanti_var_sup_cos2.name = "cos2"
                quanti_var_sup_infos = pd.concat((quanti_var_sup_infos,quanti_var_sup_coord,quanti_var_sup_cos2),axis=1)
            quanti_var_sup_infos = quanti_var_sup_infos.iloc[:nb_element,:].round(decimals=digits)
            if to_markdown:
                print(quanti_var_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
            else:
                print(quanti_var_sup_infos)


def summaryMFACT(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis for contiengence tables model
    ----------------------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFACT

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format.

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.coms
    """

    if self.model_ != "mfact":
        raise TypeError("'self' must be an object of class MFACT")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                     Multiple Factor Analysis for Contingence Tables - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = self.eig_.iloc[:self.call_["n_components"],:].T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative of % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add group informations
    print("\nGroups\n")
    group = self.group_
    group_infos = group["dist2"]
    # for i in range(ncp):
    #     group_coord = group["coord"].iloc[:,i]
    #     group_contrib = group["contrib"].iloc[:,i]
    #     group_contrib.name = "ctr"
    #     group_cos2 = group["cos2"].iloc[:,i]
    #     group_cos2.name = "cos2"
    #     group_infos = pd.concat((group_infos,group_coord,group_contrib,group_cos2),axis=1)
    group_infos = group_infos.to_frame().round(decimals=digits)
    if to_markdown:
        print(group_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(group_infos)
    
    if self.num_group_sup is not None:
        print("\nSupplementary groups\n")
        group_sup_infos = group["dist2_sup"]
        for i in range(ncp):
            group_sup_coord = group["coord_sup"].iloc[:,i]
            group_sup_cos2 = group["cos2_sup"].iloc[:,i]
            group_sup_cos2.name = "cos2"
            group_sup_infos = pd.concat((group_sup_infos,group_sup_coord,group_sup_cos2),axis=1)
        group_sup_infos = group_sup_infos.round(decimals=digits)
        if to_markdown:
            print(group_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(group_sup_infos)

    # Add individuals informations
    if self.ind_["coord"].shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
        print("\nIndividuals\n")
    ind = self.ind_
    ind_infos = ind["infos"]
    for i in range(ncp):
        ind_coord = ind["coord"].iloc[:,i]
        ind_contrib = ind["contrib"].iloc[:,i]
        ind_contrib.name = "ctr"
        ind_cos2 = ind["cos2"].iloc[:,i]
        ind_cos2.name = "cos2"
        ind_infos = pd.concat((ind_infos,ind_coord,ind_contrib,ind_cos2),axis=1)
    ind_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add supplementary individuals
    if hasattr(self, "ind_sup_"):
        if self.ind_sup_["coord"].shape[0] > nb_element:
            print(f"\nSupplementary individuals (the {nb_element} first)\n")
        else:
            print("\nSupplementary individuals\n")
        ind_sup = self.ind_sup_
        ind_sup_infos = ind_sup["dist"]
        for i in range(ncp):
            ind_sup_coord = ind_sup["coord"].iloc[:,i]
            ind_sup_cos2 = ind_sup["cos2"].iloc[:,i]
            ind_sup_cos2.name = "cos2"
            ind_sup_infos = pd.concat((ind_sup_infos,ind_sup_coord,ind_sup_cos2),axis=1)
        ind_sup_infos = ind_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(ind_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(ind_sup_infos)
    
    # Add continuous variables
    if self.freq_["coord"].shape[0] > nb_element:
        print(f"\nFrequencies (the {nb_element} first)\n")
    else:
        print("\nFrequencies\n")
    freq = self.freq_
    freq_infos = pd.DataFrame().astype("float")
    for i in range(ncp):
        freq_coord = freq["coord"].iloc[:,i]
        freq_contrib = freq["contrib"].iloc[:,i]
        freq_contrib.name = "ctr"
        freq_cos2 = freq["cos2"].iloc[:,i]
        freq_cos2.name = "cos2"
        freq_infos = pd.concat((freq_infos,freq_coord,freq_contrib,freq_cos2),axis=1)
    freq_infos = freq_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(freq_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(freq_infos)

    # Add Supplementary quantitatives variables
    if hasattr(self, "freq_sup_"):
        if self.freq_sup_["coord"].shape[0] > nb_element:
            print(f"\nSupplementary frequencies (the {nb_element} first)\n")
        else:
            print("\nSupplementary frequencies\n")
        freq_sup = self.freq_sup_
        freq_sup_infos = pd.DataFrame().astype("float")
        for i in range(ncp):
            freq_sup_coord = freq_sup["coord"].iloc[:,i]
            freq_sup_cos2 = freq_sup["cos2"].iloc[:,i]
            freq_sup_cos2.name = "cos2"
            freq_sup_infos = pd.concat((freq_sup_infos,freq_sup_coord,freq_sup_cos2),axis=1)
        freq_sup_infos = freq_sup_infos.iloc[:nb_element,:].round(decimals=digits)
        if to_markdown:
            print(freq_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(freq_sup_infos)