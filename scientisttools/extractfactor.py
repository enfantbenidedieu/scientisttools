# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scientisttools.eta2 import eta2
import scipy.stats as st




#############################################################################################
#           Multiple Factor Analysis of mixed data (MFAMIX)
#############################################################################################

def get_mfamix_ind(self):
    """
    Extract the results for individuals - MFAMIX
    --------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals
    from Multiple Factor Analysis of MIXed data (MFAMIX) outputs.

    Parameters
    ----------
    self : an object of class MFAMIX

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
    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")
    
    return self.ind_

def get_mfamix_var(self,choice = "group"):
    """
    Extract the results for variables (quantitatives and groups) MFAMIX
    --------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active quantitative variable/groups 
    from Multiple Factor Analysis of MIXed data (MFAQUAL) outputs.

    Parameters
    ----------
    self  : an object of class MFAMIX
    
    choice : the element to subset from the output. Possible values are "quanti_var","quali_var","group"

    Value
    -----
    a dictionnary of dataframes containing the results for the active quantitative/qualitative variable/groups including :

    coord	: coordinates for the quantitatives/qualitatives variable/groups
    
    cos2	: cos2 for the quantitatives/qualitative variable/groups
    
    contrib	: contributions of the quantitatives/qualitative variable/groups

    Usage
    -----

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")
    
    if choice not in ["quanti_var","quali_var","group"]:
        raise ValueError("Error : 'choice' should be one of 'quanti_var','quali_var', 'group'")
    
    if choice == "group":
        return self.group_
    if choice == "quanti_var":
        return self.quanti_var_
    if choice == "quali_var":
        return self.quali_var_
    
def get_mfamix_partial_axes(self):
    """
    Extract the results for partial axes - MFAMIX
    ---------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active 
    partial axes from Multiple Factor Analysis of MIX data (MFAMIX) outputs.

    Parameters
    ----------
    self : an object of class MFAMIX

    Return
    ------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")
    
    return self.partial_axes_

def get_mfamix(self,choice="ind"):
    """
    Extract the results for individuals/variables/group/partial axes - MFAMIX
    -------------------------------------------------------------------------

    Description
    -----------
    Extract all the results (coordinates, squared cosine and contributions) for the active individuals/quantitative 
    variables/groups/partial axes from Multiple Factor Analysis of MIXed data (MFAMIX) outputs.
    
    * get_mfaqual(): Extract the results for variables and individuals
    
    * get_mfaqual_ind(): Extract the results for individuals only
    
    * get_mfaqual_var(): Extract the results for variables (quantitatives, qualitatives and groups)
    
    * get_mfaqual_partial_axes(): Extract the results for partial axes only

    Parameters
    ----------
    self : an object of class MFAQUAL

    choice : he element to subset from the output. Possible values are "ind", "quali_var", "group" or "partial_axes".

    Return
    ------
    a dictionary of dataframes containing the results for the active individuals/qualitative variable groups/partial axes including :

    coord	: coordinates for the individuals/quantitatives/qualitative variable/groups/partial axes
    
    cos2	: cos2 for the individuals/quantitatives/qualitative variable/groups/partial axes

    contrib	: contributions of the individuals/quantitatives/qualitative variable/groups/partial axes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")

    if choice not in ["ind","quanti_var","quali_var","group","partial_axes"]:
        raise ValueError("Error : 'choice' should be one of 'ind', 'quanti_var','quali_var', 'group', 'partial_axes'")
    
    if choice == "ind":
        return get_mfamix_ind(self)
    elif choice == "partial_axes":
        return get_mfamix_partial_axes(self)
    else:
        return get_mfamix_var(self,choice=choice)
    
def summaryMFAMIX(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Multiple Factor Analysis of MIXed data model
    ------------------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class MFAMIX
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

    if self.model_ != "mfamix":
        raise ValueError("Error : 'self' must be an object of class MFAMIX")

    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.ind_["coord"].shape[0])

    # Principal Components Analysis Results
    print("                  Multiple Factor Analysis of MIXed data - Results                   \n")

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
    if self.ind_sup is not None:
        ind_sup = self.ind_sup_
        ind_sup_infos = ind["dist"]
        for i in range(ncp):
            ind_sup_coord = ind["coord"].iloc[:,i]
            ind_sup_cos2 = ind["cos2"].iloc[:,i]
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
        if self.quali_var_sup_ is not None:
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
        
def StandardScaler(X):
    return (X - X.mean())/X.std(ddof=0)

def get_dist(X, method = "euclidean",normalize=False,**kwargs) -> dict:
    if isinstance(X,pd.DataFrame) is False:
        raise ValueError("Error : 'X' must be a DataFrame")
    if normalize:
        X = X.transform(StandardScaler)
    if method in ["pearson","spearman","kendall"]:
        corr = X.T.corr(method=method)
        dist = corr.apply(lambda cor :  1 - cor,axis=0).values.flatten('F')
    else:
        dist = pdist(X.values,metric=method,**kwargs)
    return dict({"dist" :dist,"labels":X.index})






############# Hierarchical 

def get_hclust(X, method='single', metric='euclidean', optimal_ordering=False):
    Z = hierarchy.linkage(X,method=method, metric=metric)
    if optimal_ordering:
        order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z,X))
    else:
        order = hierarchy.leaves_list(Z)
    return dict({"order":order,"height":Z[:,2],"method":method,
                "merge":Z[:,:2],"n_obs":Z[:,3],"data":X})


################## MDS

def get_mds(self) -> dict:

    """
    self : an object of class MDS

    Returns
    -------
    Multidimensional Scaling - Results 
    ===============================================================
        Names       Description
    1   "coord"     "coordinates"
    2   "res.dist"  "Restitues distances"
    """
    if self.model_ not in ["mds","cmds"]:
        raise ValueError("Error : 'res' must be an object of class MDS or CMDS.")

    # Store informations
    df = dict({
        "coord"     : pd.DataFrame(self.coord_,index=self.labels_,columns=self.dim_index_),
        "res.dist"  : pd.DataFrame(self.res_dist_,index=self.labels_,columns=self.labels_)
    })
    return df





###### PCA
###############################################################################################
#       Canonical Discriminant Analysis (CANDISC)
###############################################################################################


def get_candisc_row(self):
    """
    self. : an instance of class CANDISC

    Returns
    -------
    Canonical Discriminant Analysis Analysis - Results for rows
    =========================================================
        Name        Description
    1   "coord"     "coordinates for the rows"
    """
    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class CANDISC.")

    df = dict({
        "coord" : pd.DataFrame(self.row_coord_,index=self.row_labels_,columns=self.dim_index_)
    })
    return df


def get_candisc_var(self,choice="correlation"):
    """
    self. : an instance of class CANDISC

    Returns
    -------
    Canonical Discriminant Analysis Analysis - Results for rows
    =========================================================
        Name        Description
    """

    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class CANDISC.")
    
    if choice == "correlation":
        df = dict({
            "Total" : self.tcorr_,
            "Between" : self.bcorr_,
            "Within" : self.wcorr_
        })
    elif choice == "covariance":
        df = dict({
            "Total" : self.tcov_,
            "Between" : self.bcov_,
            "Within" : self.wcov_
        })

    return df

def get_candisc(self,choice = "row"):
    """
    self. : an instance of class CANDISC

    Returns
    -------
    Canonical Discriminant Analysis Analysis - Results for rows
    =========================================================
        Name        Description
    1   "coord"     "coordinates for the rows"
    """

    if choice == "row":
        return get_candisc_row(self)
    elif choice == "var":
        return get_candisc_var(self)
    else:
        raise ValueError("Error : Allowed values are either 'row' or 'var'.")
   
def get_candisc_coef(self,choice="absolute"):
    """
    
    """
    if self.model_ != "candisc":
        raise ValueError("Error : 'self' must be an instance of class CANDISC.")
    
    if choice == "absolute":
        coef = pd.concat([pd.DataFrame(self.coef_,index=self.features_labels_,columns=self.dim_index_),
                         pd.DataFrame(self.intercept_.T,columns=["Intercept"],index=self.dim_index_).T],axis=0)
    elif choice == "score":
        coef = pd.concat([pd.DataFrame(self.score_coef_,index=self.features_labels_,columns=self.classes_),
                          pd.DataFrame(self.score_intercept_,index=["Intercept"],columns=self.classes_)],axis=0)
    return coef


def summaryCANDISC(self,digits=3,
                   nb_element=10,
                   ncp=3,
                   to_markdown=False,
                   tablefmt = "pipe",
                   **kwargs):
    """Printing summaries of Canonical Discriminant Analysis model

    Parameters
    ----------
    self        :   an obect of class CANDISC.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """
    
    row = get_candisc(self,choice="row")
    var = get_candisc(self,choice="var")
    coef = get_candisc_coef(self,choice="absolute").round(decimals=digits)
    score_coef = get_candisc_coef(self,choice="score").round(decimals=digits)
    gmean = self.gmean_.round(decimals=digits)


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Partial Principal Components Analysis Results
    print("                     Canonical Discriminant Analysis - Results                     \n")

    print("\nSummary Information")
    summary = self.summary_information_.T
    if to_markdown:
        print(summary.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(summary)
    
    print("\nClass Level information")
    class_level_infos = self.class_level_information_
    if to_markdown:
        print(class_level_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(class_level_infos)

    # Add eigenvalues informations
    print("\nImportance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    print("\nGroup means:")
    if to_markdown:
        print(gmean.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(gmean)
    
    print("\nCoefficients of canonical discriminants:")
    if to_markdown:
        print(coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(coef)
    
    print("\nClassification functions coefficients:")
    if to_markdown:
        print(score_coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(score_coef)

    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
    row_infos = row["coord"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(row_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_infos)
    
    # Add variables informations
    print(f"\nContinues variables\n")
    var_infos = pd.DataFrame(index=self.features_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        tcorr = var["Total"].iloc[:,i]
        tcorr.name ="total."+str(i+1)
        bcorr = var["Between"].iloc[:,i]
        bcorr.name ="between."+str(i+1)
        wcorr = var["Within"].iloc[:,i]
        wcorr.name ="within."+str(i+1)
        var_infos = pd.concat([var_infos,tcorr,bcorr,wcorr],axis=1)
    var_infos = var_infos.round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)






###############################################" Fonction de reconstruction #######################################################












###########################################"" Discriminant Correspondence Analysis (CDA) ###########################################

# Row informations
def get_disca_ind(self):
    pass

# Categories informations
def get_disca_mod(self):
    pass

# Group informations
def get_disca_group(self):
    pass

# Disca extract informations
def get_disca(self,choice="ind"):
    """
    
    """
    if choice == "ind":
        return get_disca_ind(self)
    elif choice == "mod":
        return get_disca_mod(self)
    elif choice == "group":
        return get_disca_group(self)
    else:
        raise ValueError("Error : give a valid choice.")

# Summary DISCA
def summaryDISCA(self):
    pass