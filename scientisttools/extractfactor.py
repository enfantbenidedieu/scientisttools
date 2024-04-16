# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scientisttools.eta2 import eta2
import scipy.stats as st





    

        
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