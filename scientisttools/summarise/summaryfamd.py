# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scientisttools.extractfactor import get_famd

def summaryFAMD(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """Printing summaries of factor analysis of mixed data model

    Parameters
    ----------
    self        :   an obect of class FAMD.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_famd(self,choice="ind")
    col = get_famd(self,choice="col")
    mod = get_famd(self,choice="mod")
    var = get_famd(self,choice="var")


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Factor Analysis of Mixed - Results
    print("                    Factor Analysis of Mixed Data - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first)\n")
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
    if self.row_sup_labels_ is not None:
        nb_elt = min(nb_element,len(self.row_sup_labels_))
        print(f"\nSupplementary Individuals\n")
        # Save all informations
        row_sup_infos = pd.DataFrame(index=self.row_sup_labels_).astype("float")
        row_sup = row["ind_sup"]
        for i in np.arange(0,ncp,1):
            row_sup_coord = row_sup["coord"].iloc[:,i]
            row_sup_cos2 = row_sup["cos2"].iloc[:,i]
            row_sup_cos2.name = "cos2"
            row_sup_infos = pd.concat([row_sup_infos,row_sup_coord,row_sup_cos2],axis=1)
        row_sup_infos = row_sup_infos.iloc[:nb_elt,:].round(decimals=digits)

        if to_markdown:
            print(row_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_infos)
    
    # Add variables informations
    print(f"\nContinuous variables\n")
    col_infos = pd.DataFrame(index=self.col_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_cos2 = col["cos2"].iloc[:,i]
        col_cos2.name = "cos2"
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr,col_cos2],axis=1)
    col_infos = col_infos.round(decimals=digits)

    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup_labels_ is not None:
        print(f"\nSupplementary continuous variables\n")
        col_sup_infos = pd.DataFrame(index=self.col_sup_labels_).astype("float")
        col_sup = col["quanti_sup"]
        for i in np.arange(0,ncp,1):
            col_sup_coord = col_sup["coord"].iloc[:,i]
            col_sup_cos2 = col_sup["cos2"].iloc[:,i]
            col_sup_cos2.name = "cos2"
            col_sup_infos =pd.concat([col_sup_infos,col_sup_coord,col_sup_cos2],axis=1)
        col_sup_infos = col_sup_infos.round(decimals=digits)

        if to_markdown:
            print(col_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_infos)

    # Add variables informations
    print(f"\nCategories\n")
    mod_infos = mod["infos"]
    for i in np.arange(0,ncp,1):
        mod_coord = mod["coord"].iloc[:,i]
        mod_cos2 = mod["cos2"].iloc[:,i]
        mod_cos2.name = "cos2"
        mod_ctr = mod["contrib"].iloc[:,i]
        mod_ctr.name = "ctr"
        mod_vtest = mod["vtest"].iloc[:,i]
        mod_vtest.name = "v.test"
        mod_infos = pd.concat([mod_infos,mod_coord,mod_ctr,mod_cos2, mod_vtest],axis=1)
    mod_infos = mod_infos.round(decimals=digits)

    if to_markdown:
        print(mod_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(mod_infos)
    
    # Add variables
    print("\nCategorical variables\n")
    var_infos = pd.DataFrame(index=self.quali_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        var_eta2 = var["eta2"].iloc[:,i]
        var_eta2.name = "eta2."+str(i+1)
        var_cos2 = var["cos2"].iloc[:,i]
        var_cos2.name = "cos2." +str(i+1)
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr."+str(i+1)
        var_infos = pd.concat([var_infos,var_eta2,var_ctr,var_cos2],axis=1)
    var_infos = var_infos.round(decimals=digits)

    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup_labels_ is not None:
        print("\nSupplementary categories\n")
        mod_sup = mod["quali_sup"]
        mod_sup_infos = np.sqrt(mod_sup["dist"])
        for i in np.arange(0,ncp,1):
            mod_sup_coord = mod_sup["coord"].iloc[:,i]
            mod_sup_cos2 = mod_sup["cos2"].iloc[:,i]
            mod_sup_cos2.name = "cos2"
            mod_sup_vtest = mod_sup["vtest"].iloc[:,i]
            mod_sup_vtest.name = "v.test"
            mod_sup_infos = pd.concat([mod_sup_infos,mod_sup_coord,mod_sup_cos2,mod_sup_vtest],axis=1)
        mod_sup_infos = mod_sup_infos.round(decimals=digits)

        if to_markdown:
            print(mod_sup_infos.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(mod_sup_infos)
        
        # Add supplementary Categorical 
        print("\nSupplementary categorical variables\n")
        corr_ratio = var["quali_sup"]["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(corr_ratio.to_markdown(tablefmt=tablefmt))
        else:
            print(corr_ratio)



