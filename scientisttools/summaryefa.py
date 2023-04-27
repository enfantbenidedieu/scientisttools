# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scientisttools.extractfactor import get_efa

def summaryEFA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """Printing summaries of exploratory factor analysis model

    Parameters
    ----------
    self        :   an obect of class EFA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_efa(self,choice="row")
    col = get_efa(self,choice="var")


    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_),len(self.col_labels_))

    # Exploratory Factor Analysis Results
    print("                     Exploratory Factor Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nIndividuals (the {nb_element} first) - Coordonnées factorielles des {nb_element} premiers ind.\n")
    row_coord = row["coord"].iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        print(row_coord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(row_coord)

    # Add supplementary individuals
    if self.row_sup_labels_ is not None:
        nb_elt = min(nb_element,len(self.row_sup_labels_))
        print(f"\nSupplementary Individuals - Coordonnées individus. \nSupplémentaires avec les COS2.\n")
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
    print(f"\nVariables - Loadings des variables avec CTR.\n")
    col_infos = pd.DataFrame(index=self.col_labels_).astype("float")
    for i in np.arange(0,ncp,1):
        col_coord = col["coord"].iloc[:,i]
        col_ctr = col["contrib"].iloc[:,i]
        col_ctr.name = "ctr"
        col_infos = pd.concat([col_infos,col_coord,col_ctr],axis=1)
    col_infos = col_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(col_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(col_infos)
    
    # Add supplementary continuous variables informations
    if self.quanti_sup_labels_ is not None:
        print(f"\nSupplementary continuous variable - Variable illustrative quantitative. \nAvec corrélation et COS2 (carré de la corrélation).\n")
        col_sup_infos = pd.DataFrame(index=self.quanti_sup_labels_).astype("float")
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
    
    # Add Supplementary categories – Variable illustrative qualitative
    if self.quali_sup_labels_ is not None:
        print("\nSupplementary categories - Variable illustrative qualitative.\nAvec dist. A l'origine et moyennes cond. des modalités, COS2 et valeur-test.\n")
        mod_sup = col["quali_sup"]
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
        
        # Add supplementary qualitatives - correlation ration
        print("\nSupplementatry qualitative variable - Correlation ratio\n")
        corr_ratio = mod_sup["eta2"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(corr_ratio.to_markdown(tablefmt=tablefmt))
        else:
            print(corr_ratio)



