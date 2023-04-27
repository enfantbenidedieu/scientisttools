# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scientisttools.extractfactor import get_ca

def summaryCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt="pipe",**kwargs):
    """Printing summaries of correspondence analysis model

    Parameters
    ----------
    self        :   an obect of class CA.
    digits      :   int, default=3. Number of decimal printed
    nb_element  :   int, default = 10. Number of element
    ncp         :   int, default = 3. Number of componennts
    to_markdown :   Print DataFrame in Markdown-friendly format.
    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/
    **kwargs    :   These parameters will be passed to tabulate.
    """

    row = get_ca(self,choice="row")
    col = get_ca(self,choice="col")

    ncp = min(ncp,self.n_components_)
    nb_element = min(nb_element,len(self.row_labels_))

    # Principal Components Analysis Results
    print("                     Correspondence Analysis - Results                     \n")

    # Add eigenvalues informations
    print("Importance of components")
    eig = pd.DataFrame(self.eig_,columns=self.dim_index_,
                       index=["Variance","Difference","% of var.","Cumulative of % of var."]).round(decimals=digits)
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    # Add individuals informations
    print(f"\nRows\n")
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
        print(f"\nSupplementary rows\n")
        # Save all informations
        row_sup_coord = row["row_sup"]["coord"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(row_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(row_sup_coord)

    # Add variables informations
    print(f"\nColumns\n")
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
    if self.col_sup_labels_ is not None:
        print(f"\nSupplementary columns\n")
        col_sup_coord = col["col_sup"]["coord"].iloc[:,:ncp].round(decimals=digits)
        if to_markdown:
            print(col_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs))
        else:
            print(col_sup_coord)