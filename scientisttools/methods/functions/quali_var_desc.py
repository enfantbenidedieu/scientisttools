# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
from mapply.mapply import mapply

def quali_var_desc(X,cluster,proba=0.05,n_workers=1):
    """
    Description of qualitatuve variables
    ------------------------------------

    Description
    -----------
    Performans description of qualitative variables

    Usage
    -----
    ```python
    >>> quali_var_desc(X,cluster,proba=0.05,n_workers=1)
    ```

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_rows, n_columns)

    `cluster` : pandas series of shape(n_rows,)

    `proba` : the significance threshold considered to characterized the category (by default 0.05).

    `n_workers` : maximum amount of workers (processes) to spawn. See https://mapply.readthedocs.io/en/stable/_code_reference/mapply.html

    Returns
    -------
    tuple containing:
        * `chi2_test`: chi-square statistic test
        * `category`: dictionary of description of qualitative variable by cluster
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    ######## Tableau Disjonctif complex
    dummies = pd.concat((pd.get_dummies(X[col],prefix=col,prefix_sep='=') for col in X.columns),axis=1)            
    dummies_stats = dummies.agg(func=[np.sum,np.mean]).T
    dummies_stats.columns = ["n(s)","p(s)"]

    # chi2 & Valeur - test
    chi2_test = pd.DataFrame(columns=["statistic","dof","pvalue"]).astype("float")
    v_test = pd.DataFrame().astype("float")
    for col in X.columns:
        # Crosstab
        tab = pd.crosstab(X[col],cluster)
        tab.index = [col+"="+x for x in tab.index]

        # Chi2 test
        chi = sp.stats.chi2_contingency(tab,correction=False)
        row_chi2 = pd.DataFrame({"statistic" : chi.statistic,"dof" : chi.dof,"pvalue" : chi.pvalue},index=[col])
        chi2_test = pd.concat((chi2_test,row_chi2),axis=0)

        # Valeur - test
        nj, nk, n = tab.sum(axis=1), tab.sum(axis=0), tab.sum().sum()
        for j in tab.index.tolist():
            for k in tab.columns.tolist():
                pi = (nj.loc[j]*nk.loc[k])/n
                num, den = tab.loc[j,k] - pi, ((n-nk.loc[k])/(n-1))*(1-nj.loc[j]/n)*pi
                tab.loc[j,k] = num/np.sqrt(den)
        v_test = pd.concat((v_test,tab),axis=0) 
    # Convert to int
    chi2_test["dof"] = chi2_test["dof"].astype(int)
    # Filter using probability
    chi2_test = chi2_test.sort_values(by="pvalue").query("pvalue < @proba")

    # vtest probabilities
    vtest_prob = mapply(v_test,lambda x : 2*(1-sp.stats.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False,n_workers=n_workers)

    # Listing MOD/CLASS
    dummies_classe = pd.concat([dummies,cluster],axis=1)
    mod_class = dummies_classe.groupby("clust").mean().T.mul(100)

    # class/Mod
    class_mod = dummies_classe.groupby("clust").sum().T
    class_mod = class_mod.div(dummies_stats["n(s)"].values,axis="index").mul(100)

    var_category = {}
    for i in np.unique(cluster):
        df = pd.concat((class_mod.loc[:,i],mod_class.loc[:,i],dummies_stats["p(s)"].mul(100),vtest_prob.loc[:,i],v_test.loc[:,i]),axis=1)
        df.columns = ["Class/Mod","Mod/Class","Global","pvalue","vtest"]
        df = df.sort_values(by="vtest",ascending=False).query("pvalue < @proba")
        if df.shape[0] == 0:
            df = None
        var_category[str(i)] = df
    return chi2_test,var_category