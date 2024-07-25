# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
from mapply.mapply import mapply
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW

def quanti_var_desc(X,cluster,weights=None,proba=0.05,n_workers=1):
    """
    Description of quantitative variables
    -------------------------------------

    Description
    -----------
    Performs description of quantitative variables

    Usage
    -----
    ```python
    >>> quanti_var_desc(X,cluster,weights=None,proba=0.05,n_workers=1)
    ```

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_rows, n_columns)

    `cluster` : pandas series of shape(n_rows,)

    `weights` : an optional individuals weights. If None

    `proba` : the significance threshold considered to characterized the category (by default 0.05)

    `n_workers` : maximum amount of workers (processes) to spawn. See https://mapply.readthedocs.io/en/stable/_code_reference/mapply.html

    Return
    ------
    tuple containing :
        * `corr_eta2` : square correlation ratio
        * `quanti` : dictionary of description of quantitative variable by cluster
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if X is a pandas series
    if isinstance(X,pd.Series):
        X = X.to_frame()

    # Check if X is a pandas DataFrame
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Check if cluster is a pandas series
    if not isinstance(cluster,pd.Series):
        raise TypeError("'cluster' must be a pandas series.")

    # Check if all columns are numerics
    all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")

    if weights is None:
        weights = np.ones(X.shape[0])/X.shape[0]
    else:
        weights = np.array([x/np.sum(weights) for x in weights])
        
    # Dimension du tableau
    n  = X.shape[0]
    # Weights mean adn weighted standard deviation
    d = DescrStatsW(X,weights=weights,ddof=0)
    means = pd.Series(d.mean,index=X.columns,name="mean")
    std = pd.Series(d.std,index=X.columns,name="std")

    # Set cluster name
    cluster.name = "clust"

    # Concatenate with original data
    data = pd.concat([X,cluster],axis=1)

    # Moyenne conditionnelle - variance conditionnelle and effectif
    gmean, gstd,nk = data.groupby('clust').mean().T, data.groupby("clust").std(ddof=0).T, data.groupby('clust').size()

    # valeur-test
    v_test = mapply(gmean,lambda x :np.sqrt(n - 1)*(x-d.mean)/d.std, axis=0,progressbar=False,n_workers=n_workers)
    v_test = pd.concat(((v_test.loc[:,k]/np.sqrt((n - nk.loc[k])/nk.loc[k])).to_frame(k) for k in nk.index),axis=1)
    # Calcul des probabilités associées aux valeurs test
    vtest_prob = mapply(v_test,lambda x : 2*(1-sp.stats.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False,n_workers=n_workers)

    # Arrange all result
    quanti = {}
    for k  in nk.index:
        df = pd.concat((v_test.loc[:,k],gmean.loc[:,k],means,gstd.loc[:,k],std,vtest_prob.loc[:,k]),axis=1)
        df.columns = ["vtest","Mean in category","Overall mean","sd in categorie","Overall sd","pvalue"]
        df = df.sort_values(by="vtest",ascending=False).query("pvalue < @proba")
        if df.shape[0] == 0:
           df = None
        quanti[str(k)] = df
    
    # Square correlation ratio
    quanti_var = pd.DataFrame(index=X.columns,columns=['R2','pvalue']).astype("float")
    for col in X.columns:
        df = pd.concat((X[col],cluster),axis=1)
        df.columns = ["y","x"]
        res = smf.ols(formula="y~C(x)", data=df).fit()
        quanti_var.loc[col,:] = [res.rsquared,res.f_pvalue]
    # Subset using pvalue
    quanti_var = quanti_var.sort_values(by="R2",ascending=False).query('pvalue < @proba')
    if quanti_var.shape[0] == 0:
        quanti_var = None
    return quanti_var,quanti