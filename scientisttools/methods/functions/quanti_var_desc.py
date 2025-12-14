# -*- coding: utf-8 -*-
from numpy import array, ones, average, sqrt, cov
from pandas import DataFrame, Series, api, concat
from scipy import stats
from collections import OrderedDict
import statsmodels.formula.api as smf

#intern function
from .conditional_wmean import conditional_wmean
from .conditional_wstd import conditional_wstd

def quanti_var_desc(X,cluster,weights=None,proba=0.05):
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
    `X`: a pandas DataFrame of shape (n_rows, n_columns)

    `cluster`: a pandas series of shape(n_rows,)

    `weights`: an optional individuals weights. If None

    `proba`: a numeric indicating the significance threshold considered to characterized the category (by default 0.05)

    Returns
    ------
    tuple containing :
    
    `corr_eta2`: square correlation ratio
    
    `quanti`: dictionary of description of quantitative variable by cluster
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is a pandas Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if cluster is a pandas series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(cluster,Series):
        raise TypeError("'cluster' must be a pandas series.")

    #check if all columns are numerics
    all_num = all(api.types.is_numeric_dtype(X[c]) for c in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")

    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    else:
        weights = array([x/sum(weights) for x in weights])
        
    #dimension du tableau
    n_rows  = X.shape[0]
    #weighted average and weighted standard deviation
    center, scale = Series(average(X,axis=0,weights=weights),index=X.columns,name="center"), Series([sqrt(cov(X.loc[:,k],aweights=weights)) for k in X.columns],index=X.columns,name="std")

    #conditional weighted mean, conditional weighted standard deviation and size
    gcenter, gscale, n_k = conditional_wmean(X=X,Y=cluster,weights=weights), conditional_wstd(X=X,Y=cluster,weights=weights), cluster.value_counts()
    
    #valeur-test
    v_test = gcenter.sub(center,axis=0).div(scale,axis=0).mul(sqrt(n_rows - 1)).div(sqrt((n_rows - n_k)/n_k),axis=1)
    #calcul des probabilités associées aux valeurs test
    def prob(x):
        return 2*(1-stats.norm(0,1).cdf(abs(x)))
    vtest_prob = v_test.transform(prob)

    #reorder all result
    quanti = OrderedDict()
    for k  in n_k.index:
        df = concat((v_test.T.loc[:,k],gcenter.T.loc[:,k],center,gscale.T.loc[:,k],scale,vtest_prob.T.loc[:,k]),axis=1)
        df.columns = ["vtest","Mean in category","Overall mean","sd in categorie","Overall sd","pvalue"]
        df = df.sort_values(by="vtest",ascending=False).query("pvalue < @proba")
        if df.shape[0] == 0:
           df = None
        quanti[str(k)] = df
    
    #squared correlation ratio
    quanti_var = DataFrame(index=X.columns,).astype("float")
    def sqeta(k):
        df = concat((X[k],cluster),axis=1)
        df.columns = ["y","x"]
        res = smf.ols(formula="y~C(x)", data=df).fit()
        return DataFrame(OrderedDict(R2=res.rsquared,pvalue=res.f_pvalue),index=[k])
    #concatenate and subset using pvalue
    quanti_var = concat((sqeta(k) for k in X.columns),axis=0).sort_values(by="R2",ascending=False).query('pvalue < @proba')
    if quanti_var.shape[0] == 0:
        quanti_var = None
    return quanti_var, quanti