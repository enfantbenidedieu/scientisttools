# -*- coding: utf-8 -*-
from numpy import abs, sqrt
from pandas import DataFrame, concat, get_dummies, crosstab
from scipy import stats
from collections import OrderedDict

def quali_var_desc(X,cluster,proba=0.05):
    """
    Description of qualitative variables
    ------------------------------------

    Description
    -----------
    Performns description of qualitative variables

    Usage
    -----
    ```python
    >>> quali_var_desc(X,cluster,proba=0.05,n_workers=1)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_rows, n_columns)

    `cluster`: a pandas Series of shape(n_rows,)

    `proba`: a numeric indicating the significance threshold considered to characterized the category (by default 0.05).

    Returns
    -------
    tuple containing:
        * `chi2_test`: chi-square statistic test
        * `category`: dictionary of description of qualitative variable by cluster
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #disjunctive table
    dummies = concat((get_dummies(X[q],prefix=q,prefix_sep='=',dtype=int) for q in X.columns),axis=1)  
    n_s, p_s = dummies.sum(axis=0), dummies.mean(axis=0)  
    n_s.name, p_s.name = "n(s)","p(s)"

    #chi-squared & Value-test
    chi2_test, v_test = DataFrame(columns=["statistic","dof","pvalue"]).astype("float"), DataFrame().astype("float")
    for q in X.columns:
        #crosstab
        tab = crosstab(X[q],cluster)
        tab.index = [q+"="+x for x in tab.index]

        #chi2 test
        chi = stats.chi2_contingency(tab,correction=False)
        row_chi2 = DataFrame(OrderedDict(statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[q])
        chi2_test = concat((chi2_test,row_chi2),axis=0)

        #value-test (vtest)
        nj, nk, n = tab.sum(axis=1), tab.sum(axis=0), tab.sum().sum()
        for j in tab.index:
            for k in tab.columns:
                pi = (nj.loc[j]*nk.loc[k])/n
                num, den = tab.loc[j,k] - pi, ((n-nk.loc[k])/(n-1))*(1-nj.loc[j]/n)*pi
                tab.loc[j,k] = num/sqrt(den)
        v_test = concat((v_test,tab),axis=0) 
    #convert to int
    chi2_test["dof"] = chi2_test["dof"].astype(int)
    #filter using probability
    chi2_test = chi2_test.sort_values(by="pvalue").query("pvalue < @proba")

    #vtest probabilities
    def prob(x):
        return 2*(1-stats.norm(0,1).cdf(abs(x)))
    vtest_prob = v_test.transform(prob)

    #listing MOD/CLASS
    dummies_classe = concat([dummies,cluster],axis=1)
    mod_class = dummies_classe.groupby("clust").mean().T.mul(100)

    #class/Mod
    class_mod = dummies_classe.groupby("clust").sum().T
    class_mod = class_mod.div(n_s.values,axis=0).mul(100)

    var_category = OrderedDict()
    for i in cluster.unique():
        df = concat((class_mod.loc[:,i],mod_class.loc[:,i],p_s.mul(100),vtest_prob.loc[:,i],v_test.loc[:,i]),axis=1)
        df.columns = ["Class/Mod","Mod/Class","Global","pvalue","vtest"]
        df = df.sort_values(by="vtest",ascending=False).query("pvalue < @proba")
        if df.shape[0] == 0:
            df = None
        var_category[str(i)] = df
    return chi2_test, var_category