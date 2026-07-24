 # -*- coding: utf-8 -*-
from numpy import number, array, ones, zeros, nan,sqrt
from pandas import concat, Series, DataFrame, CategoricalDtype, get_dummies, crosstab
from scipy.stats import chi2_contingency, norm, hypergeom
from statsmodels.api import WLS
from collections import namedtuple, OrderedDict
from typing import NamedTuple

#interns fn
from ..functions.concat_empty import concat_empty
from ..functions.statistics import wmean, wstd, func_groupby

def catdes(X, 
           num_var, 
           w=None,
           proba=0.05) -> NamedTuple:
    """
    Categories description
    
    Description of the categories of one factor by categorical variables and/or continuous variables.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data with at least one categorical variable and a set of continuous variables and/or categorical variables.

    num_var : int, str
        The indexe or name of the variable to characterized. 

    row_w : 1d array-like of shape (n_samples,), default = None
        An optional individuals weights.

    proba : float, default 0.05
        The significance threshold considered to characterized the category.

    Returns
    -------
    result : catdescResult
        An object containing the result of categories description, with the following attributes:

        test_chi2 : DataFrame
            The categorical variables which characterized the factor are listed in ascending order (from the one which characterized the most the factor to the one which significantly characterized with the proba proba
        category : OrderedDict
            Description of each category of the num_var by each category of all the categorical variables.
        quanti_var : DataFrame
            The global description of the num_var variable by the continuous variables with the square correlation coefficient and the p-value of the F-test in a one-way analysis of variance (assuming the hypothesis of homoscedsticity).
        quanti : OrderedDict
            The description of each category of the num_var variable by the continuous variables.

    References
    ----------
    [1] Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall. Lebart, L., Morineau, A. and Piron, M. (1995) Statistique exploratoire multidimensionnelle, Dunod.

    See also
    --------
    :class:`~scientisttools.condes`
        Continuous variable description.

    Example
    -------
    >>> from scientisttools.dataset import wine
    >>> from scientisttools import catdes
    >>> res = catdes(wine.data, num_var=1)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # functions
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def wlsreg(y,x,w):
        data = concat((y.to_frame("y"),x.to_frame("x")),axis=1)
        wls = WLS.from_formula("y ~ C(x)",weights=w,data=data).fit()
        return Series([wls.rsquared,wls.f_pvalue],index=["R2","pvalue"])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Drop level if ndim greater than 1 and reset columns name
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.columns.nlevels > 1:
        X.columns = X.columns.droplevel()

    #set index name to None
    X.index.name = None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set labels
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if num_var is None:
        raise ValueError("num_var must be assign")
    elif not isinstance(num_var,(int,str)):
        raise TypeError(f"{type(num_var)} is not supported.")
    num_label = num_var if isinstance(num_var,str) else X.columns[num_var]

    #check if num_label is in  columns
    if not (num_label in X.columns):
        raise ValueError(f"{num_label} is not in X.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # split X into x and y
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # split X into x and y
    y, x = X[num_label], X.drop(columns=[num_label])
    # unique element in y
    uq_classe = sorted(list(y.unique()))
    # convert y to categorical data type
    y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

    # number of rows
    n_rows = x.shape[0]

    # set weights
    if w is None:
        w = ones(n_rows)/n_rows
    else:
        w = array(w)/sum(w)

    if proba is None:
        proba = 5e-2
    elif not isinstance(proba,(float,int)):
        raise TypeError(f"{type(proba)} is not supported.")
    elif proba < 0 or proba > 1:
        raise ValueError(f"the 'proba' value {proba} is not within the required range of 0 and 1.")
    
    # call informations
    call_ = OrderedDict(X=X,num_var=num_label,w=w,proba=proba)
    #convert to namedtuple
    res_ = OrderedDict(call=namedtuple("call",call_.keys())(*call_.values()))
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #split x into continuous and categorical variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #initialisation
    x_quanti, x_quali, k1, k2 = None, None, 0, 0

    # select all numerics columns
    is_quanti = x.select_dtypes(include=number)
    if not is_quanti.empty:
        x_quanti = is_quanti.to_frame() if isinstance(is_quanti,Series) else is_quanti
        k1 = x_quanti.shape[1]

    # select object or category
    is_quali = x.select_dtypes(include=["object","category"])
    if not is_quali.empty:
        x_quali = is_quali.to_frame() if isinstance(is_quali,Series) else is_quali
        k2 = x_quali.shape[1]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # statistics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if k1 > 0:
        quanti = concat((wlsreg(y=x_quanti[k],x=y,w=w).to_frame(k) for k in x_quanti.columns),axis=1).T
        quanti = quanti[quanti["pvalue"]<proba]
        if quanti.shape[0] > 0:
            res_["quanti_var"] = quanti.sort_values(by="R2",ascending=False)

        # weighted average and weighted standard deviation
        center, scale = wmean(X=x_quanti,w=w), wstd(X=x_quanti,w=w)
        # conditional weighted mean, conditional weighted standard deviation and size
        y_center, y_scale = func_groupby(X=x_quanti,by=y,func="mean",w=w), func_groupby(X=x_quanti,by=y,func="std",w=w)
        # proportion
        p_k = (get_dummies(y,dtype=int).T * w).sum(axis=1)
        # valeur-test
        vtest = (((y_center - center)/scale).T * sqrt((n_rows-1)/((1/p_k) - 1))).T
        # critical probabilities
        vtest_prob = vtest.apply(lambda x : 2*norm(0,1).sf(abs(x)),axis=0)
        
        # rearrange
        quanti = OrderedDict()
        for k  in list(p_k.index):
            df = concat((vtest.loc[k,:],y_center.loc[k,:],center,y_scale.loc[k,:],scale,vtest_prob.loc[k,:]),axis=1)
            df.columns = ["vtest","Mean in category","Overall mean","sd in category","Overall sd","pvalue"]
            df = df[df["pvalue"]<proba]
            if df.shape[0] != 0:
                quanti[str(k)] = df.sort_values(by="vtest",ascending=False)
        if len(quanti)!= 0:
            res_["quanti"] = quanti
    
    if k2 > 0:
        # disjunctive table
        dummies = concat((get_dummies(x_quali[q],prefix=q,prefix_sep='=',dtype=int) for q in x_quali.columns),axis=1)  
        n_s, p_s = dummies.sum(axis=0), dummies.mean(axis=0)  
        n_s.name, p_s.name = "n(s)","p(s)"

        # chi-squared & Value-test
        chi2_test, vtest = None, None
        for q in x_quali.columns:
            # crosstab
            tab = crosstab(x_quali[q],y)
            tab.index = [f"{q}={x}" for x in tab.index]
            # chi2 test
            chi = chi2_contingency(tab,correction=False)
            row_chi2 = DataFrame(OrderedDict(pvalue=chi.pvalue,dof=chi.dof),index=[q])
            chi2_test = concat_empty(chi2_test,row_chi2,axis=0)

            # value-test (vtest) and associated probability
            vt = DataFrame(zeros(tab.shape),index=tab.index,columns=tab.columns)
            nj, nk, n = tab.sum(axis=1), tab.sum(axis=0), tab.sum(axis=0).sum()
            pk = nk/n
            for j in tab.index:
                for k in tab.columns:
                    f_jk, f_k = tab.loc[j,k]/nj.loc[j], pk.loc[k]
                    pmf = hypergeom.pmf(k=tab.loc[j,k],n=nk[k],N=nj[j],M=n)
                    a = hypergeom.cdf(k=tab.loc[j,k]-1,n=nk[k],N=nj[j],M=n)
                    b = hypergeom.sf(k=tab.loc[j,k],n=nk[k],N=nj[j],M=n)
                    zvalue = min(2*a+pmf,2*b+pmf)
                    if zvalue <= proba:
                        t = 1 if f_jk > f_k else 0
                        val = (1-2*t)*norm(0,1).ppf(zvalue/2)
                    else:
                        val = nan
                    vt.loc[j,k] = val
            vtest = concat_empty(vtest,vt,axis=0)
            #convert to int
            chi2_test["dof"] = chi2_test["dof"].astype(int)
            #filter using probability
            chi2_test = chi2_test[chi2_test["pvalue"]<proba]
            if chi2_test.shape[0] != 0:
                res_["test_chi2"] = chi2_test.sort_values(by="pvalue")
        
        #vtest probabilities
        vtest_prob = vtest.apply(lambda x : 2*norm(0,1).sf(abs(x)),axis=0)
        #listing mode/class - class/Mod
        mod_class = 100*func_groupby(X=dummies,by=y,func="mean",w=w).T
        class_mod = (100*func_groupby(X=dummies,by=y,func="sum")/n_s).T
        category = OrderedDict()
        for i in uq_classe:
            df = concat((class_mod.loc[:,i],mod_class.loc[:,i],p_s.mul(100),vtest_prob.loc[:,i],vtest.loc[:,i]),axis=1)
            df.columns = ["Class/Mod","Mod/Class","Global","pvalue","vtest"]
            df = df.sort_values(by="vtest",ascending=False)
            df = df[df["pvalue"]<proba]
            if df.shape[0] != 0:
                category[str(i)] = df
        if len(category) > 0:
            res_["category"] = category
    return namedtuple("catdesResult",res_.keys())(*res_.values())