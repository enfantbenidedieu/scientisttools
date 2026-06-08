# -*- coding: utf-8 -*-
from numpy import ndarray,array,ones,sqrt, abs, c_
from pandas import DataFrame, Series, crosstab, concat
from scipy.stats import t, chi2_contingency, contingency
from scipy.stats import chi2, contingency, chi2_contingency
from statsmodels.api import WLS
from collections import OrderedDict, namedtuple
from typing import NamedTuple

#interns functions
from .statistics import wcorr
from .utils import check_is_dataframe, is_numeric_dtype, is_object_or_category_dtype, check_is_all_object_or_category_dtype
from .concat_empty import concat_empty

def wpearsonr(
        x,y,w=None
) -> NamedTuple:
    """
    Test for weighted pearson correlation coefficient.

    performns weighted pearson correlation coefficient, degree of freedom and associated p-value for testing non-correlation.
    
    Parameters
    ----------
    x : 1d array-like of shape (n_samples,)
        ``x`` values.

    y : 1d array-like of shape (n_samples,)
        ``y`` values.

    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in ``x``.

    Returns
    -------
    wpearsonr : wpearsonrResult
        An object with the following attributes
    
        statistic : float
            The weighted Pearson product-moment correlation coefficient.
        
        dof : int
            The degre of freedom.  
        
        pvalue : float 
            The critical p-value associated.
    
    Examples
    --------
    >>> from numpy import arange, array
    >>> from scientisttools import wpearsonr
    >>> x, y, wt = arange(1,11), array([1,2,3,8,7,6,5,8,9,10]), array([0,0,0,1,1,1,1,1,0,0])
    >>> res = wpearsonr(x=x,y=y,weights=wt)
    >>> res
    ...wpearsonrResult(statistic=-0.24253562503633294, dof=8, pvalue=0.49957589436325933)
    """
    statistic = wcorr(DataFrame(c_[x,y]),w=w).iloc[0,1]
    t_stat, dof = statistic*sqrt(((len(x)-2)/(1- statistic**2))), len(x) - 2
    pvalue = 2*t.sf(abs(t_stat),dof)
    #convert to ordered dictionary
    res_ = OrderedDict(statistic=statistic,dof=dof,pvalue=pvalue)
    return namedtuple("wpearsonrResult",res_.keys())(*res_.values())

def eta2test(
        x,y,w=None
) -> NamedTuple:
    """
    Test of Eta squared

    performns weighted eta-squared coefficient, degree of freedom and associated p-value for testing non-correlation.

    Parameters
    ----------
    x : 1d array-like of shape (n_samples,)
        ``x`` values. ``x`` contains categories values.

    y : 1d array-like of shape (n_samples,)
        ``y`` values. ``y`` contains numerics values.

    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in ``x``.

    Returns
    -------
    eta2test : eta2testResult
        An object with the following attributes
    
        statistic : float
            The eta squared coefficient.
        
        df_num : int
            The numerator degre of freedom.  

        df_denom : int
            The denominator degre of freedom.  
        
        pvalue : float 
            The critical p-value associated.

    References
    ----------
    F. Bertrand, M. Maumy-Bertrand, Initiation à la Statistique avec R, Dunod, 4ème édition, 2023.
    """
    n_rows = len(x)
    #set weights
    if w is None:
        w = ones(n_rows)/n_rows
    elif not isinstance(w,(list,tuple,ndarray,Series)):
        raise TypeError("'w' must be a 1d array-like of rows weights.")
    elif len(w) != n_rows:
        raise ValueError(f"'w' must be a 1d array-like of shape ({n_rows},).")
    else:
        w = array(w)/sum(w)
    #weighted least squared
    wls = WLS.from_formula("y ~ C(x)",weights=w,data=DataFrame(c_[y,x],columns=["y","x"])).fit()
    #convert to ordered dictionary
    res_ = OrderedDict(statistic=float(wls.rsquared),df_num=int(wls.df_model),df_denom=int(wls.df_resid),pvalue=float(wls.f_pvalue))
    return namedtuple("eta2testResult",res_.keys())(*res_.values())

def association(
        X
):
    """
    Association between nominal variables

    Compute the degree of association between two nominales variables

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        ``X`` contains nominal variables.

    Returns
    -------
    result : association
        An object with the following attributes:

        association : DataFrame of shape (n_columns*(n_columns - 1)/2, 5)
            The degree of association between two nominal variables ("cramer", "tschuprow", "pearson").

        chi2 : DataFrame of shape (n_columns*(n_columns - 1)/2, 4)
            The pearson's chi-squared test.
        
        gtest : DataFrame of shape (n_columns*(n_columns - 1)/2, 4)
            The log-likelihood ratio (i.e the "G-test").
        
    References
    ----------
    [1] "Contingency table", https://en.wikipedia.org/wiki/Contingency_table
    
    [2] "Pearson's chi-squared test", https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    
    [3] "Tschuprow's T", https://en.wikipedia.org/wiki/Tschuprow's_T
    
    [4] "Cramer's V", https://en.wikipedia.org/wiki/Cramer's_V
    
    [5] "Nominal Association: Phi and Cramer's V", http://www.people.vcu.edu/~pdattalo/702SuppRead/MeasAssoc/NominalAssoc.html
    
    [6] Gingrich, Paul, `Association Between Variables <http://uregina.ca/~gingrich/ch11a.pdf>`_.

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import association
    >>> poison = load_poison()
    >>> res = association(poison.iloc[:,2:],0.05)
    >>> res.association #association
    >>> res.chi2 #pearson's chi-squared test
    >>> res.gtest #log-likelihood ratio (G-test)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of pandas DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if all columns in X are categorics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_all_object_or_category_dtype(X)

    #Chi2 statistic test
    chi2_test, association, g_test = None, None, None
    idx = 0
    for i in range(X.shape[1]-1):
        for j in range(i+1,X.shape[1]):
            #contingency table
            tab = crosstab(X.iloc[:,i],X.iloc[:,j])
            #pearson chi-squared test
            statistic, pvalue, dof, _ = chi2_contingency(tab,lambda_=None,correction=False)
            row_chi2 = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],statistic=statistic,dof=dof,pvalue=pvalue),index=[idx])
            chi2_test = concat_empty(chi2_test,row_chi2,axis=0,ignore_index=True)
            #log-likelihood test (G-test)
            g_stat, g_pvalue, g_dof = chi2_contingency(tab, lambda_="log-likelihood")[:3]
            row_gtest = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],statistic=g_stat,dof=g_dof,pvalue=g_pvalue),index=[idx])
            g_test = concat_empty(g_test,row_gtest,axis=0,ignore_index=True)
            #others association tests (cramer, tschuprow, pearson)
            asso_test = [contingency.association(tab,method=k,correction=False) for k in ["cramer","tschuprow","pearson"]]
            row_asso = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],cramer=asso_test[0],tschuprow=asso_test[1],pearson=asso_test[2]),index=[idx])
            association = concat_empty(association,row_asso,axis=0,ignore_index=True)
            idx += 1
    #transform to int
    chi2_test["dof"], g_test["dof"] = chi2_test["dof"].astype(int), g_test["dof"].astype(int)
    #convert to namedtuple
    return namedtuple("association",["association","chi2","gtest"])(association,chi2_test,g_test)

def wcorrtest(
        X,w=None
) -> DataFrame:
    """
    Weighted correlation test

    Performns weighted correlation test

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in ``X``.

    Returns
    -------
    corrtest : DataFrame (n_columns*(n_columns - 1), 5)
        The weighted correlation tests.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)

    #set number of rows and number of columns
    n_rows, n_cols = X.shape

    #set weights
    if w is None:
        w = ones(n_rows)/n_rows
    elif not isinstance(w,(list,tuple,ndarray,Series)):
        raise TypeError("'w' must be a 1d array-like of rows weights.")
    elif len(w) != n_rows:
        raise ValueError(f"'w' must be a 1d array-like of shape ({n_rows},).")
    else:
        w = array(w)/sum(w)
        
    corr_test = DataFrame(columns=["variable1","variable2","test","statistic","pvalue"]).astype("float")
    idx = 0
    for i in range(n_cols-1):
        for j in range(i+1,n_cols):
            if is_numeric_dtype(X.iloc[:,i]) and is_numeric_dtype(X.iloc[:,j]):
                res, test = wpearsonr(x=X.iloc[:,i].values,y=X.iloc[:,j].values,w=w), "Pearson correlation"
            elif is_object_or_category_dtype(X.iloc[:,i]) and is_object_or_category_dtype(X.iloc[:,j]):
                res, test = chi2_contingency(crosstab(X.iloc[:,i],X.iloc[:,j]),lambda_=None,correction=False), "Pearson chi-squared"
            elif is_numeric_dtype(X.iloc[:,i]) and is_object_or_category_dtype(X.iloc[:,j]):
                res, test = eta2test(x=X.iloc[:,j].values,y=X.iloc[:,i].values,w=w), "Eta-squared ratio"
            elif is_object_or_category_dtype(X.iloc[:,i]) and is_numeric_dtype(X.iloc[:,j]):
                res, test = eta2test(x=X.iloc[:,i].values,y=X.iloc[:,j].values,w=w), "Eta-squared ratio"
            else:
                raise TypeError("Variables should be either quantitative or qualitative")
            row_corr = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],test=test,statistic=res.statistic,pvalue=res.pvalue),index=[idx])
            corr_test = concat((corr_test,row_corr),axis=0,ignore_index=True)
            idx += 1
    return corr_test