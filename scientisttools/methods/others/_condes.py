# -*- coding: utf-8 -*-
from numpy import array, ones, c_, number, cov, sqrt
from pandas import Series, DataFrame, concat, get_dummies
from scipy.stats import t
from statsmodels.api import WLS
from collections import namedtuple, OrderedDict
from typing import NamedTuple

def condes(X, 
           num_var, 
           w=None, 
           proba=0.05):
    """
    Continuous variables description
    
    Description of variables by continuous variable.

    Parameters
    ----------
    X : DataFrame of shape (n_rows, n_columns)
        Input data with at least one continuous variable and a set of continuous variables and/or categorical variables.

    num_var : int, str
        The indexe or name of the variable to characterized. 

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional individuals weights.

    proba : float, default 0.05
        The significance threshold considered to characterized the category.

    Returns
    -------
    result : contdesResult
        An object with the following attributes:

        quanti : DataFrame of shape (n_quanti, 2)
            The description of the num_var variable by the quantitative variables. The variables are sorted in ascending order (from the one which characterized the most to the one which significantly characterized with the proba proba).
        quali : DataFrame of shape (n_quali, 2)
            The categorical variables which characterized the continuous variables are listed in ascending order.
        category : DataFrame of shape (n_levels, 2)
            Description of the continuous variable num_var by each category of all the categorical variables.

    Example
    -------
    >>> from scientisttools.dataset import decathlon
    >>> from scientisttools import condes
    >>> res = condes(decathlon.data, num_var=2)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # functions
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def cortest(x,y,w):
        m = cov(m=c_[x,y],rowvar=False,aweights=w)
        cor, dof = m[0,1]/sqrt(m[0,0]*m[1,1]), len(x) - 2
        t_value = cor*sqrt((dof/(1- cor**2))) if (1- cor**2) != 0 else 0
        p_value = 2*t.sf(abs(t_value),dof)
        return Series([cor,p_value],index=["correlation","pvalue"])
    
    def wlsreg(y,x,w):
        data = concat((y.to_frame("y"),x.to_frame("x")),axis=1)
        wls = WLS.from_formula("y ~ C(x)",weights=w,data=data).fit()
        return Series([wls.rsquared,wls.f_pvalue],index=["R2","pvalue"])

    def wlsreg2(y,x,w):
        data = concat((y.to_frame("y"),x.to_frame("x")),axis=1)
        wls = WLS.from_formula("y ~ C(x)",weights=w,data=data).fit()
        return Series([wls.params.values[1],wls.pvalues.values[1]],index=["Estimate","pvalue"])

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
        raise ValueError("nm_var must be assign")
    elif not isinstance(num_var,(int,str)):
        raise TypeError(f"{type(num_var)} is not supported.")
    num_label = num_var if isinstance(num_var,str) else X.columns[num_var]

    #check if num_label is in X columns
    if not (num_label in X.columns):
        raise ValueError(f"{num_label} is not in X.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #split X into x and y
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #split X into x and y
    y, x = X[num_label], X.drop(columns=[num_label])

    #number of rows
    n_rows = x.shape[0]

    #set weights
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
    # split x into continuous and categorical variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # initialisation
    x_quanti, x_quali, k1, k2 = None, None, 0, 0

    #select all numerics columns
    is_quanti = x.select_dtypes(include=number)
    if not is_quanti.empty:
        x_quanti = is_quanti.to_frame() if isinstance(is_quanti,Series) else is_quanti
        k1 = x_quanti.shape[1]

    #select object or category
    is_quali = x.select_dtypes(include=["object","category"])
    if not is_quali.empty:
        x_quali = is_quali.to_frame() if isinstance(is_quali,Series) else is_quali
        k2 = x_quali.shape[1]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if k1 > 0:
        quanti = concat((cortest(x=y,y=x_quanti[k],w=w).to_frame(k) for k in x_quanti.columns),axis=1).T
        quanti = quanti[quanti["pvalue"]<proba]
        if quanti.shape[0] > 0:
            res_["quanti"] = quanti.sort_values(by="correlation",ascending=False)
    
    if k2 > 0:
        dummies = concat((get_dummies(x_quali[k],prefix=k,prefix_sep="=",dtype=int) for k in x_quali.columns),axis=1)
        quali = concat((wlsreg(y=y,x=x[k],w=w).to_frame(k) for k in x_quali.columns),axis=1).T
        category = concat((wlsreg2(y=y,x=dummies[k],w=w).to_frame(k) for k in dummies.columns),axis=1).T
        quali, category = quali[quali["pvalue"]<proba], category[category["pvalue"]<proba]
        if quali.shape[0] > 0:
            res_["quali"] = quali.sort_values(by="R2",ascending=False)
        if category.shape[0] > 0:
            res_["category"] = category
    return namedtuple("contdesResult",res_.keys())(*res_.values())