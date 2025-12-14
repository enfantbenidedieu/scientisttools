# -*- coding: utf-8 -*-
from numpy import ones, ndarray, array
from pandas import DataFrame, Series, concat, crosstab
from pandas.api.types import is_numeric_dtype, is_string_dtype
from collections import OrderedDict
from scipy.stats import chi2_contingency

#intern functions
from .wpearsonr import wpearsonr
from .eta2 import eta2

def corrmatrix(X:DataFrame,weights=None):

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/1darray/Series of individuals weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/1darray/Series with length {X.shape[0]}.")
    else:
        weights = array(list(map(lambda x : x/sum(weights),weights)))

    corr_test = DataFrame(columns=["variable1","variable2","test","statistic","pvalue"]).astype("float")
    idx = 0
    for i in range(X.shape[1]-1):
        for j in range(i+1,X.shape[1]):
            if is_numeric_dtype(X.iloc[:,i]) and is_numeric_dtype(X.iloc[:,j]):
                corr = wpearsonr(x=X.iloc[:,i].values,y=X.iloc[:,j].values,weights=weights)
                statistic, pvalue, test = corr.statistic, corr.pvalue, "Pearson correlation"
            elif is_numeric_dtype(X.iloc[:,i]) and is_string_dtype(X.iloc[:,j]):
                corr = eta2(categories=X.iloc[:,j].values,value=X.iloc[:,i].values)
                statistic, pvalue, test = corr["Eta2"], corr["pvalue"], "Squared correlation ratio"
            elif is_string_dtype(X.iloc[:,i]) and is_numeric_dtype(X.iloc[:,j]):
                corr = eta2(categories=X.iloc[:,i].values,value=X.iloc[:,j].values)
                statistic, pvalue, test = corr["Eta2"], corr["pvalue"], "Squared correlation ratio"
            elif is_string_dtype(X.iloc[:,i]) and is_string_dtype(X.iloc[:,j]):
                chi2 = chi2_contingency(crosstab(X.iloc[:,i],X.iloc[:,j]),lambda_=None,correction=False)
                statistic, pvalue, test = chi2.statistic, chi2.pvalue, "Pearson chi-squared"
            else:
                Exception("Variables should be either quantitative or qualitative")
            row_corr = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],test=test,statistic=statistic,pvalue=pvalue),index=[idx])
            corr_test = concat((corr_test,row_corr),axis=0,ignore_index=True)
            idx += 1
    return corr_test