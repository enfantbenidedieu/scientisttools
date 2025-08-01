# -*- coding: utf-8 -*-
from pandas import api, DataFrame, concat

def summarize(X:DataFrame) -> DataFrame:
    """
    Summarize DataFrame
    -------------------

    Usage
    -----
    ```python
    >>> summarize(X)
    ```

    Parameters
    ----------
    `X`: pandas dataframe with `n` rows and `p` columns containing either numeric or categorical columns

    Returns
    -------
    a pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """

    #If all variables are numeric
    if all(api.types.is_numeric_dtype(X[k]) for k in X.columns):
        res = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        res["count"] = res["count"].astype("int")
        return res
    #If all columns are categorical
    elif all(api.types.is_string_dtype(X[q]) for q in X.columns):
        def freq_prop(q):
            eff = X[q].value_counts().to_frame("count").reset_index().rename(columns={q : "categorie"}).assign(proportion = lambda x : x["count"]/x["count"].sum())
            eff.insert(0,"variable",q)
            return eff
        return concat(map(lambda q : freq_prop(q), X.columns),axis=0,ignore_index=True)
    else:
        TypeError("All columns must be either numeric or categorical.")

# sum of columns by gourp
def sum_col_by(X:DataFrame,X_quali:DataFrame) -> DataFrame:
    """
    Sum of columns by group 
    --------------------------

    Usage
    -----
    ```python
    >>> sum_col_by(X,X_quali)
    ```

    Parameters
    ----------
    `X`: pandas dataframe of quantitative variable
    
    `X_quali`: pandas dataframe of qualitative variable

    Return
    ------
    a pandas dataframe 

    Author(s)
    ---------  
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com  
    """
    #sum by 
    def sum_by(q):
        data = concat((X,X_quali[q]),axis=1).groupby(by=q,as_index=True).sum()
        data.index.name = None
        return data
    return concat((map(lambda q : sum_by(q=q), X_quali.columns)),axis=0)