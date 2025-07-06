# -*- coding: utf-8 -*-
from pandas import concat, DataFrame

def sum_table(X:DataFrame,X_quali:DataFrame,q:str) -> DataFrame:
    """
    Summarize columns by group 
    --------------------------

    Usage
    -----
    ```python
    >>> sum_table(X,X_quali,q)
    ```

    Parameters
    ----------
    `X`: pandas dataframe of quantitative variable
    
    `X_quali`: pandas dataframe of qualitative variable

    `q`: a string specifying the name of the qualitative variable

    Return
    ------
    a pandas dataframe with summarize data

    Author(s)
    ---------  
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com  
    """
    data = concat((X,X_quali[q]),axis=1).groupby(by=q,as_index=True).sum()
    data.index.name = None
    return data