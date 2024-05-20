# -*- coding: utf-8 -*-
import pandas as pd

def splitmix(X):
    """
    Split mixed data
    ----------------

    Description
    -----------
    Splits a mixed data matrix in two data sets: one with the quantitative variables and one with the qualitative variables.

    Usage
    -----
    > from scientisttools import splitmix
    > X_quanti = splitmix(X)["quanti"]
    > X_quali = splitmix(X)["quali"]

    Parameters
    ----------
    X : pandas dataframe of mixed data

    Return
    ------
    quanti : pandas dataframe containing only the quantitative variables

    quali : pandas dataframe containing only the qualitative variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    # Check if pandas dataframe
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # select object of category
    quali = X.select_dtypes(include=["object","category"])
    if quali.shape[1]==0:
        X_quali = None
    else:
        for col in quali.columns:
            quali[col] = quali[col].astype("object")
        X_quali = quali
    
    # exclude object of category
    quanti = X.select_dtypes(exclude=["object","category"])
    if quanti.shape[1]==0:
        X_quanti = None
    else:
        for col in quanti.columns:
            quanti[col] = quanti[col].astype("float")
        X_quanti = quanti
    
    return {"quanti" : X_quanti, "quali" : X_quali}