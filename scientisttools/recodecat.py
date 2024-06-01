# -*- coding: utf-8 -*-
import pandas as pd
from .revaluate_cat_variable import revaluate_cat_variable

def recodecat(X,dummy_na=False):
    """
    Recoding of the categorical variables
    -------------------------------------

    Description
    -----------
    Recoding of the categorical variables

    Usage
    -----
    > from scientisttools import recodecat
    > rec = recodecat(X)

    Parameters
    ----------
    X : pandas dataframe of catgeorical variables

    dummy_na : Add a column to indicate NaNs, if False NaNs are ignored.

    Return
    ------
    TDC : Dummy coded data

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
    X = X.select_dtypes(include=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be either object or category")
    else:
        for col in X.columns:
            X[col] = X[col].astype("object")

    # Revaluate
    X = revaluate_cat_variable(X)
    dummies = pd.concat((pd.get_dummies(X[col],dtype=int,dummy_na=dummy_na) for col in X.columns),axis=1)
    return {"X" : X, "dummies" : dummies}