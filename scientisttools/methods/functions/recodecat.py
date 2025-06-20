# -*- coding: utf-8 -*-
from pandas import DataFrame, Categorical, concat, get_dummies
from .revaluate_cat_variable import revaluate_cat_variable
from collections import namedtuple

def recodecat(X,dummy_na=False):
    """
    Recoding of the categorical variables
    -------------------------------------

    Description
    -----------
    Recoding of the categorical variables

    Usage
    -----
    ```python
    >>> from scientisttools import recodecat
    >>> rec = recodecat(X)
    ```

    Parameters
    ----------
    `X` : pandas dataframe of categorical variables

    `dummy_na` : Add a column to indicate NaNs, if False NaNs are ignored.

    Return
    ------
    namedtuple of dataframe containing:
    `X` : pandas dataframe of categorical data.

    `dummies` : pandas dataframe of disjunctive table.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if pandas dataframe
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # select object of category
    X = X.select_dtypes(include=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be either object or category")
    else:
        for j in X.columns:
            X[j] = Categorical(X[j],categories=sorted(X[j].dropna().unique().tolist()),ordered=True)

    # Revaluate
    X = revaluate_cat_variable(X)
    dummies = concat((get_dummies(X[j],dtype=int,dummy_na=dummy_na) for j in X.columns),axis=1)
    return namedtuple("recodecat",["X","dummies"])(X,dummies)