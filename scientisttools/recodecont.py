# -*- coding: utf-8 -*-
import pandas as pd

def recodecont(X):
    """
    Recoding of the continuous data matrix
    ----------------------------------------

    Description
    -----------
    Recoding of the continuous data matrix

    Usage
    -----
    > from scientisttools import recodecont
    > recodcont = recodecont(X)

    Parameters
    ----------
    X : pandas dataframe of continuous variables

    Return
    ------
    Z : the standardizd continuous dataframe

    means : the means of the columns of X

    std : the standard deviations of the columns of X

    Xcod : the continuous matrix X with missing values replaced with the column mean values

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    # Check if pandas dataframe
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # exclude object of category
    X = X.select_dtypes(exclude=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be numeric")
    else:
        for col in X.columns:
            X[col] = X[col].astype("float")

    # Fill NA by mean
    if X.shape[0] > 1:
        for col in X.columns:
            if X.loc[:,col].isnull().any():
                X.loc[:,col] = X.loc[:,col].fillna(X.loc[:,col].mean())
    
    if X.shape[0] == 1:
        Xcod = X
        Z = None
        means = X
        std = None
    else:
        Xcod = X
        means = X.mean(axis=0)
        std = X.std(axis=0,ddof=0)
        Z = (Xcod - means.values.reshape(1,-1))/std.values.reshape(1,-1)
    
    return {"Z" : Z, "means" : means, "std" : std, "Xcod" : Xcod}
