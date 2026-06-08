# -*- coding: utf-8 -*-
from sklearn.impute import SimpleImputer
from pandas import DataFrame, Series

def func_fillna(
        X, method = "mean"
):
    """
    Fill NA/NAN

    Impute missing values with average, median or mode.

    Replace missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.

    Parameters
    ----------
    X : DataFrame of shape (n_rows, n_columns)
        Input data.

    method : str, default = "mean"
        The imputation method:

        - If 'mean', then replace missing values using the mean along each column. Can only be used with numeric data.
        - If 'median', then replace missing values using the median along each column. Can only be used with numeric data.
        - If 'most_frequent', then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

    Returns
    -------
    X : array-like of shape (n_rows, n_columns)
        Ouput data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class Series or DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X, (Series,DataFrame)):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.Series or pd.DataFrame")
    
    if method not in ("mean","median","most_frequent"):
        raise ValueError("Not convenient method.")
    
    clf = SimpleImputer(strategy=method)
    return  DataFrame(clf.fit_transform(X),index=X.index,columns=X.columns)