# -*- coding: utf-8 -*-
from sklearn.impute import SimpleImputer
from pandas import DataFrame, Series

def func_fillna(X, method = "mean"):
    """
    Fill NA/NAN

    Impute missing values with average, median or mode.

    Replace missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.

    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Input data.

    method : {"mean","median","most_frequent"}, default = "mean"
        The imputation method:

        * 'mean', then replace missing values using the mean along each column. Can only be used with numeric data.
        * 'median', then replace missing values using the median along each column. Can only be used with numeric data.
        * 'most_frequent', then replace missing using the most frequent value along each column. 
            Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

    Returns
    -------
    Y : array-like of shape (n_samples, n_columns)
        Ouput data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class Series or DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X, (Series,DataFrame)):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.Series or pd.DataFrame")
    
    if method not in ("mean","median","most_frequent"):
        raise ValueError("Not convenient method.")
    
    if isinstance(X,Series):
        colnames = X.name
    else:
        colnames = X.columns
    
    clf = SimpleImputer(strategy=method)
    Y = DataFrame(clf.fit_transform(X),index=X.index,columns=colnames)
    return  Y