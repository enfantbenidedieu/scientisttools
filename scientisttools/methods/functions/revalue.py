# -*- coding: utf-8 -*-
from pandas import CategoricalDtype, DataFrame, Series

def revalue(
        X,
) -> DataFrame:
    """
    Revalue Categorical Variables

    Check if two categoricals variables have same categories and replace with new values.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns) or Series of shape (n_samples,)
        X contains categoricals variables.

    Returns
    -------
    Y : DataFrame of shape (n_samples, n_columns)
        Revaluated data
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.DataFrame")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if all columns are categorics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not all(X[k].dtype in ("object","category") for k in X.columns):
        raise TypeError("All columns in X must be categorics.")
    
    #find columns with at least one element in common
    def find_intersection(X):
        i, res = 0, {}
        for k in range(X.shape[1]-1):
            for l in range(k+1,X.shape[1]):
                intersect = list(set(X.iloc[:,k].dropna().unique()) & set(X.iloc[:,l].dropna().unique()))
                if len(intersect) > 0:
                    res[i] = intersect
                    i +=1
        return res

    #check if shape greater than 1:
    if X.shape[1]>1:
        all_levels = find_intersection(X)
        while len(all_levels) != 0:
            y = X.isin(all_levels[0]).any()
            cols = list(y[y==True].index)
            if len(cols) > 1:
                for j in cols:
                    X[j] = X[j].map({x : "{}_{}".format(j,x) for x in list(X[j].dropna().unique())})
            all_levels = find_intersection(X)
          
    #convert to categorical
    for q in X.columns:
        X[q] = X[q].astype(CategoricalDtype(categories=sorted(list(X[q].dropna().unique())),ordered=True))
    return X