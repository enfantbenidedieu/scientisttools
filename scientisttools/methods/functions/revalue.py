# -*- coding: utf-8 -*-
from pandas import Series, DataFrame, Categorical

def revalue(X):
    """
    Revalue Categoricals Variables
    ------------------------------

    Description
    -----------
    Check if two categoricals variables have same levels and replace with new values

    Usage
    -----
    ```python
    >>> revalue(X)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_columns) or a pandas Series of shape (n_samples,)
        X contains categoricals variables.

    Return(s)
    ---------
    `Y`: a pandas DataFrame of shape (n_samples, n_columns)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if isinstance(X,Series): #convert to DataFrame if X is a pandas Series
        X = X.to_frame()
    
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #check if shape greater than 1:
    Y = X.copy()
    if Y.shape[1]>1:
        for i in range(X.shape[1]-1):
            for j in range(i+1,X.shape[1]):
                if (X.iloc[:,i].dtype in ["object","category"]) and (X.iloc[:,j].dtype in ["object","category"]):
                    intersect = list(set(X.iloc[:,i].dropna().unique().tolist()) & set(X.iloc[:,j].dropna().unique().tolist()))
                    if len(intersect)>=1:
                        valuei = {x : X.columns.tolist()[i]+"_"+str(x) for x in X.iloc[:,i].dropna().unique().tolist()}
                        valuej = {x : X.columns.tolist()[j]+"_"+str(x) for x in X.iloc[:,j].dropna().unique().tolist()}
                        Y.iloc[:,i], Y.iloc[:,j] = X.iloc[:,i].map(valuei), X.iloc[:,j].map(valuej)

    #convert to categorical
    for q in Y.columns:
        if Y[q].dtype in ["object","category"]:
            Y[q] = Categorical(Y[q],categories=sorted(Y[q].dropna().unique().tolist()),ordered=True)
    return Y