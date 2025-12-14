# -*- coding: utf-8 -*-
from pandas import DataFrame

def get_sup_label(X:DataFrame,indexes=None,axis=0) -> None|list:
    """
    Get supplementary labels
    ------------------------

    Description
    -----------
    Get supplementary labels

    Usage
    -----
    ```
    >>> get_sup_label(X,indexes,axis)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_columns)
        Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

    `indexes`: None or an integer or a string or a list or a tuple or a range indicating the indexes or names of the supplementary elements (rows or columns).

    `axis`: None or a string or an integer indicating which axis to aggregate
        * None or 0 or "index" indicates aggregating along rows
        * 1 or "columns" indicates aggregating along columns

    Return(s)
    --------
    `label`: None or a list indicating the the names of the supplementary elements.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set labels
    if axis in [None, 0, "index"]:
        names = X.index
    elif axis in [1, "columns"]:
        names = X.columns
    else:
        raise ValueError("'axis' must be either index (0) or columns (1).")

    if indexes is not None:
        if isinstance(indexes,str): #this is a string
            label = [indexes]
        elif isinstance(indexes,(int,float)): #this is an integer or a float
            label = [names[int(indexes)]]
        elif isinstance(indexes,range): #this is a range
            label = names[list(indexes)].tolist()
        elif isinstance(indexes,(list,tuple)): #this is a list or a tuple
            if all(isinstance(x,str) for x in indexes):
                label = [str(x) for x in indexes]
            elif all(isinstance(x,(int,float)) for x in indexes):
                label = names[[int(x) for x in indexes]].tolist()
            else:
                raise TypeError("All elements in list or tuple must have the same type: either string or integer.")
        else:
            raise TypeError("'indexes' must be either None or an integer or a string or a list or a tuple or a range indicating the indexes or names of the supplementary elements.")
    else:
        label = None

    return label