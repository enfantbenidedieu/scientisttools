# -*- coding: utf-8 -*-
import pandas as pd

def get_melt(X,level=- 1, dropna=True):
    """
    Stack the prescribed level(s) from columns to index
    --------------------------------------------------

    Return a reshaped DataFrame or Series having a multi-level index with one or more 
    new inner-most levels compared to the current DataFrame. The new inner-most levels 
    are created by pivoting the columns of the current dataframe:

    Parameters
    ----------
    X       : DataFrame
    level   : int, str, list, default -1
            Level(s) to stack from the column axis onto the index axis, 
            defined as one index or label, or a list of indices or labels.
    dropna  : bool, default True
            Whether to drop rows in the resulting Frame/Series with missing values. 
            Stacking a column level onto the index axis can create combinations of index 
            and column values that are missing from the original dataframe.

    Return
    ------
        Stacked dataframe or series.
    
    """
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
                f"{type(X)} is not supported. Please convert to a DataFrame with "
                "pd.DataFrame. For more information see: "
                "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    return X.stack(level=level, dropna=dropna).rename_axis(('Var1', 'Var2')).reset_index(name='value')