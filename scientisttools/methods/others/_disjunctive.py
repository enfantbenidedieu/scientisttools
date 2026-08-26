# -*- coding: utf-8 -*-
from numpy import zeros
from pandas import DataFrame, Series, concat, get_dummies

def disjunctive(X, 
                cols=None, 
                prefix=False, 
                sep="_"):
    """
    Convert categorical variable into dummy/indicator variables

    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Input data.

    cols : None or list, default = None
        Columns from orginal disjunctive table.

    prefix : bool, default = False
        If True, append DataFrame with columns names.

    sep : str, default = "_"
        If appending prefix, separator/delimiter to use.

    Returns
    -------
    DataFrame : DataFrame of shape (n_samples, n_categories)
        Dummy-coded data.
    """
    #convert to DataFrame if Series
    if isinstance(X, Series): X = X.to_frame()

    def func_dummies(x, 
                     prefix = False, 
                     sep = "_"):
        """
        Binary coding

        Parameters
        ----------
        x : Series of shape (n_samples,)
            Input data.

        prefix : bool, default = False
            If True, append DataFrame with columns names.

        sep : str, default = "_"
            If appending prefix, separator/delimiter to use.

        Returns
        -------
        DataFrame :
            Dummy-coded data.
        """
        return get_dummies(x,prefix=x.name,prefix_sep=sep,dtype=int) if prefix else get_dummies(x,dtype=int)

    #dummies
    df = concat((func_dummies(X[j],prefix=prefix,sep=sep) for j in list(X.columns)),axis=1)
    #update if cols is not None
    if cols is not None:
        #initialize
        df_all = DataFrame(zeros((X.shape[0],len(cols))),index=X.index,columns=cols)
        #update with dummies in new individuals
        if len(cols) >= df.shape[1]:
            df_all.loc[:,list(df.columns)] = df
        else:
            df_all.loc[:,cols] = df.loc[:,cols]
    else:
        df_all = df.copy()
    return df_all