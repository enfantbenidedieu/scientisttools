# -*- coding: utf-8 -*-
from pandas import DataFrame, concat, get_dummies, CategoricalDtype
from pandas.api.types import is_numeric_dtype, is_string_dtype

def model_matrix(X):
    """
    Model Matrix - Construct Design Matrices

    Create a design (model) matrix. If variable is categoric, it create a disjuntive table without the first category.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    Return
    ------
    data : DataFrame of shape (n_samples, n_new_columns)
        Output data.
    """
    #check if X is an instance of class pd.DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    def recode(x):
        if is_numeric_dtype(x):
            return x
        elif is_string_dtype(x):
            uq_x = sorted(x.unique().tolist())
            x = x.astype(CategoricalDtype(categories=uq_x,ordered=True))
            return get_dummies(x,prefix=x.name,prefix_sep="",drop_first=True,dtype=int)
        else:
            raise TypeError("{} is not supported.".format(type(x)))
    
    data = concat((recode(x=X[k]) for k in X.columns),axis=1)
    return data