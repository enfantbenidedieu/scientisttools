# -*- coding: utf-8 -*-
from numpy import array, ones
from pandas import Series, DataFrame

#load intern functions
from .recodecont import recodecont
from .wpearsonr import wpearsonr

def contdesc(x,y,weights=None,proba=0.05) -> DataFrame:
    """
    Continuous variables description
    --------------------------------

    Description
    -----------
    Description continuous by quantitative variables

    Parameters
    ----------
    `x`: pandas series/dataframe of continuous variables of shape (n_rows,) or (n_rows, n_columns)

    `y`: pandas series of continues variables of shape (n_rows,)

    `weights` : an optional individuals weights 

    `proba` : the significance threshold considered to characterized the category (by default 0.05)

    Return
    ------
    `value` : pandas dataframe of shape (n_columns, 2)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if x is an instance of pandas series
    if isinstance(x,Series):
        x = x.to_frame() 
    
    # Check if x is an instance of pandas DataFrame
    if not isinstance(x,DataFrame):
        raise TypeError(f"{type(x)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # Set weights
    if weights is None:
        weights = ones(x.shape[0])/x.shape[0]
    else:
        weights = array([x/sum(weights) for x in weights])

    # Fill NA with the mean
    x = recodecont(x).X

    # For continuous variables
    value = DataFrame(index=x.columns,columns=["correlation","pvalue"]).astype("float")
    for col in x.columns:
        res = wpearsonr(x=x[col],y=y,weights=weights)
        value.loc[col,:] = [res.statistic,res.pvalue]
    value = value.query('pvalue < @proba').sort_values(by="correlation",ascending=False)
    return value