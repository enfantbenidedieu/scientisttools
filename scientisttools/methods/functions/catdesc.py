 # -*- coding: utf-8 -*-
from pandas import concat, DataFrame, get_dummies
import statsmodels.formula.api as smf
from collections import namedtuple
from typing import NamedTuple

def catdesc(data,coord,proba=0.05) -> NamedTuple:
    """
    Categories description
    ----------------------

    Description
    -----------
    Description of the categories of one factor by categorical variables

    Usage
    -----
    ```python
    >>> catdesc(data,coord,proba=0.05)
    ```

    Parameters
    ----------
    `data` : pandas dataframe of qualitatives variables of shape (n_rows, n_columns)

    `coord`: pandas series of shape (n_rows,)

    `proba`: the significance threshold considered to characterized the category (by default 0.05)

    Return
    ------
    namedtuple of two dataframe

    `quali` : pandas dataframe

    `category` : pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Correlation ratio
    quali = DataFrame(index=data.columns,columns=['R2','pvalue']).astype("float")
    for col in data.columns:
        df = concat((coord,data[col]),axis=1)
        df.columns = ["y","x"]
        res = smf.ols(formula="y~C(x)", data=df).fit()
        quali.loc[col,:] = [res.rsquared,res.f_pvalue]
    # Subset using pvalue
    quali = quali.query('pvalue < @proba').sort_values(by="R2",ascending=False)

    # OLS regression
    dummies = concat((get_dummies(data[col],prefix=col,prefix_sep="=",dtype=int) for col in data.columns),axis=1)
    category = DataFrame(index=dummies.columns,columns=["Estimate","pvalue"]).astype("float")
    for col in dummies.columns:
        df = concat((coord,dummies[col]),axis=1)
        df.columns = ["y","x"]
        res = smf.ols(formula="y~C(x)", data=df).fit()
        category.loc[col,:] = [res.params.values[1],res.pvalues.values[1]]
    # Subset using pvalue
    category = category.query("pvalue < @proba")
    return namedtuple("CatDescResult",["quali","category"])(quali, category)