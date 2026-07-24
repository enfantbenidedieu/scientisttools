# -*- coding: utf-8 -*-
from numpy import append
from pandas import Series

#intern function
from ..functions.utils import check_is_dataframe
from ..functions.statistics import wcorr, wpcorr

def kaisermsa(X, 
              w=None):
    """
    Calculate the Kaiser-Meyer-Olkin criterion for items and overall

    This statistic represents the degree to which each observed variable is predicted, without error, by the other variables in the dataset. In general, a KMO < 0.6 is considered inadequate.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        The DataFrame from which to calculate KMOs.

    w : 1d array-like of shape (n_samples,), default = None
        The weights of the individuals.

    Returns
    -------
    kmo: Series of shape (n_columns+1,)
        The KM0 criterion.
    """
    #check if X is an instance of pandas DataFrame class
    check_is_dataframe(X)
    #pearson correlation and partial correlation 
    corr, pcorr= wcorr(X=X,w=w), wpcorr(X=X,w=w)
    for c in corr.columns:
        corr.loc[c,c], pcorr.loc[c,c] = 0, 0
    overall = (corr**2).sum(axis=0).sum()/((corr**2).sum(axis=0).sum()+(pcorr**2).sum(axis=0).sum())
    kmo_per_var = (corr**2).sum(axis=0)/((corr**2).sum(axis=0)+(pcorr**2).sum(axis=0))
    index = X.columns.tolist()
    index.insert(0,"overall")
    return Series(append(overall,kmo_per_var),index=index,name="Kaiser's MSA")