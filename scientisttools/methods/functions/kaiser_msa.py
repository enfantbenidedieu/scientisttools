# -*- coding: utf-8 -*-
from numpy import fill_diagonal, append
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype

#intern function
from .wpcorrcoef import wpcorrcoef

# Kaiser's Measure of Sampling Adequacy (MSA)
def kaiser_msa(X) -> Series:
  """
  Calculate the Kaiser-Meyer-Olkin criterion for items and overall
  ----------------------------------------------------------------

  Description
  -----------
  This statistic represents the degree to which each observed variable is predicted, without error, by the other variables in the dataset. In general, a KMO < 0.6 is considered inadequate.

  Parameters
  ----------
  `X`: a pandas DataFrame of shape (n_samples, n_columns)
      The DataFrame from which to calculate KMOs.

  Returns
  -------
  
  """
  if not isinstance(X,DataFrame):#check if X is an instance of pandas DataFrame class
      raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
  
  if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
      raise TypeError("All columns must be numeric")
  
  corr, pcorr= X.corr(method="pearson"), wpcorrcoef(X)
  fill_diagonal(corr.values,0)
  fill_diagonal(pcorr.values,0)
  overall = corr.pow(2).sum().sum()/(corr.pow(2).sum().sum()+pcorr.pow(2).sum().sum())
  kmo_per_var = corr.pow(2).sum(axis=0)/(corr.pow(2).sum(axis=0)+pcorr.pow(2).sum(axis=0))
  index = X.columns.tolist()
  index.insert(0,"overall")
  return Series(append(overall,kmo_per_var),index=index,name="Kaiser's MSA")