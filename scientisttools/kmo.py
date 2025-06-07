# -*- coding: utf-8 -*-
import pingouin as pg
import numpy as np
import pandas as pd

# Indice KMO Global
def kmo_index(x) -> pd.Series:
  corr, pcorr= x.corr(method="pearson").values, x.pcorr().values
  np.fill_diagonal(corr,0)
  np.fill_diagonal(pcorr,0)
  overall = np.sum(corr**2)/(np.sum(corr**2)+np.sum(pcorr**2))
  kmo_per_var = np.sum(corr**2, axis=0)/(np.sum(corr**2, axis=0)+np.sum(pcorr**2, axis=0))
  index = x.columns.tolist()
  index.insert(0,"overall")
  return pd.Series(np.append(overall,kmo_per_var),index=index,name="kmo")