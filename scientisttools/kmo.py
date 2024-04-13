# -*- coding: utf-8 -*-
import pingouin as pg
import numpy as np
import pandas as pd

# Indice KMO Global
def global_kmo_index(x):
  # Matrice des corrélations
  corr = x.corr(method="pearson").values
  # Matrice des corrélations partielles
  pcorr = x.pcorr().values
  # Indice KMO global
  np.fill_diagonal(corr,0)
  np.fill_diagonal(pcorr,0)
  return np.sum(corr**2)/(np.sum(corr**2)+np.sum(pcorr**2))

# Indice KMO par variable
def per_item_kmo_index(x):
  # Matrice des corrélations linéaires
  corr = x.corr(method = "pearson").values
  # Matrice des corrélations partielles
  pcorr = x.pcorr().values
  # Indice KMO global
  np.fill_diagonal(corr,0)
  np.fill_diagonal(pcorr,0)
  A = np.sum(corr**2, axis=0)
  B = np.sum(pcorr**2, axis=0)
  kmo_per_item = A /(A+B)
  return pd.Series(kmo_per_item,index=x.columns.tolist(),name="KMO")