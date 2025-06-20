# -*- coding: utf-8 -*-
from pandas import DataFrame,Series,concat
from collections import namedtuple

#intern functions
from .splitmix import splitmix
from .recodecont import recodecont
from .recodecat import recodecat

def recodevarfamd(X):
    """
    Recode variables for Factor Analysis of Mixed Data
    
    
    """
    # Check if pandas dataframe
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with ""pd.DataFrame. For more information see: ""https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #split dataset
    X_quanti, X_quali = splitmix(X).quanti, splitmix(X).quali

    # Set 
    nb_moda, dummies = None, None

    if X_quanti is not None:
        X_quanti = recodecont(X_quanti).X
        n1, k1 = X_quanti.shape
    
    if X_quali is not None:
        rec2 = recodecat(X_quali)
        X_quali, dummies = rec2.X, rec2.dummies
        n2, k2 = X_quali.shape
        nb_moda = Series([X_quali[j].nunique() for j in X_quali.columns],index=X_quali.columns)
    
    # Collapse result
    if X_quanti is not None and X_quali is not None:
        X, n, k = concat((X_quanti,X_quali),axis=1), n1, k1 + k2
    
    if X_quanti is not None and X_quali is None:
        X, n, k, k2 = X_quanti, n1, k1, 0
    
    if X_quanti is None and X_quali is not None:
        X, n, k, k1 = X_quali, n2, k2, 0

    return namedtuple("recodevar",["X","quanti","quali","dummies","nb_moda","n","k","k1","k2"])(X,X_quanti,X_quali,dummies,nb_moda,n,k,k1,k2)