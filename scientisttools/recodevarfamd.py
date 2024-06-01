
import numpy as np
import pandas as pd
from .splitmix import splitmix
from .recodecont import recodecont
from .recodecat import recodecat

def recodevarfamd(X):
    """
    
    
    
    """
    # Check if pandas dataframe
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X_quanti = splitmix(X)["quanti"]
    X_quali = splitmix(X)["quali"]

    # Set 
    nb_moda = None
    dummies = None

    if X_quanti is not None:
        X_quanti = recodecont(X_quanti)["Xcod"]
        n1, k1 = X_quanti.shape
    
    if X_quali is not None:
        n2, k2 = X_quali.shape
        rec2 = recodecat(X_quali)
        X_quali = rec2["X"]
        dummies = rec2["dummies"]
        nb_moda = pd.Series([X_quali[col].nunique() for col in X_quali.columns],index=X_quali.columns)
    
    # Collapse result
    if X_quanti is not None and X_quali is not None:
        X = pd.concat((X_quanti,X_quali),axis=1)
        n = n1
        k = k1 + k2
    
    if X_quanti is not None and X_quali is None:
        X = X_quanti
        n = n1
        k = k1
        k2 = 0
    
    if X_quanti is None and X_quali is not None:
        X = X_quali
        n = n2
        k = k2
        k1 = 0

    return {"X" : X, "n":n, "k" : k,"k1" : k1, "k2" : k2, "nb_moda" : nb_moda,
            "quanti" : X_quanti, "quali" : X_quali,"dummies" : dummies}