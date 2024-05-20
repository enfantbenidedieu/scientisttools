
import numpy as np
import pandas as pd
from .splitmix import splitmix
from .recodecont import recodecont
from .recodecat import recodecat

def recodevar(X):
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
    dummies = None
    dummies_cod = None
    nb_moda = None

    if X_quanti is not None:
        n1, k1 = X_quanti.shape
        rec1 = recodecont(X=X_quanti)
        Z1 = rec1["Z"]
        means1 = rec1["means"]
        std1 = rec1["std"]
        Y1 = rec1["Xcod"]
    
    if X_quali is not None:
        n2, k2 = X_quali.shape
        dummies = recodecat(X_quali)
        means2 = dummies.mean(axis=0)
        nk = dummies.sum(axis=0) 
        std2 = np.sqrt(nk/n2)

        dummies_cod = dummies/std2.values.reshape(1,-1)
        means = dummies_cod.mean(axis=0)
        Z2 = (dummies_cod - means.values.reshape(1,-1))

        dummies_cent = dummies - (nk/n2).values.reshape(1,-1)
        nb_moda = pd.Series([X[col].nunique() for col in X_quali.columns],index=X_quali.columns)
    
    # Collapse result
    if X_quanti is not None and X_quali is not None:
        n = n1
        k = k1 + k2
        Z = pd.concat((Z1,Z2),axis=1)
        Y = pd.concat((Y1,dummies),axis=1)
        W = pd.concat((Z1,dummies_cent),axis=1)
        mean = pd.concat((means1,means2),axis=0)
        std = pd.concat((std1,std2))
    
    if X_quanti is not None and X_quali is None:
        n = n1
        k = k1
        k2 = 0
        Z = Z1
        Y = Y1
        W = Z1
        mean = means1
        std = std1
    
    if X_quanti is None and X_quali is not None:
        n = n2
        k = k2
        k1 = 0
        Z = Z2
        Y = dummies
        W = dummies_cent
        mean = means2
        std = std2

    return {"X" : X, "Y" : Y, "Z" : Z,"W": W, "n":n, "k" : k,"k1" : k1, "k2" : k2,"means" : mean, "std" : std, "nb_moda" : nb_moda,
            "dummies" : dummies, "dummies_cod" : dummies_cod,"quanti" : X_quanti, "quali" : X_quali}