# -*- coding: utf-8 -*-
from numpy import sqrt, ones, array,ndarray
from pandas import DataFrame,Series,concat
from collections import namedtuple, OrderedDict

#intern functions
from .splitmix import splitmix
from .recodecont import recodecont
from .recodecat import recodecat

def recodevarfamd(X,weights=None):
    """
    Recode variables for Factor Analysis of Mixed Data (FAMD)
    ---------------------------------------------------------

    Usage
    -----
    ```python
    >>> recodevarfamd(X,weights=None)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with `n` rows and `k` columns (quantitative and/or qualitative)

    Return
    ------
    a namedtuple

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import recodevarfamd
    >>> rec = recodevarfamd(wine)
    ```
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with ""pd.DataFrame. For more information see: ""https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/1darray/Series of individuals weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/1darray/Series with length {X.shape[0]}.")
    else:
        weights = array(list(map(lambda x : x/sum(weights),weights)))

    #split dataset
    X_quanti, X_quali = splitmix(X).quanti, splitmix(X).quali

    #set 
    nb_moda, dummies = None, None

    if X_quanti is not None:
        n1, k1 = X_quanti.shape
        rec1 = recodecont(X=X_quanti,weights=weights)
        X_quanti, Z1, center1, scale1 = rec1.X, rec1.Z, rec1.center, rec1.scale 
        
    if X_quali is not None:
        n2, k2 = X_quali.shape
        rec2 = recodecat(X_quali)
        X_quali, dummies = rec2.X, rec2.dummies
        center2 = dummies.apply(lambda x :  x*weights,axis=0).sum(axis=0)
        center2.name, scale2 = "center", Series(sqrt(center2),index=dummies.columns,name="scale")
        Z2 = dummies.apply(lambda x : (x - center2)/scale2,axis=1)
        nb_moda = Series([X_quali[j].nunique() for j in X_quali.columns],index=X_quali.columns)

    #collapse result
    if X_quanti is not None and X_quali is not None:
        n, k, Z, center, scale = n1, k1 + k2, concat((Z1,Z2),axis=1), concat((center1,center2),axis=0), concat((scale1,scale2))
    
    if X_quanti is not None and X_quali is None:
        n, k, k2, Z, center, scale = n1, k1, 0, Z1, center1, scale1
    
    if X_quanti is None and X_quali is not None:
        n, k, k1, Z, center, scale = n2, k2, 0, Z2, center2, scale2

    #convert to dict
    res = OrderedDict(X=X,quanti=X_quanti,quali=X_quali,Z=Z,dummies=dummies,n=n,k=k,k1=k1,k2=k2,center=center,scale=scale,nb_moda=nb_moda)
    return namedtuple("recodevarFAMD",res.keys())(*res.values())

def recodevarpcamix(X, weights=None):
    """
    Recode variables for Principal Component Analysis of Mixed Data (PCAMIX)
    ------------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> recodevarpcamix(X,weights=None)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with `n` rows and `k` columns (quantitative and/or qualitative)

    `weights`: an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals

    Return
    ------
    a namedtuple

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import recodevarpcamix
    >>> rec = recodevarpcamix(wine)
    ```
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with ""pd.DataFrame. For more information see: ""https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/1darray/Series of individuals weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/1darray/Series with length {X.shape[0]}.")
    else:
        weights = array(list(map(lambda x : x/sum(weights),weights)))

    #split dataset
    X_quanti, X_quali = splitmix(X).quanti, splitmix(X).quali

    #initialize variables
    dummies, nb_moda = None, None

    if X_quanti is not None:
        n1, k1 = X_quanti.shape
        rec1 = recodecont(X=X_quanti, weights=weights)
        X_quanti, Z1, center1, scale1 = rec1.X, rec1.Z, rec1.center, rec1.scale 
    
    if X_quali is not None:
        n2, k2 = X_quali.shape
        rec2 = recodecat(X_quali)
        X_quali, dummies = rec2.X, rec2.dummies
        center2 = dummies.apply(lambda x : x*weights,axis=0).sum(axis=0)
        center2.name, scale2 =  "center" , Series(ones(dummies.shape[1]),index=dummies.columns,name="scale")
        Z2 = dummies.apply(lambda x : (x - center2)/scale2,axis=1)
        nb_moda = Series([X_quali[col].nunique() for col in X_quali.columns],index=X_quali.columns,name="count")
    
    #collapse result
    if X_quanti is not None and X_quali is not None:
        n, k, Z, center, scale = n1, k1 + k2, concat((Z1,Z2),axis=1), concat((center1,center2),axis=0), concat((scale1,scale2))
    
    if X_quanti is not None and X_quali is None:
        n, k, k2, Z, center, scale = n1, k1, 0, Z1, center1, scale1
    
    if X_quanti is None and X_quali is not None:
        n, k, k1, Z, center, scale = n2, k2, 0, Z2, center2, scale2

    #convert to dict
    res = OrderedDict(X=X,quanti=X_quanti,quali=X_quali,Z=Z,dummies=dummies,n=n,k=k,k1=k1,k2=k2,center=center,scale=scale,nb_moda=nb_moda)
    return namedtuple("recodevarPCAMIX",res.keys())(*res.values())

def recodevarhillsmith(X, weights=None):
    """
    Recode variables for Hill and Smith Analyis of Mixed Data (HillSmith)
    ----------------------------------------------------------------------

    Usage
    -----
    ```python
    >>> recodevarhillsmith(X,weights)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with `n` rows and `k` columns (quantitative and/or qualitative)

    `weights`: an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals

    Return
    ------
    a namedtuple

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import recodevarhillsmith
    >>> rec = recodevarhillsmith(wine)
    ```
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with ""pd.DataFrame. For more information see: ""https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set weights
    if weights is None:
        weights = ones(X.shape[0])/X.shape[0]
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/1darray/Series of individuals weights.")
    elif len(weights) != X.shape[0]:
        raise ValueError(f"'weights' must be a list/tuple/1darray/Series with length {X.shape[0]}.")
    else:
        weights = array(list(map(lambda x : x/sum(weights),weights)))

    #split dataset
    X_quanti, X_quali = splitmix(X).quanti, splitmix(X).quali

    #initialize variables
    dummies, nb_moda = None, None

    if X_quanti is not None:
        n1, k1 = X_quanti.shape
        rec1 = recodecont(X=X_quanti, weights=weights)
        X_quanti, Z1, center1, scale1 = rec1.X, rec1.Z, rec1.center, rec1.scale 
    
    if X_quali is not None:
        n2, k2 = X_quali.shape
        rec2 = recodecat(X_quali)
        X_quali, dummies = rec2.X, rec2.dummies
        center2 = dummies.apply(lambda x : x*weights,axis=0).sum(axis=0)
        center2.name, scale2 =  "center" , Series(center2.values,index=dummies.columns,name="scale")
        Z2 = dummies.apply(lambda x : (x - center2)/scale2,axis=1)
        nb_moda = Series([X_quali[col].nunique() for col in X_quali.columns],index=X_quali.columns,name="count")
    
    #collapse result
    if X_quanti is not None and X_quali is not None:
        n, k, Z, center, scale = n1, k1 + k2, concat((Z1,Z2),axis=1), concat((center1,center2),axis=0), concat((scale1,scale2))
    
    if X_quanti is not None and X_quali is None:
        n, k, k2, Z, center, scale = n1, k1, 0, Z1, center1, scale1
    
    if X_quanti is None and X_quali is not None:
        n, k, k1, Z, center, scale = n2, k2, 0, Z2, center2, scale2

    #convert to dict
    res = OrderedDict(X=X,quanti=X_quanti,quali=X_quali,Z=Z,dummies=dummies,n=n,k=k,k1=k1,k2=k2,center=center,scale=scale,nb_moda=nb_moda)
    return namedtuple("recodevarHillSmith",res.keys())(*res.values())

def recodevarmpca(X:DataFrame,weights=None):
    """
    Recode variables for Mixed Principal components Anslysis (MPCA)
    ---------------------------------------------------------------

    Usage
    -----
    ```python
    >>> recodevarmpca(X,weights)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame with `n` rows and `k` columns (quantitative and/or qualitative)

    `weights`: an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals

    Return
    ------
    a namedtuple

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```
    >>> from scientisttools.datasets import autos1990
    >>> from scientisttools import recodevarmpca
    >>> rec = recodevarmpca(autos1990)
    ```
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with ""pd.DataFrame. For more information see: ""https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #et number of rows
    n_rows = X.shape[0]

    #set weights
    if weights is None:
        weights = ones(n_rows)/n_rows
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list/tuple/1darray/Series of individuals weights.")
    elif len(weights) != n_rows:
        raise ValueError(f"'weights' must be a list/tuple/1darray/Series with length {n_rows}.")
    else:
        weights = array(list(map(lambda x : x/sum(weights),weights)))

    #split dataset
    X_quanti, X_quali = splitmix(X).quanti, splitmix(X).quali

    #set number of quantitative and qualitative variables
    k1, k2 = X_quanti.shape[1], X_quali.shape[1]

    #initialize variables
    dummies, nb_moda = None, None

    if X_quanti is not None:
        n1, k1 = X_quanti.shape
        rec1 = recodecont(X=X_quanti, weights=weights)
        X_quanti, Z1, center1, scale1 = rec1.X, rec1.Z, rec1.center, rec1.scale 
    
    if X_quali is not None:
        n2, k2 = X_quali.shape
        rec2 = recodecat(X_quali)
        X_quali, dummies = rec2.X, rec2.dummies
        center2 = dummies.apply(lambda x : x*weights,axis=0).sum(axis=0)
        center2.name, scale2 =  "center" , Series(center2.values,index=dummies.columns,name="scale")
        Z2 = dummies.apply(lambda x : (x - center2)/scale2,axis=1)
        nb_moda = Series([X_quali[col].nunique() for col in X_quali.columns],index=X_quali.columns,name="count")
    
    #collapse result
    if X_quanti is not None and X_quali is not None:
        n, k, Z, center, scale = n1, k1 + k2, concat((Z1,Z2),axis=1), concat((center1,center2),axis=0), concat((scale1,scale2))
    
    if X_quanti is not None and X_quali is None:
        n, k, k2, Z, center, scale = n1, k1, 0, Z1, center1, scale1
    
    if X_quanti is None and X_quali is not None:
        n, k, k1, Z, center, scale = n2, k2, 0, Z2, center2, scale2

    #convert to dict
    res = OrderedDict(X=X,quanti=X_quanti,quali=X_quali,Z=Z,dummies=dummies,n=n,k=k,k1=k1,k2=k2,center=center,scale=scale,nb_moda=nb_moda)
    return namedtuple("recodevarHillSmith",res.keys())(*res.values())
