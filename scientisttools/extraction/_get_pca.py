# -*- coding: utf-8 -*-

def get_pca_ind(obj):
    """
    Extract the results for individuals - PCA/PCArot

    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) of the active individuals from Principal Component Analysis (PCA) outputs.

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.PCArot`.

    Returns
    -------
    result : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, ncp)
            the coordinates of the individuals.
        contrib : DataFrame of shape (n_samples, ncp)
            The relative contributions of the individuals.
        cos2 : DataFrame of shape (n_samples, ncp)
            The squared cosinus of the individuals.
        infos : DataFrame of shape (n_samples, 4) 
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.
            
    See Also
    --------
    get_pca : Extract the results for individuals/variables - PCA
    get_pca_var : Extract the results for variables - PCA.
    
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, get_pca_ind
    >>> clf = PCA(ind_sup=range(41,46), sup_var = (10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=range(41,46), sup_var = (10,11,12))
    >>> #extract the results for individuals
    >>> ind = get_pca_ind(clf)
    >>> ind.coord.head()   #coordinates of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.cos2.head()    #cos2 of individuals
    >>> ind.infos.head()   #additionals informations of individuals
    """
    if not (obj.__class__.__name__ in ("PCA","PCArot")):
        raise TypeError("obj must be an object of class PCA, PCArot")
    return obj.ind_

def get_pca_var(obj):
    """
    Extract the results for variables - PCA/PCArot
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Principal Component Analysis (PCA) outputs.

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.PCArot`.

    Returns
    -------
    result : quanti_var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the variables.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.
            
    See Also
    --------
    get_pca : Extract the results for individuals/variables - PCA
    get_pca_ind : Extract the results for individuals - PCA.

    Examples
    --------
    >>> from scientisttools.datasets import decathlon 
    >>> from scientisttools import PCA, get_pca_var
    >>> pca = PCA(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> pca.fit(decathlon.data)
    PCA(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> #extract the results for variables
    >>> quanti_var = get_pca_var(pca)
    >>> quanti_var.coord.head()   #coordinates of variables
    >>> quanti_var.contrib.head() #contributions of variables
    >>> quanti_var.cos2.head()    #cos2 of variables
    >>> quanti_var.infos.head()   #additionals informations of variables
    """
    if not (obj.__class__.__name__ in ("PCA","PCArot")):
        raise TypeError("obj must be an object of class PCA, PCArot")
    return obj.quanti_var_

def get_pca(obj,element="ind"):
    """
    Extract the results for individuals/variables - PCA/PCArot
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables from Principal Component Analysis (PCA) outputs.

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.PCA`, :class:`~scientisttools.PCArot`.

    element : {"ind","var"}, default = "ind"
        The element to subset from the output.
        
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    result : ind/var
        An object containing all the results for the active individuals/variables, with the following attributes:

        coord : DataFrame of shape (n_samples, ncp) or (n_columns, ncp)
            The coordinates of the individuals or variables.
        contrib : DataFrame of shape (n_samples, ncp) or (n_columns, ncp)
            The relative contributions of the individuals or variables.
        cos2 : DataFrame of shape (n_samples, ncp) or (n_columns, ncp)
            The squared cosinus of the individuals or variables.
        infos : DataFrame of shape (n_samples, 4) or (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals/variables.

    See Also
    --------
    get_pca_ind : Extract the results for individuals - PCA.
    get_pca_var : Extract the results for variables - PCA.
    
    Examples
    --------
    >>> from scientisttools.datasets import decathlon
    >>> from scientisttools import PCA, get_pca
    >>> clf = PCA(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    PCA(ind_sup=range(41,46),sup_var=(10,11,12))
    >>> #extract the results for individuals
    >>> ind = get_pca(clf,"ind")
    >>> ind.coord.head()   #coordinates of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.cos2.head()    #cos2 of individuals
    >>> ind.infos.head()   #additionals informations of individuals
    >>> #extract the results for variables
    >>> quanti_var = get_pca(clf,"var") 
    >>> quanti_var.coord.head()   #coordinates of variables
    >>> quanti_var.contrib.head() #contributions of variables
    >>> quanti_var.cos2.head()    #cos2 of variables
    >>> quanti_var.infos.head()   #additionals informations of variables
    """
    if element == "ind":
        return get_pca_ind(obj)
    elif element == "var":
        return get_pca_var(obj)
    else:
        raise ValueError("element should be one of 'ind', 'var'")