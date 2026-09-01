# -*- coding: utf-8 -*-

def get_fa_ind(obj):
    """
    Extract the results for individuals - FA/FArot
    
    Extract all the results of the active individuals from Factor Analysis (FA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`, :class:`~scientisttools.FArot`.

    Returns
    -------
    result : ind
        An object containing all the results for the active individuals including:

        coord : DataFrame of shape (n_samples, ncp)
            The coordinates of the individuals.
            
    See Also
    --------
    get_fa : Extract the results for individuals/variables - FA/FArot
    get_fa_var : Extract the results for variables - FA/FArot
    
    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, get_fa_ind
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    FA(max_iter=1,ncp=2)
    >>> #extract the results for individuals
    >>> ind = get_fa_ind(clf)
    >>> ind.coord.head() # coordinates of the individuals
    """
    if not (obj.__class__.__name__ in ("FA","FArot")):
        raise TypeError("'obj' must be an object of class FA, FArot")
    return obj.ind_

def get_fa_var(obj):
    """
    Extract the results for variables - FA/FArot
    
    Extract all the results for the active variables from Factor Analysis (FA) outputs.

    Parameters
    ----------
    obj : class 
        An instance of class :class:`~scientisttools.FA`, :class:`~scientisttools.FArot`.

    Returns
    -------
    result : var
        An object containing all the results for the active variables including:

        coord : DataFrame of shape (n_columns, ncp)
            The loadings of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The contribution of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, prior communality, final communality) of the variables.
            
    See Also
    --------
    get_fa : Extract the results for individuals/variables - FA/FArot
    get_fa_ind : Extract the results for individuals - FA/FArot
    
    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, get_fa_var
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    FA(max_iter=1,ncp=2)
    >>> #extract the results for variables
    >>> quanti_var = get_fa_var(clf)
    >>> quanti_var.coord.head()   #coordinates of variables
    >>> quanti_var.contrib.head() #contributions of variables
    """
    if obj.__class__.__name__ != "FA":
        raise TypeError("'obj' must be an object of class FA")
    return obj.quanti_var_

def get_fa(obj, element= "ind"):
    """
    Extract the results for individuals/variables - FA/FArot
    
    Extract all the results for the active individuals/variables from Factor Analysis (FA) outputs.

    Parameters
    ----------
    obj : class
        An instance of class :class:`~scientisttools.FA`, :class:`~scientisttools.FArot`.

    element : {"ind","var"}, default = "ind"
        The element to subset from the output. 
        
        * "ind" for individuals
        * "var" for variables
                
    Returns
    -------
    result : ind/var
        An object containing all the results for the active individuals/variables.

    See also
    --------
    get_fa_ind : Extract the results for individuals - FA/FArot.
    get_fa_var : Extract the results for variables - FA/FArot.

    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, get_fa
    >>> clf = FA(ncp=2,max_iter=1)
    >>> clf.fit(beer)
    FA(max_iter=1,ncp=2)
    >>> #extract the results for individuals
    >>> ind = get_fa(clf, "ind")
    >>> ind.coord # coordinates of individuals
    >>> #extract the results for variables
    >>> quanti_var = get_fa(clf, "var")
    >>> quanti_var.coord.head()   #coordinates of variables
    >>> quanti_var.contrib.head() #contributions of variables
    """
    if element == "ind":
        return get_fa_ind(obj)
    elif element == "var":
        return get_fa_var(obj)
    else:
        ValueError("element should be one of 'ind', 'var'")