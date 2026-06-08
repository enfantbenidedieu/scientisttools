# -*- coding: utf-8 -*-

def get_mix_ind(
        obj
):
    """
    Extract the results for individuals - FAMD/PCAmix/MPCA
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the individuals from Factor Analysis of Mixed Data (FAMD),
    Principal Component Analysis of Mixed Data (PCAmix) and Mixed Principal Component Analysis (MPCA) outputs.

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`.

    Returns
    -------
    result : ind
        An object containing all the results for the individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, n_components)
            The coordinates for the individuals.
        cos2 : DataFrame of shape (n_samples, n_components)
            The squared cosinus for the individuals.
        contrib : DataFrame of shape (n_samples, n_components)
            The relative contributions for the individuals.
        infos : DataFrame of shape (n_samples, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the individuals.

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_mix_ind
    >>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> clf.fit(autos2005)
    FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> #extract results for individuals
    >>> ind = get_mix_ind(clf)
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    """
    if not (obj.__class__.__name__ in ("FAMD","PCAmix","MPCA")):
        raise ValueError("'obj' must be an object of class FAMD, PCAmix, MPCA")
    return obj.ind_
    
def get_mix_quanti_var(
        obj
):
    """
    Extract the results for quantitative variables - FAMD/PCAmix/MPCA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for continuous variables from Factor Analysis of Mixed Data (FAMD), 
    Principal Component Analysis of Mixed Data (PCAmix) and Mixed Principal Component Analysis (MPCA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`.

    Returns
    -------
    result : quanti_var
        An object containing the results for the continuous variables, with the following attributes:

        coord : DataFrame of shape (n_quanti, n_components)
            The coordinates for the quantitative variables.
        cos2 : DataFrame of shape (n_quanti, n_components)
            The squared cosinus for the quantitative variables.
        contrib : DataFrame of shape (n_quanti, n_components)
            The relative contributions for the quantitative variables.

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_mix_quanti_var
    >>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> clf.fit(autos2005)
    FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> #extract results for quantitatives variables
    >>> quanti_var = get_mix_quanti_var(clf)
    >>> quanti_var.coord #coordinates of quantitative variables
    >>> quanti_var.cos2 #cos2 of quantitative variables
    >>> quanti_var.contrib #contribution of quantitative variables
    """
    if not (obj.__class__.__name__ in ("FAMD","PCAmix","MPCA")):
        raise ValueError("'obj' must be an object of class FAMD, PCAmix, MPCA")
    return obj.quanti_var_
    
def get_mix_quali_var(
        obj
):
    """
    Extract the results for qualitative variables/levels - FAMD/PCAmix/MPCA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for categorical variables/levels from Factor Analysis of Mixed Data (FAMD),
    Principal Component Analysis of Mixed Data (PCAmix) and Mixed Principal Component Analysis (MPCA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`.

    Returns
    -------
    result : quali_var
        An object containing the results for the qualitative variables/levels, with the following attributes:

        coord : DataFrame of shape (n_levels, n_components)
            The coordinates for the levels.
        cos2 : DataFrame of shape (n_levels, n_components)
            The squared cosinus for the levels.
        contrib : DataFrame of shape (n_levels, n_components)
            The relative contributions for the levels.
        vtest : DataFrame of shape (n_levels, n_components)
            The value-test for the levels.
        dist2 : Series of shape (n_levels,)
            The squared distance to origin for the levels.

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_mix_quali_var
    >>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> clf.fit(autos2005)
    FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> #extract results for qualitatives variables
    >>> quali_var = get_mix_quali_var()
    >>> quali_var.coord.head() #coordinates of levels
    >>> quali_var.cos2.head() #cos2 of levels
    >>> quali_var.contrib.head() #contribution of levels
    >>> quali_var.vtest.head() #value-test of levels
    >>> quali_var.dist2.head() #dist2 of levels
    """
    if not (obj.__class__.__name__ in ("FAMD","PCAmix","MPCA")):
        raise ValueError("'obj' must be an object of class FAMD, PCAmix, MPCA")
    return obj.quali_var_
    
def get_mix_var(
        obj
):
    """
    Extract the results for variables - FAMD/PCAmix/MPCA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for quantitative and qualitative variables from Factor Analysis of Mixed Data (FAMD),
    Principal Component Analysis of Mixed Data (PCAmix) and Mixed Principal Component Analysis (MPCA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`.

    Returns
    -------
    result : var
        An object containing the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, n_components)
            The coordinates for the variables,
        cos2 : DataFrame of shape (n_columns, n_components)
            The squared cosinus for the variables,
        contrib : DataFrame of shape (n_columns, n_components)
            The relative contributions for the variables.

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_mix_var
    >>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> clf.fit(autos2005)
    FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> #extract results for variables
    >>> var = get_mix_var(clf)
    >>> var.coord.head() #coordinates of variables
    >>> var.cos2.head() #cos2 of variables
    >>> var.contrib.head() #contribution of variables
    """
    if not (obj.__class__.__name__ in ("FAMD","PCAmix","MPCA")):
        raise ValueError("'obj' must be an object of class FAMD, PCAmix, MPCA")
    return obj.var_

def get_mix(
        obj, element = "ind"
):
    """
    Extract the results for individuals and variables - FAMD/PCAmix/MPCA
    
    Extract all the results (coordinates, squared cosine and relative contributions) for the individuals and variables from Factor Analysis of Mixed Data (FAMD), 
    Principal Component Analysis of Mixed Data (PCAmix) and Mixed Principal Component Analysis (MPCA) outputs.

        * :class:`~scientisttools.get_mix`: Extract the results for variables and individuals
        * :class:`~scientisttools.get_mix_ind`: Extract the results for individuals only
        * :class:`~scientisttools.get_mix_var`: Extract the results for variables only
        * :class:`~scientisttools.get_mix_quali_var`: Extract the results for qualitative variables/levels only
        * :class:`~scientisttools.get_mix_quanti_var`: Extract the results for quantitative variables only

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FAMD`, :class:`~scientisttools.PCAmix`, :class:`~scientisttools.MPCA`.

    element : {"ind","var","quali_var","quanti_var"}, default = "ind"
        The element to subset from the output. Possible values are: 

    Returns
    -------
    result : ind/quanti_var/quali_var/var
        An object containing the results for the active individuals and variables.

    See also
    --------
    :class:`~scientisttools.get_mix_ind`
        Extract the results for individuals - FAMD/PCAmix/MPCA.
    :class:`~scientisttools.get_mix_quanti_var`
        Extract the results for quantitative variables - FAMD/PCAmix/MPCA.
    :class:`~scientisttools.get_mix_quali_var`
        Extract the results for qualitative variables/levels - FAMD/PCAmix/MPCA.
    :class:`~scientisttools.get_mix_var`
        Extract the results for variables - FAMD/PCAmix/MPCA.

    Examples
    --------
    >>> from scientisttools.datasets import autos2005
    >>> from scientisttools import FAMD, get_mix
    >>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> clf.fit(autos2005)
    FAMD(ind_sup=range(38,45),sup_var=range(12,16))
    >>> #extract results for individuals
    >>> ind = get_mix(clf, "ind")
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> #extract results for quantitatives variables
    >>> quanti_var = get_mix(clf, "quanti_var")
    >>> quanti_var.coord.head() #coordinates of quantitative variables
    >>> quanti_var.cos2.head() #cos2 of quantitative variables
    >>> quanti_var.contrib.head() #contribution of quantitative variables
    >>> #extract results for qualitatives variables
    >>> quali_var = get_mix(clf, "quali_var")
    >>> quali_var.coord.head() #coordinates of categories/variables
    >>> quali_var.cos2.head() #cos2 of categories/variables
    >>> quali_var.contrib.head() #contribution of categories/variables
    >>> quali_var.vtest.head() #value-test of categories/variables
    >>> #extract results for variables
    >>> var = get_mix(clf, "var")
    >>> var.coord.head() #coordinates of variables
    >>> var.cos2.head() #cos2 of cvariables
    >>> var.contrib.head() #contribution of variables
    """
    if element == "ind":
        return get_mix_ind(obj)
    elif element == "var":
        return get_mix_var(obj)
    elif element == "quanti_var":
        return get_mix_quanti_var(obj)
    elif element == "quali_var":
        return get_mix_quali_var(obj)
    else:
        raise ValueError("'element' should be one of 'ind', 'ind_sup', 'quanti_var', 'quali_var', 'var'")