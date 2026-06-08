# -*- coding: utf-8 -*-

def get_dmfa_ind(
        obj
):
    """
    Extract the results for individuals - DMFA
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals from Dual Multiple Factor Analysis (DMFA) outputs.

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.DMFA`.

    Returns
    -------
    result : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, n_components)
            the coordinates of the individuals.
        contrib : DataFrame of shape (n_samples, n_components)
            The relative contributions of the individuals.
        cos2 : DataFrame of shape (n_samples, n_components)
            The squared cosinus of the individuals.
        infos : DataFrame of shape (n_samples, 4) 
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, get_dmfa_ind
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    DMFA(group=4)
    >>> #extract all results for the individuals
    >>> ind = get_dmfa_ind(clf)
    >>> ind.coord.head() #coordinates for the individuals
    >>> ind.cos2.head() #cos2 for the individuals
    >>> ind.contrib.head() #contributions for the individuals
    >>> ind.infos.head() #additionals informations for the individuals
    """
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    return obj.ind_

def get_dmfa_var(
        obj
):
    """
    Extract the results for variables - DMFA
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active variables from Dual Multiple Factor Analysis (DMFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.DMFA`.

    Returns
    -------
    result : var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, n_components)
            The coordinates of the variables.
        contrib : DataFrame of shape (n_columns, n_components)
            The relative contributions of the variables.
        cos2 : DataFrame of shape (n_columns, n_components)
            The squared cosinus of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, get_dmfa_var
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    >>> #extract all the results for the variables
    >>> var = get_dmfa_var(clf)
    >>> var.coord.head() #coordinates of variables
    >>> var.contrib.head() #contributions of variables
    >>> var.cos2.head() #cos2 of variables
    >>> var.infos.head() #additionals informations of variables
    """
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    return obj.quanti_var_
    
def get_dmfa_group(
        obj
):
    """
    Extract the results for groups - DMFA
    
    Extract all the results (coordinates, normalized coordinates and squared cosinus) for the active groups from Dual Multiple Factor Analysis (DMFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.DMFA`.
    
    Returns
    -------
    result : group
        An object containing all the results for the active groups, with the following attributes:

        coord : DataFrame of shape (n_groups, n_components)
            The coordinates of the groups.
        coord_n : DataFrame of shape (n_groups, n_components)
            The normalized coordinates of the groups.
        cos2 : DataFrame of shape (n_groups, n_components)
            The squared cosinus of the groups.

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, get_dmfa_group
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    >>> #extract all the results for the variables
    >>> group = get_dmfa_group(clf)
    >>> group.coord.head() #coordinates of groups
    >>> group.coord_n.head() #normalized coordinates of groups
    >>> group.cos2.head() #cos2 of groups
    """
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    return obj.group_
   
def get_dmfa(
        obj,element="ind"
):
    """
    Extract the results for individuals/variables/groups - DMFA
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active individuals/variables/groups from Dual Multiple Factor Analysis (DMFA) outputs:
    
        * :class:`~scientisttools.get_dmfa`: Extract the results for variables/groups and individuals
        * :class:`~scientisttools.get_dmfa_ind`: Extract the results for individuals only
        * :class:`~scientisttools.get_dmfa_var`: Extract the results for variables only
        * :class:`~scientisttools.get_dmfa_group`: Extract the results for groups only
    
    Parameters
    ----------
    obj : 
        An object of class :class:`~scientisttools.DMFA`.

    element : {"ind","var","group"}, default = "ind"
        The element to subset from the output. 

    Returns
    -------
    result : ind/var/group
        An object containing all the results for the active individuals/variables/groups.

    See also
    --------
    :class:`~scientisttools.get_dmfa_group`
        Extract the results for groups - DMFA.
    :class:`~scientisttools.get_dmfa_ind`
        Extract the results for individuals - DMFA.
    :class:`~scientisttools.get_dmfa_var`
        Extract the results for variables - DMFA.

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, get_dmfa
    >>> clf = DMFA(group=4)
    >>> clf.fit(iris)
    >>> #extract the results for individuals
    >>> ind = get_dmfa(clf,"ind")
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.infos.head() #additionals informations of individuals
    >>> #extract the results for variables
    >>> var = get_dmfa(clf,"var") 
    >>> var.coord.head() #coordinates of variables
    >>> var.contrib.head() #contributions of variables
    >>> var.cos2.head() #cos2 of variables
    >>> var.infos #additionals informations of variables
    >>> group = get_dmfa(clf,"group")
    >>> group.coord.head() #coordinates of groups
    >>> group.coord_n.head() #normalized coordinates of groups
    >>> group.cos2.head() #cos2 of groups
    """
    if element == "ind":
        return get_dmfa_ind(obj)
    elif element == "var":
        return get_dmfa_var(obj)
    elif element == "group":
        return get_dmfa_group(obj)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'group'")