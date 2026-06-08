# -*- coding: utf-8 -*-

def get_mca_ind(
        obj
):
    """
    Extract the results for individuals - MCA

    Extract all the results (factor coordinates, squared cosinus, relative contributions and additional informations) for the active individuals from Multiple Correspondence Analysis (MCA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

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
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca_ind
    >>> clf = MCA(sup_var = range(4))
    >>> clf.fit(poison)
    MCA(sup_var = range(4))
    >>> #extract results for the individuals
    >>> ind = get_mca_ind(clf) 
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.infos.head() #additionals informations of individuals
    """
    if obj.__class__.__name__ != "MCA":
        raise TypeError("'obj' must be an object of class PCA")
    return obj.ind_
            
def get_mca_var(
        obj
):
    """
    Extract the results for the variables/levels - MCA
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active levels from Multiple Correspondence Analysis (MCA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    Returns
    -------
    result : levels
        An object containing all the results for the active variables/levels, with the following attributes:

        coord : DataFrame of shape (n_levels, n_components)
            The coordinates of the levels.
        coord_n : DataFrame of shape (n_levels, n_components)
            The normalized coordinates (barycentric) of the levels.
        cos2 : DataFrame of shape (n_levels, n_components)
            The squared cosinus of the levels.
        contrib : DataFrame of shape (n_levels, n_components)
            The relative contributions of the levels.
        vtest : DataFrame of shape (n_levels, n_components)
            The value-test of the levels.
        infos : DataFrame of shape (n_levels, 4)
            Additionnal informations (weight, square distance to origin, inertia and percentage of inertia) of levels.

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca_var
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison)
    MCA(sup_var=range(4))
    >>> #extract results for the variables
    >>> levels = get_mca_var(mca) 
    >>> levels.coord.head() #coordinates of the levels
    >>> levels.coord_n.head() #normalized coordinates (barycentric) of the levels
    >>> levels.cos2.head() #cos2 of the levels
    >>> levels.contrib.head() #contributions of the levels
    >>> levels.vtest.head() #vtest of the levels
    >>> levels.infos.head() #additionals informations of the levels
    """
    if obj.__class__.__name__ != "MCA":
        raise TypeError("'obj' must be an object of class MCA")
    return obj.levels_
    
def get_mca_quali_var(
        obj
):
    """
    Extract the results for the qualitative variables - MCA
    
    Extract all the results (coordinates, squared cosinus, relative contributions and additionals informations) for the active qualitative variables from Multiple Correspondence Analysis (MCA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    Returns
    -------
    result : quali_var
        An object containing all the results for the active qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_columns, n_components)
            The coordinates of the qualitative variable, which is eta-squared.
        contrib : DataFrame of shape (n_columns, n_components)
            The relative contributions of the qualitative variables.
        infos : DataFrame of shape (n_columns, 3)
            Additionnal informations (weight, inertia and percentage of inertia) of the qualitative variables.

    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca_var
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison)
    MCA(sup_var=range(4))
    >>> #extract results for the qualitative variables
    >>> quali_var = get_mca_quali_var(clf) 
    >>> quali_var.coord.head() #coordinates of the qualitative variables
    >>> quali_var.contrib.head() #contributions of the qualitative variables
    >>> quali_var.infos.head() #additionals informations of the qualitative variables
    """
    if obj.__class__.__name__  != "MCA":
        raise TypeError("'obj' must be an object of class MCA")
    return obj.quali_var_
    
def get_mca(
        obj, element="ind"
):
    """
    Extract the results for individuals/variables - MCA

    Extract all the results (coordinates, squared cosine, contributions and additionals informations) for the active individuals/variable categories from Multiple Correspondence Analysis (MCA) outputs.

        * :class:`~scientisttools.get_mca`: Extract the results for variables and individuals
        * :class:`~scientisttools.get_mca_ind`: Extract the results for individuals only
        * :class:`~scientisttools.get_mca_var`: Extract the results for variables/levels only
        * :class:`~scientisttools.get_mca_quali_var`: Extract the results for qualitative variables only
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MCA`.

    element : {"ind", "var", "quali_var"}, default = "ind"
        The element to subset from the output.
    
    Returns
    -------
    result : ind/levels/quali_var
        an object containing all the results for the active individuals/variable categories, with the following attributes:

        coord : DataFrame of shape (n_samples/n_levels/n_columns, n_components)
            The coordinates for the individuals/levels/qualitative variables.
        cos2 : DataFrame of shape (n_samples/n_levels/n_columns, n_components)
            The squared cosinus for the individuals/levels.
        contrib : DataFrame of shape (n_samples/n_levels/n_columns, n_components)
            The relative contributions for the individuals/levels/qualitative variables.
        infos : DataFrame of shape (n_samples/n_levels/n_columns, 4)
            Additionals informations for the individuals/levels/qualitative variables.
        
    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, get_mca
    >>> clf = MCA(sup_var=range(4))
    >>> clf.fit(poison)
    MCA(sup_var=range(4))
    >>> #extract results for the individuals
    >>> ind = get_mca(clf, element = "ind")
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.infos.head() #additionals informations of individuals
    >>> #extract results for the levels
    >>> var = get_mca(clf, element = "var")
    >>> var.coord.head() #coordinates of the levels
    >>> var.coord_n.head() #normalized coordinates of the levels
    >>> var.cos2.head() #cos2 of the levels
    >>> var.contrib.head() #contributions of the levels
    >>> var.vtest.head() #vtest of the levels
    >>> var.infos.head() #additionals informations of the levels
    >>> #extract results for the qualitative variables
    >>> quali_var = get_mca(clf, element = "quali_var")
    >>> quali_var.coord.head() #coordinates of the qualitative variables
    >>> quali_var.contrib.head() #contributions of the qualitative variables
    >>> quali_var.infos.head() #additionals informations of the qualitative variables
    """
    if element == "ind":
        return get_mca_ind(obj)
    elif element == "var":
        return get_mca_var(obj)
    elif element == "quali_var":
        return get_mca_quali_var(obj)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'quali_var'")