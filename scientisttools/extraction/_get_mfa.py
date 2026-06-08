# -*- coding: utf-8 -*-

def get_mfa_ind(
        obj
):
    """
    Extract the results for individuals - MFA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active individuals from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.

    Returns
    -------
    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, n_components)
            the coordinates of the individuals.
        contrib : DataFrame of shape (n_samples, n_components)
            The relative contributions of the individuals.
        cos2 : DataFrame of shape (n_samples, n_components)
            The squared cosinus of the individuals.
        infos : DataFrame of shape (n_samples, 4) 
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.
        coord_partiel : dictionary of DataFrame of shape (n_samples, n_components)
            Partiel coordinates of the individuals.
        within_inertia : DataFrame of shape of shape (n_samples, n_components)
            Within inertia of the individuals.
        within_partial_inertia: dictionary of DataFrame of shape (n_samples, n_components)
            Partial within inertia of the inertia.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, get_mfa_ind
    >>> clf = MFA(ncp=5,group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    MFA(group=(2,5,3,10,9,2),group_type=("n","s","s","s","s","s"),name_group = ("origin","odor","visual","odor.after.shaking","taste","overall"),ncp=5,num_group_sup=(0,5))
    >>> #extract the results for individuals
    >>> ind = get_mfa_ind(clf)
    >>> ind.coord.head() #coordinates of individuals
    >>> ind.contrib.head() #contributions of individuals
    >>> ind.cos2.head() #cos2 of individuals
    >>> ind.infos.head() #additionals informations of individuals
    """
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be an object of class MFA")
    return obj.ind_

def get_mfa_quanti_var(
        obj
):
    """
    Extract the results for quantitatie variables - MFA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active variables/groups/frequencies from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.
    
    Returns
    -------
    quanti_var_ : quanti_var
        An object containing all the results for the active quantitative variables, with the following attributes:

        coord : DataFrame of shape (n_columns, n_components)
            The coordinates of the quantitative variables.
        contrib : DataFrame of shape (n_columns, n_components)
            The relative contributions of the quantitative variables.
        cos2 : DataFrame of shape (n_columns, n_components)
            The squared cosinus of the quantitative variables.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, get_mfa_quanti_var
    >>> clf = MFA(ncp=5,group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    MFA(group=(2,5,3,10,9,2),group_type=("n","s","s","s","s","s"),name_group = ("origin","odor","visual","odor.after.shaking","taste","overall"),ncp=5,num_group_sup=(0,5))
    >>> #extract the results for quantitative variables
    >>> quanti_var = get_mfa_quanti_var(clf)
    >>> quanti_var.coord.head() #coordinates of quantitative variables
    >>> quanti_var.contrib.head() #contributions of quantitative variables
    >>> quanti_var.cos2.head() #cos2 of quantitative variables
    >>> quanti_var.infos.head() #additionals informations of quantitative variables
    """
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be an object of class MFA")
    return obj.quanti_var_
    
def get_mfa_quali_var(
        obj
):
    """
    Extract the results for qualitative variables/levels - MFA
   
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active qualitative variables/levels from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.

    Returns
    -------
    quali_var_ : quali_var
        An object containing all the results for the active qualitative variables/levels, with the following attributes:

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
        coord_partiel : dictionary of DataFrame of shape (n_samples, n_components)
            Partiel coordinates of the levels.
        
    Examples
    --------
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MFA, get_mfa_quali_var
    >>> clf = MFA(ncp=5,group=(2,2,5,6),group_type=("s","n","n","n"),name_group=poison.name,num_group_sup=(0,1))
    >>> mfa.fit(poison.data)
    MFA(group=(2,2,5,6),group_type=("s","n","n","n"),name_group=("desc","desc2","symptom","eat"),ncp=5,num_group_sup=(0,1))
    >>> #extract results for the qualitative variables
    >>> quali_var = get_mfa_var(clf) 
    >>> quali_var.coord.head() #coordinates of the levels
    >>> quali_var.coord_n.head() #normalized coordinates (barycentric) of the levels
    >>> quali_var.cos2.head() #cos2 of the levels
    >>> quali_var.contrib.head() #contributions of the levels
    >>> quali_var.vtest.head() #vtest of the levels
    >>> quali_var.coord_partiel.head() #partiel coordinates of the levels
    """
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be an object of class MFA")
    return obj.quali_var_
    
def get_mfa_group(
        obj
):
    """
    Extract the results for groups - MFA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active groups from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.
    
    Returns
    -------
    group_ : group
        An object containing all the results for the active groups, with the following attributes:

        coord : DataFrame of shape (n_groups, n_components)
            the coordinates of the groups.
        contrib : DataFrame of shape (n_groups, n_components)
            The relative contributions of the groups.
        cos2 : DataFrame of shape (n_groups, n_components)
            The squared cosinus of the groups.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, get_mfa_group
    >>> clf = MFA(ncp=5,group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    MFA(group=(2,5,3,10,9,2),group_type=("n","s","s","s","s","s"),name_group = ("origin","odor","visual","odor.after.shaking","taste","overall"),ncp=5,num_group_sup=(0,5))
    >>> #extract the results for groups
    >>> group = get_mfa_group(clf)
    >>> group.coord.head() #coordinates of groups
    >>> group.contrib.head() #contributions of groups
    >>> group.cos2.head() #cos2 of groups
    """
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be an object of class MFA")
    return obj.group_
    
def get_mfa_freq(
        obj
):
    """
    Extract the results for frequencies - MFA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active frequencies from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.MFA`.
    
    Returns
    -------
    freq_ : freq
        An object containing all the results for the active frequencies, with the following attributes:

        coord : DataFrame of shape (n_frequences, n_components)
            The coordinates of the frequencies.
        contrib : DataFrame of shape (n_frequences, n_components)
            The relative contributions of the frequencies.
        cos2 : DataFrame of shape (n_frequences, n_components)
            The squared cosinus of the frequencies.
   
    Examples
    --------
    >>> from scientisttools.datasets import mortality
    >>> from scientisttools import MFA, get_mfa_freq
    >>> clf = MFA(group=mortality.group,type_group=("f","f"),name_group=mortality.name)
    >>> clf.fit(mortality.data)
    MFA(group=(9,9),group_type=("f","f"),name_group = ("y1979","y2006"),ncp=5)
    >>> #extract the results for frequencies
    >>> freq = get_mfa_freq(clf)
    >>> freq.coord.head() #coordinates of frequencies
    >>> freq.contrib.head() #contributions of frequencies
    >>> freq.cos2.head() #cos2 of frequencies
    >>> freq.infos.head() #additionals informations of frequencies
    """
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'self' must be an object of class MFA")
    return obj.freq_
            
def get_mfa_partial_axes(
        obj
):
    """
    Extract the results for partial axes - MFA
    
    Extract all the results (coordinates, squared cosine and contributions) for the active partial axes from Multiple Factor Analysis (MFA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`.

    Returns
    -------
    partial_axes_ : partial_axes
        An object containing all the results for the partial axes (coordinates, correlation between variables and axes, correlation between partial axes)

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, get_mfa_group
    >>> clf = MFA(ncp=5,group=wine.group,group_type=("n","s","s","s","s","s"),name_group = wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    MFA(group=(2,5,3,10,9,2),group_type=("n","s","s","s","s","s"),name_group = ("origin","odor","visual","odor.after.shaking","taste","overall"),ncp=5,num_group_sup=(0,5))
    >>> #extract the results for partial axes
    >>> partial_axes = get_mfa_partial_axes(clf)
    """
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be an object of class MFA")
    return obj.partial_axes_

def get_mfa(
        obj, element = "group"
):
    """
    Extract the results for individuals/variables/groups/frequencies/partial axes - MFA
    
    Extract all the results (coordinates, squared cosinus and relative contributions) for the active individuals/variables/groups/frequencies/partial axes from Multiple Factor Analysis (MFA, MFAQUAL, MFAMIX, MFACT) outputs.
    
        * :class:`~scientisttools.get_mfa`: Extract the results for variables and individuals
        * :class:`~scientisttools.get_mfa_ind`: Extract the results for individuals only
        * :class:`~scientisttools.get_mfa_quanti_var`: Extract the results for quantitative variables only
        * :class:`~scientisttools.get_mfa_quali_var`: Extract the results for qualitative variables only
        * :class:`~scientisttools.get_mfa_group`: Extract the results for groups only
        * :class:`~scientisttools.get_mfa_freq`: Extract the results for frequencies only
        * :class:`~scientisttools.get_mfa_partial_axes`: Extract the results for partial axes only

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`, :class:`~scientisttools.MFACT`.

    element : {"ind", "quanti_var", "group", "quali_var", "freq", "partial_axes"}, default = "ind"
        The element to subset from the output.

    Returns
    -------
    result : ind/quanti_var/quali_var/group/freq/partial_axes
        An object containing the results for the active individuals, variables, groups, frequencies and partial axes.
    
    See also
    --------
    :class:`~scientisttools.get_mfa_ind`
        Extract the results for individuals - MFA.
    :class:`~scientisttools.get_mfa_quanti_var`
        Extract the results for quantitative variables - MFA.
    :class:`~scientisttools.get_mfa_quali_var`
        Extract the results for qualitative variables/levels - MFA.
    :class:`~scientisttools.get_famd_freq`
        Extract the results for frequencies - MFA.
    :class:`~scientisttools.get_famd_group`
        Extract the results for groups - MFA.
    :class:`~scientisttools.get_famd_partial_axes`
        Extract the results for partial axes - MFA.
    """
    if element == "ind":
        return get_mfa_ind(obj)
    elif element == "quanti_var":
        return get_mfa_quanti_var(obj)
    elif element == "quali_var":
        return get_mfa_quali_var(obj)
    elif element == "freq":
        return get_mfa_freq(obj)
    elif element == "group":
        return get_mfa_group(obj)
    elif element == "partial_axes":
        return get_mfa_partial_axes(obj)
    else:
        raise ValueError("'element' should be one of 'ind', 'quanti_var', 'group', 'quali_var', 'freq','partial_axes'")