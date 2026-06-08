# -*- coding: utf-8 -*-

def get_ca_row(
        obj
):
    """
    Extract the results for rows - CA

    Extract all the results (coordinates, square cosinus, relative contributions and additionals informations) for the active rows from Correspondence Analysis (CA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    Returns
    -------
    row_ : row
        An object containing all the results for the active rows with the following attributes:

        coord : DataFrame of shape (n_rows, n_components)
            The coordinates for the active rows.
        cos2 : DataFrame of shape (n_rows, n_components)
            The squared cosinus for the active rows,
        contrib : DataFrame of shape (n_rows, n_components) 
            The relative contributions for the active rows,
        infos : DataFrame of shape (n_rows, 5)
            Additionnals informations (weights, margin, square distance to origin, inertia and percentage of inertia) for the active rows. 

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, get_ca_row
    >>> clf = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> #extract the results of rows
    >>> row = get_ca_row(clf)
    >>> row.coord.head() #row coordinates
    >>> row.cos2.head() #rows cos2
    >>> row.conrib.head() #row contributions
    >>> row.infos.head() #row additionals informations
    """
    if obj.__class__.__name__ != "CA":
        raise TypeError("'obj' must be an object of class CA")
    return obj.row_
            
def get_ca_col(
        obj
):
    """
    Extract the results for columns - CA

    Extract all the results (coordinates, square cosinus, relative contributions and additionals informations) for the active columns from Correspondence Analysis (CA) outputs.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    Returns
    -------
    col_ : col
        An object containing all the results for the active columns with the following attributes: 

        coord : DataFrame of shape (n_columns, n_components)
            The coordinates for the active columns.
        cos2 : DataFrame of shape (n_columns, n_components)
            The square cosinus for the active columns.
        contrib : DataFrame of shape (n_columns, n_components)
            The relative contributions for the active columns.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (margin, square distance to origin, inertia and percentage of inertia) for the active columns.
    
    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, get_ca_col
    >>> clf = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> #extract the results of columns
    >>> col = get_ca_col(clf)
    >>> col.coord.head() #column coordinates
    >>> col.cos2.head() #column cos2
    >>> col.contrib.head() #column contributions
    >>> col.infos.head() #column additionals informations
    """
    if obj.__class__.__name__ != "CA": 
        raise TypeError("'obj' must be an object of class CA")
    return obj.col_

def get_ca(
        obj, element = "row"
):
    """
    Extract the results for rows/columns - CA

    Extract all the results (coordinates, square cosinus, relative contributions and additionals informations) for the active rows/columns from Correspondence Analysis (CA) outputs.

        - :class:`~scientisttools.get_ca`: Extract the results for rows and columns.
        - :class:`~scientisttools.get_ca_row`: Extract the results for rows only.
        - :class:`~scientisttools.get_ca_col`: Extract the results for columns only.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.CA`.

    element : {"col","row"}, default = "row"
        The element to subset from the output.

    Returns
    -------
    result : row/col
        An object containing all the results for the active rows/columns with the following attributes:

        coord : DataFrame of shape (n_rows/n_columns, n_components)
            The coordinates for the active rows/columns.
        cos2 : DataFrame of shape (n_rows/n_columns, n_components)
            The square cosinus for the active rows/columns.
        contrib : DataFrame of shape (n_rows/n_columns, n_components)
            The relative contributions of the active rows/columns.
        infos : DataFrame of shape (n_rows/n_columns, 5/4) 
            Additionals informations of the active rows/columns.

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, get_ca
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    >>> #extract the results of rows
    >>> row = get_ca(clf, element = "row")
    >>> row.coord.head() #rows coordinates
    >>> row.cos2.head() #rowss cos2
    >>> row.conrib.head() #rows contributions
    >>> row.infos.head() #rows additionals informations
    >>> #extract the results of columns
    >>> col = get_ca(clf, element = "col")
    >>> col.coord.head() #columns coordinates
    >>> col.cos2.head() #columns cos2
    >>> col.contrib.head() #columns contributions
    >>> col.infos.head() #columns additionals informations
    """
    if element == "row":
        return get_ca_row(obj)
    elif element == "col":
        return get_ca_col(obj)
    else:
        raise ValueError("'element' should be one of 'row', 'col'")