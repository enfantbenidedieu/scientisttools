# -*- coding: utf-8 -*-
from numpy import ndarray
from pandas import DataFrame
from collections import OrderedDict

def predict_sup(X,Y,weights,axis=0) -> OrderedDict:
    """
    Predict supplementary elements (rows/columns)
    ---------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin for new elements (rows/columns) with generalized factor analysis

    Usage
    -----
    ```python
    >>> predict_sup(X,Y,weights,axis)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of standardized data

    `Y`: a 2D numpy array of the right/left matrix of generalized singular value decomposition (GSVD)

    `weights`: a pandas Series of weights (rows/columns)

    `axis`: None or a string or an integer indicating which axis to aggregate
        * None or 0 or "index" indicates aggregating along rows
        * 1 or "columns" indicates aggregating along columns

    Return(s)
    ---------
    an ordered dictionary of pandas DataFrames/Series containing all the results for the supplementary elements, including:
    
    `coord`: coordinates of the supplementary rows/columns,

    `cos2`: squared cosinus of the supplementary rows/columns,

    `dist2`: squared distance to origin of the supplementary rows/columns.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com   
    """
    if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    if not isinstance(Y,ndarray): #check if Y is an instance of class np.array
        raise TypeError(f"{type(Y)} is not supported. Please convert to a 2D array with np.array. For more information see: https://numpy.org/devdocs//reference/generated/numpy.array.html")

    if axis in [None, 0, "index"]:
        #coordinates of the new rows
        coord = X.mul(weights,axis=1).dot(Y)
        #cos2 of the new rows
        sqdisto = X.pow(2).mul(weights,axis=1).sum(axis=1)
        #cos2 of the new rows
        sqcos = coord.pow(2).div(sqdisto,axis=0)
    elif axis in [1, "columns"]:
        #coordinates of the new columns
        coord = X.mul(weights,axis=0).T.dot(Y)
        #cos2 of the new columns
        sqdisto = X.pow(2).mul(weights,axis=0).sum(axis=0)
        #cos2 of the new columns
        sqcos = coord.pow(2).div(sqdisto,axis=0)
    else:
        raise ValueError("'axis' must be either index (0) or columns (1).")
    
    #set columns and names
    coord.columns, sqcos.columns, sqdisto.name  = ["Dim."+str(x+1) for x in range(coord.shape[1])], ["Dim."+str(x+1) for x in range(sqcos.shape[1])], "Sq. Dist."

    #convert to ordered dictionary
    return OrderedDict(coord=coord,cos2=sqcos,dist2=sqdisto)