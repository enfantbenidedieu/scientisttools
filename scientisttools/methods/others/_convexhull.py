# -*- coding: utf-8 -*-
from pandas import DataFrame
from scipy.spatial import ConvexHull

def convexhull(X):
    """
    Convex hull model

    Parameters
    ----------
    X : DataFrame of shape (n_samples, 2)
        Input data containing containing the coordinates of the individuals for which the confidence ellipses are constructed. 

    Returns
    -------
    values : DataFrame of shape (n_hulls, 2)
        Matrix of the points forming the border of the ellipse.
    """
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if DataFrame
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if number of columns if more than 3
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.shape[1] != 2:
        raise TypeError("Number of columns must be equal to 2.")
    
    if X.shape[0] < 3:
        return X
    hull = ConvexHull(X.values)
    return X.iloc[hull.vertices]