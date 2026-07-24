# -*- coding: utf-8 -*-
from pandas import concat, DataFrame

# interns functions
from ._convexhull import convexhull
from ._ellipse import ellipse

def data_ellipse(X, 
                 ellipse_type = "confidence",
                 axis = [0,1], 
                 level = 0.95, 
                 npoints = 100, 
                 bary = False):
    """
    Construct ellipses

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_components + 1)
        Input data containing containing the coordinates of the individuals for which the ellipses are constructed. 
        This data can contain more than 2 variables; the variables taken into account are chosen after. 
        The last column must be a categorical variable which allows to associate one row to an ellipse.  

    ellipse_type : {"confidence","convex"}, default = "confidence"
        The frame type. Possible values are : "convex", "confidence".

        * "convex": plot convex hull of a set of points as :class:`~scientisttools.convexhull`.
        * "confidence": plot confidence ellipses around group mean points as :class:`~scientisttools.ellipse`.

    axis : list, default = [0,1]
        The dimensions that are taken into account for which the confidence ellipses are constructed.
    
    level : float, default = 0.95
        Confidence level used to construct the ellipses.

    npoints : int, default = 100
        Number of points used to draw the ellipses.

    bary : bool, default = False
        If bary = True, the coordinates of the ellipse around the barycentre of individuals are calculated

    Returns
    -------
    values : DataFrame of shape (n_values, 3)
        The last column is the categorical variable of X, the two others columns give the coordinates of the ellipses on the two dimensions chosen.    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if number of columns if more than 3
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.shape[1] < 3:
        raise TypeError("Number of columns must at least be 3.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check valid ellipse_type
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (ellipse_type in ("confidence","convex")):
        raise ValueError("ellipse_type should be one 'confidence','convex'")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if axis is an instance of list
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")

    habillage = X.columns[-1]
    y, X = X[habillage].astype("category"), X.iloc[:,axis]
    index = sorted(y.unique())
    index_dict = {k : y[y==k].index.tolist() for k in index}
    values = DataFrame().astype("float")
    for k, r in index_dict.items():
        # confidence ellipse
        if ellipse_type == "confidence":
            data = ellipse(X=X.loc[r,:],level=level,npoints=npoints,bary=bary)
        # convex ellipse
        else:
            data = convexhull(X=X.loc[r,:])
        # insert habillage category
        data.loc[:,habillage] = k
        # concatenate
        values = concat((values,data),axis=0,ignore_index=True)
    # convert to category
    values[habillage] = values[habillage].astype("category")
    return values