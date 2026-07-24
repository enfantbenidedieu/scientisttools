# -*- coding: utf-8 -*-
from numpy import mean, cov, sqrt,cos,arccos,pi,linspace,c_
from pandas import DataFrame
from scipy.stats import chi2

def ellipse(X,
            level = 0.95,
            npoints = 100,
            bary = False):
    """
    Draw Two-Dimensional Ellipse

    Draw a two-dimensional ellipse that traces a bivariate normal density contour for a given data X.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, 2)
        Input data containing containing the coordinates of the individuals for which the confidence ellipses are constructed. 

    level : float, default = 0.95
        Confidence level used to construct the ellipses.

    npoints : int, default = 100
        Number of points used to draw the ellipses.

    bary : bool, default = False
        If bary = True, the coordinates of the ellipse around the barycentre of individuals are calculated

    Returns
    -------
    values : DataFrame of shape (npoints,2)
        Matrix of the points forming the border of the ellipse.
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
    if X.shape[1] != 2:
        raise TypeError("Number of columns must be equal to 2.")
    
    # average
    mu = mean(X,axis=0).values
    # covariance
    covar = cov(m=X,rowvar=False,ddof=1)
    if bary:
        covar = covar/X.shape[0]
    r = covar[0,1]
    sigma = sqrt([covar[0,0],covar[1,1]])
    if sigma[0] > 0:
        r = r/sigma[0]
    if sigma[1] > 0:
        r = r/sigma[1]
    r = min(max(r,-1),1)
    d, a, t = arccos(r), linspace(0,2*pi,npoints), sqrt(chi2.ppf(level,2))
    x = t*sigma[0]*cos(a + (d/2)) + mu[0]
    y = t*sigma[1]*cos(a - (d/2)) + mu[1]
    values = DataFrame(c_[x,y],columns=X.columns)
    return values