# -*- coding: utf-8 -*-
from scipy import stats
from numpy import prod, sqrt,c_,array,empty
from pandas import DataFrame
from collections import namedtuple, OrderedDict

#likelihood ratio test
def lrtest(
        rho,n_samples,n_xcols,n_ycols
):
    """
    Likelihood ratio test

    Performs likelihood ratio test.

    Parameters
    ----------
    rho : 1d array-like of shape (s,) with s = min(p,q)
        Canonical correlations.
    n_samples : int
        Samples size.
    n_xcols : int
        Number of columns of ``X``.
    n_ycols : int
        Number of columns of ``Y``.
    
    Returns
    -------
    result : lrtestResult
        An object with the following attributes:

        header : str
            The test header.
        statistic : DataFrame of shape (1, 5)
            Likelihood ratio test, which is Wilks' Lambda, using F-approximation (Rao's F).
    """
    s = min(n_xcols,n_ycols)
    value = array([prod((1-rho**2)[i:s]) for i in range(s)])
    fvalue, dof1, dof2, pvalue = empty((s,),dtype=float), empty((s,),dtype=float), empty((s,),dtype=float), empty((s,),dtype=float)
    de = n_samples - 1.5 - 0.5*(n_xcols + n_ycols)
    for i in range(s):
        dof1[i] = (n_xcols - i)*(n_ycols - i)
        nu = 1 if (n_xcols - i) == 1 or (n_ycols - i) == 1 else sqrt((dof1[i]**2-4)/((n_xcols - i)**2+(n_ycols - i)**2-5))
        dof2[i] = de*nu - dof1[i]/2 + 1
        fvalue[i] = (dof2[i]/dof1[i])*((1-(value[i]**(1/nu)))/(value[i]**(1/nu)))
        pvalue[i] = stats.f.sf(fvalue[i],dof1[i],dof2[i])
    #convert to DataFrame
    statistic = DataFrame(c_[value,fvalue,dof1,dof2,pvalue],columns=["Likelihood Ratio","Approximate F value","Num DF","Den DF","Pr>F"],index=[f"Can{x+1}" for x in range(s)])
    statistic["Num DF"] = statistic["Num DF"].astype(int)
    #header
    header = "\nTest of H0: The canonical correlations in the current row and all that follow are zero\nWilks' Lambda, using F-approximation (Rao's F)"
    return namedtuple("lrtestResult",["header","statistic"])(header,statistic)

def hotelling_test(
        rho,n_samples,n_xcols,n_ycols
):
    """
    Hotelling-Lawley Trace test

    Parameters
    ----------
    rho : 1d array-like of shape (s,) with s = min(p,q)
        Canonical correlations.
    n_samples : int
        Samples size.
    n_xcols : int
        Number of columns of ``X``.
    n_ycols : int
        Number of columns of ``Y``.
    
    Returns
    -------
    result : hotellingtestResult
        An object with the following attributes:

        header : str
            The test header.
        statistic : DataFrame of shape (1, 5)
            Hotelling-Lawley Trace, using F-approximation.
    """
    s = min(n_xcols,n_ycols)
    value = array([sum((rho**2/(1-rho**2))[i:s]) for i in range(s)])
    fvalue, dof1, dof2, pvalue = empty((s,),dtype=float), empty((s,),dtype=float), empty((s,),dtype=float), empty((s,),dtype=float)
    for i in range(s):
        dof1[i], dof2[i] = (n_xcols - i)*(n_ycols - i), s*(n_samples  - 2 - n_xcols - n_ycols + 2*i) + 2
        fvalue[i] = value[i]/s/dof1[i]*dof2[i]
        pvalue[i] = stats.f.sf(fvalue[i],dof1[i],dof2[i])
    #convert to DataFrame
    statistic = DataFrame(c_[value,fvalue,dof1,dof2,pvalue],columns=["Value","Approximate F value","Num DF","Den DF","Pr>F"],index=[f"Can{x+1}" for x in range(s)])
    statistic[["Num DF","Den DF"]] = statistic[["Num DF","Den DF"]].astype(int)
    #statistics test header
    header = "\nHotelling-Lawley Trace, using F-approximation"
    return namedtuple("hotellingtestResult",["header","statistic"])(header,statistic)

def pillai_test(
        rho,n_samples,n_xcols,n_ycols
):
    """
    Pillai's Bartlett Trace test

    Parameters
    ----------
    rho : 1d array-like of shape (s,) with s = min(p,q)
        Canonical correlations.
    n_samples : int
        Samples size.
    n_xcols : int
        Number of columns of ``X``.
    n_ycols : int
        Number of columns of ``Y``.
    
    Returns
    -------
    result : pillaitestResult
        An object with the following attributes:

        header : str
            The test header.
        statistic : DataFrame of shape (1, 5)
            Pillai-Bartlett Trace, using F-approximation.
    """
    s = min(n_xcols,n_ycols)
    value = array([sum((rho**2)[i:s]) for i in range(s)])
    fvalue, dof1, dof2, pvalue = empty((s,),dtype=float), empty((s,),dtype=float), empty((s,),dtype=float), empty((s,),dtype=float)
    for i in range(s):
        dof1[i], dof2[i]  = (n_xcols - i)*(n_ycols - i), s*(n_samples - 1 + s - n_xcols - n_ycols + 2*i) 
        fvalue[i] = value[i] / dof1[i] * dof2[i] / (s - value[i])
        pvalue[i] = stats.f.sf(fvalue[i],dof1[i],dof2[i])
    #convert to DataFrame
    statistic = DataFrame(c_[value,fvalue,dof1,dof2,pvalue],columns=["Value","Approximate F value","Num DF","Den DF","Pr>F"],index=[f"Can{x+1}" for x in range(s)])
    statistic[["Num DF","Den DF"]] = statistic[["Num DF","Den DF"]].astype(int)
    #statistics test header
    header = "\nPillai-Bartlett Trace, using F-approximation"
    return namedtuple("pillaitestResult",["header","statistic"])(header,statistic)

def roy_test(
        rho,n_samples,n_xcols,n_ycols
):
    """
    Roy's Greatest Root test

    Parameters
    ----------
    rho : 1d array-like of shape (s,) with s = min(p,q)
        Canonical correlations.
    n_samples : int
        Samples size.
    n_xcols : int
        Number of columns of ``X``.
    n_ycols : int
        Number of columns of ``Y``.
    
    Returns
    -------
    result : roytestResult
        An object with the following attributes:

        header : str
            The test header.
        statistic : DataFrame of shape (1, 5)
            Roy's Largest Root, using F-approximation
    """
    value = rho[0]**2
    dof1,dof2 = n_xcols, n_samples - 1 - n_ycols
    fvalue = value/dof1 * dof2 /(1 - value)
    pvalue = stats.f.sf(value,dof1,dof2)
    #convert to DataFrame
    statistic = DataFrame([[value,fvalue,dof1,dof2,pvalue]],columns=["Value","Approximate F value","Num DF","Den DF","Pr>F"])
    statistic[["Num DF","Den DF"]] = statistic[["Num DF","Den DF"]].astype(int)
    #statistic test header
    header = "\nRoy's Largest Root, using F-approximation"
    return namedtuple("roytestResult",["header","statistic"])(header,statistic)    